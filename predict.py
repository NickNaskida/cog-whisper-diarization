# Prediction interface for Cog ⚙️
import base64
import datetime
import subprocess
import os

import requests
import time
import torch
import re

import torchaudio
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel, BatchedInferencePipeline
from cog import BasePredictor, BaseModel, Input, File, Path


class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v3"
        model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
        )
        self.model = BatchedInferencePipeline(model=model)

    def predict(
        self,
        file_string: str = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file: Path = Input(description="Or an audio file", default=None),
        hf_token: str = Input(
            default=None,
            description="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips. You need to agree to the terms in 'https://huggingface.co/pyannote/speaker-diarization-3.1' and 'https://huggingface.co/pyannote/segmentation-3.0' first.",
        ),
        group_segments: bool = Input(
            description="Group segments of same speaker shorter apart than 2 seconds",
            default=True,
        ),
        transcript_output_format: str = Input(
            description="Specify the format of the transcript output: individual words with timestamps, full text of segments, or a combination of both.",
            default="both",
            choices=["words_only", "segments_only", "both"],
        ),
        num_speakers: int = Input(
            description="Number of speakers, leave empty to autodetect.", ge=1, le=50, default=2,
        ),
        translate: bool = Input(
            description="Translate the speech into English.",
            default=False,
        ),
        language: str = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto detect language.",
            default=None,
        ),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
        batch_size: int = Input(
            description="Batch size for inference. (Reduce if face OOM error)",
            default=64,
            ge=1
        ),
        # word_timestamps: bool = Input(description="Return word timestamps", default=True), needs to be implemented
        offset_seconds: int = Input(
            description="Offset in seconds, used for chunked inputs", default=0, ge=0
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Check if either filestring, filepath or file is provided, but only 1 of them
        """ if sum([file_string is not None, file_url is not None, file is not None]) != 1:
            raise RuntimeError("Provide either file_string, file or file_url") """

        # Generate a temporary filename
        temp_wav_filename = f"temp-{time.time_ns()}.wav"

        try:
            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        file,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )
            elif file_url is not None:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as file:
                    file.write(response.content)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            elif file_string is not None:
                audio_data = base64.b64decode(
                    file_string.split(",")[1] if "," in file_string else file_string
                )
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)

            try:
                self.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                ).to(torch.device("cuda"))
            except Exception as e:
                print(f"https://huggingface.co/pyannote/speaker-diarization-3.1 cannot be loaded, please check the hf_token provided.: {e}")
                raise

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt,
                offset_seconds,
                group_segments,
                language,
                word_timestamps=True,
                transcript_output_format=transcript_output_format,
                translate=translate,
                batch_size=batch_size,
            )

            print(f"done with inference")
            # Return the results as a JSON object
            return Output(
                segments=segments,
                language=detected_language,
                num_speakers=detected_num_speakers,
            )

        except Exception as e:
            raise RuntimeError("Error Running inference with local model", e)
        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def simple_sent_tokenize(self, text):
        """
        A simple, language-agnostic sentence tokenizer.
        Splits on '.', '!', '?', ':', ';' and handles common abbreviations.
        """
        # Split on sentence-ending punctuation, but keep the punctuation
        sentences = re.split(r'([.!?:;])\s+', text)

        # Recombine sentence-ending punctuation with sentences
        sentences = [''.join(group) for group in zip(sentences[::2], sentences[1::2] + [''])]

        # Handle common abbreviations (Mr., Mrs., Dr., etc.)
        final_sentences = []
        current_sentence = ''
        for sentence in sentences:
            if re.search(r'\b[A-Z][a-z]?\.', sentence):
                current_sentence += sentence + ' '
            else:
                final_sentences.append(current_sentence + sentence)
                current_sentence = ''

        if current_sentence:
            final_sentences.append(current_sentence)

        return final_sentences

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        offset_seconds=0,
        group_segments=True,
        language=None,
        word_timestamps=True,
        transcript_output_format="both",
        translate=False,
        batch_size: int = 64,
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            language=language,
            task="translate" if translate else "transcribe",
            hotwords=prompt,
            batch_size=batch_size,
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start + offset_seconds),
                "end": float(s.end + offset_seconds),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start + offset_seconds),
                        "end": float(w.end + offset_seconds),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds, {len(segments)} segments"
        )

        print("Starting diarization")

        waveform, sample_rate = torchaudio.load(audio_file_wav)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        time_diraization_end = time.time()
        print(f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds")

        print("Starting merging")

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin
        min_speaker_duration = 0.5  # Minimum speaker duration in seconds
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {speaker for _, _, speaker in diarization_list}
        detected_num_speakers = len(unique_speakers)

        speaker_idx = 0
        n_speakers = len(diarization_list)

        for segment in segments:
            segment_start = round(segment["start"] + offset_seconds, 2)
            segment_end = round(segment["end"] + offset_seconds, 2)
            segment_text = []
            segment_words = []
            current_speaker = None
            speaker_changes = []

            for word in segment["words"]:
                word_start = round(word["start"] + offset_seconds - margin, 2)
                word_end = round(word["end"] + offset_seconds + margin, 2)
                word_assigned = False

                while speaker_idx < n_speakers and not word_assigned:
                    turn, _, speaker = diarization_list[speaker_idx]
                    turn_start = round(turn.start, 2)
                    turn_end = round(turn.end, 2)

                    if (turn_start <= word_start < turn_end) or (turn_start < word_end <= turn_end) or (
                            word_start <= turn_start and word_end >= turn_end):
                        if current_speaker != speaker:
                            if not speaker_changes or (word_start - speaker_changes[-1][0]) >= min_speaker_duration:
                                current_speaker = speaker
                                speaker_changes.append((word_start, speaker))

                        segment_text.append(word["word"])
                        word["word"] = word["word"].strip()
                        segment_words.append(word)
                        word_assigned = True

                    if turn_end <= word_end:
                        speaker_idx += 1
                    else:
                        break

            if segment_text:
                combined_text = " ".join(segment_text)
                cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()

                # Use our simple sentence tokenization
                sentences = self.simple_sent_tokenize(cleaned_text)
                sentence_boundaries = [0]
                current_length = 0
                for sentence in sentences:
                    current_length += len(sentence)
                    sentence_boundaries.append(current_length)

                # Merge speaker changes that occur mid-sentence
                merged_speaker_changes = []
                for i, (start, speaker) in enumerate(speaker_changes):
                    if i == 0 or (
                            merged_speaker_changes and start - merged_speaker_changes[-1][0] >= min_speaker_duration):
                        # Check if this change is near a sentence boundary
                        word_index = next(
                            (i for i, w in enumerate(segment_words) if w["start"] >= start - offset_seconds),
                            len(segment_words) - 1)
                        char_index = sum(len(w["word"]) + 1 for w in segment_words[:word_index])
                        if any(abs(char_index - boundary) < 10 for boundary in sentence_boundaries):
                            merged_speaker_changes.append((start, speaker))

                # Split the segment if there are significant speaker changes
                if len(merged_speaker_changes) > 1:
                    for i, (start, speaker) in enumerate(merged_speaker_changes):
                        end = merged_speaker_changes[i + 1][0] if i + 1 < len(merged_speaker_changes) else segment_end

                        sub_segment_words = [w for w in segment_words if
                                             start - offset_seconds <= w["start"] < end - offset_seconds]
                        sub_segment_text = " ".join([w["word"] for w in sub_segment_words])

                        new_segment = {
                            "avg_logprob": segment["avg_logprob"],
                            "start": round(start - offset_seconds, 2),
                            "end": round(end - offset_seconds, 2),
                            "speaker": speaker,
                            "text": sub_segment_text,
                            "words": sub_segment_words,
                        }
                        final_segments.append(new_segment)
                else:
                    new_segment = {
                        "avg_logprob": segment["avg_logprob"],
                        "start": round(segment_start - offset_seconds, 2),
                        "end": round(segment_end - offset_seconds, 2),
                        "speaker": merged_speaker_changes[0][1] if merged_speaker_changes else None,
                        "text": cleaned_text,
                        "words": segment_words,
                    }
                    final_segments.append(new_segment)

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds, {len(final_segments)} segments, {detected_num_speakers} speakers"
        )

        # Check if final_segments is empty
        if not final_segments:
            return [], detected_num_speakers, transcript_info.language

        print("Starting cleaning")
        segments = final_segments
        # Make output
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "speaker": segments[0]["speaker"],
            "avg_logprob": segments[0]["avg_logprob"],
        }

        if transcript_output_format in ("segments_only", "both"):
            current_group["text"] = segments[0]["text"]
        if transcript_output_format in ("words_only", "both"):
            current_group["words"] = segments[0]["words"]

        for i in range(1, len(segments)):
            # Calculate time gap between consecutive segments
            time_gap = segments[i]["start"] - segments[i - 1]["end"]

            # If the current segment's speaker is the same as the previous segment's speaker,
            # and the time gap is less than or equal to 2 seconds, group them
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 2 and group_segments:
                current_group["end"] = segments[i]["end"]
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] += " " + segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group.setdefault("words", []).extend(segments[i]["words"])
            else:
                # Add the current_group to the output list
                output.append(current_group)

                # Start a new group with the current segment
                current_group = {
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    "speaker": segments[i]["speaker"],
                    "avg_logprob": segments[i]["avg_logprob"],
                }
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] = segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group["words"] = segments[i]["words"]

        # Add the last group to the output list
        output.append(current_group)

        time_cleaning_end = time.time()
        print(
            f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return output, detected_num_speakers, transcript_info.language
