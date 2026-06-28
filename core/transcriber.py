# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
#     with open(chunk_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="gpt-4o-mini-transcribe",
#             file=audio_file,
#         )

#     return transcript.text


# # transcribe all the chunks one by one
# def transcribe_all(chunks: list, translate: bool = False) -> str:
#     full_transcript = ""

#     for i, chunk in enumerate(chunks):
#         print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
#         text = transcribe_chunk(chunk, translate=translate)

#         full_transcript += text + " "

#     print("Transcription complete.")

#     return full_transcript

# transcriber.py

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def transcribe_chunk(chunk_path: str) -> str:
    with open(chunk_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",  # cheaper than gpt-4o-transcribe
            file=audio_file,
        )
    return transcript.text


def transcribe_all(chunks, language: str = "english") -> str:
    # If chunks is already a plain string, it means YouTube transcript
    # was fetched directly — skip Whisper entirely
    if isinstance(chunks, str):
        return chunks

    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        full_transcript += transcribe_chunk(chunk) + " "

    print("Transcription complete.")
    return full_transcript.strip()