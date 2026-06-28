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


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        full_transcript += transcribe_chunk(chunk) + " "
    print("Transcription complete.")
    return full_transcript.strip()