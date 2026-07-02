from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper language codes
LANGUAGE_MAP = {
    "english": "en",
    "hindi":   "hi",
}


def transcribe_chunk(chunk_path: str, language_code: str = "en") -> str:
    with open(chunk_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language_code,  # explicit hint = faster + more accurate
        )
    return transcript.text


def transcribe_all(chunks: list, language: str = "english") -> str:
    language_code = LANGUAGE_MAP.get(language, "en")
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        full_transcript += transcribe_chunk(chunk, language_code) + " "

    print("Transcription complete.")
    return full_transcript.strip()
