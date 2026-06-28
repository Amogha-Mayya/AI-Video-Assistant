from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

import wave
import os

path = "downloads/RAG Explained in 12 Minutes.wav_chunk_1.wav"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(path, "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
    )

print(result.text)