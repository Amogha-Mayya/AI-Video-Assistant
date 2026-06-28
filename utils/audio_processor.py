# import yt_dlp 
# from pydub import AudioSegment
# import os

# DOWNLOAD_DIR = 'downloads'
# os.makedirs(DOWNLOAD_DIR,exist_ok = True)

# # download youtube video & store it in WAV format
# def download_youtube_audio(url :str) ->str:
#     output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
#     ydl_opts = {
#         "format": "bestaudio/best",
#         "outtmpl": output_path,
#         "postprocessors": [
#             {
#                 "key": "FFmpegExtractAudio",
#                 "preferredcodec": "wav",
#                 "preferredquality": "192",
#             }
#         ],
#         "quiet": True,
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
#     return filename


# # convert your audio to suitable format compatible with Whisper AI
# def convert_to_wav(input_path: str) -> str:
#     """Convert any audio/video file to WAV format using pydub."""
#     output_path = os.path.splitext(input_path)[0] + "_converted.wav"
#     audio = AudioSegment.from_file(input_path)
#     audio = audio.set_channels(1).set_frame_rate(16000) # mono voice & 16Khz
#     audio.export(output_path, format="wav")
#     return output_path


# # audio chunking (10 minutes chunking)
# def chunk_audio(wav_path : str , chunk_minutes : int = 2) -> list:
#     audio = AudioSegment.from_wav(wav_path)
#     chunk_ms = chunk_minutes * 60 * 1000 

#     chunks = []

#     for i, start in enumerate(range(0,len(audio),chunk_ms)):
#         chunk = audio[start : start + chunk_ms]
#         chunk_path = f"{wav_path}_chunk_{i}.wav"
#         chunk.export(chunk_path , format = "wav")

#         chunks.append(chunk_path)
    
#     return chunks

# # trigger function
# def process_input(source: str) -> list:
#     if source.startswith("http://") or source.startswith("https://"):
#         print("Detected YouTube URL. Downloading audio...")
#         wav_path = download_youtube_audio(source)
#     else:
#         print("Detected local file. Converting to WAV...")
#         wav_path = convert_to_wav(source)

#     print("Chunking audio...")
#     chunks = chunk_audio(wav_path)
#     print(f"Audio ready — {len(chunks)} chunk(s) created.")
#     return chunks

# audio_processor.py
import os
import re
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ── YouTube transcript (no download, no tokens) ──────────────────────────────

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL.")


# def get_youtube_transcript(url: str) -> str:
#     video_id = extract_video_id(url)
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)
#     return " ".join([entry["text"] for entry in transcript])

def get_youtube_transcript(url: str) -> str:
    video_id = extract_video_id(url)

    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)

    return " ".join(snippet.text for snippet in transcript)


# ── Local file handling ───────────────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 2) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


# ── Main entry point ─────────────────────────────────────────────────────────

def process_input(source: str) -> list | str:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Fetching transcript...")
        try:
            transcript = get_youtube_transcript(source)
            print("Transcript fetched directly — skipping Whisper.")
            # Return as string; transcriber.py will detect this and skip Whisper
            return transcript

        except TranscriptsDisabled:
            raise RuntimeError(
                "This video has transcripts disabled. "
                "Please download the video and upload the audio file instead."
            )
        except NoTranscriptFound:
            raise RuntimeError(
                "No transcript found for this video (no captions available). "
                "Please upload the audio file instead."
            )
        except Exception as e:
            raise RuntimeError(f"Could not fetch YouTube transcript: {e}")
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
        print("Chunking audio...")
        chunks = chunk_audio(wav_path)
        print(f"Audio ready — {len(chunks)} chunk(s) created.")
        return chunks