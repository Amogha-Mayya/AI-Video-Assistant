from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import os
import yake


def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.1,
        max_tokens=350,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=80,
    )
    return splitter.split_text(transcript)


# ── English summarisation (existing map-reduce) ──────────────────────────────

def _summarize_english(transcript: str) -> str:
    llm = get_llm()
    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", """Summarize this meeting transcript chunk.
Return ONLY concise bullet points.
Include:
- Key discussion points
- Decisions made
- Action items
Avoid repetition. Maximum 6 bullet points."""),
        ("human", "{text}"),
    ])
    map_chain = map_chain = map_prompt | llm | StrOutputParser()
    chunk_summaries = map_chain.batch([{"text": c} for c in chunks])
    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", """Combine these partial meeting summaries.
- Remove duplicates, merge similar points.
- Keep only important information.
- Maximum 8 bullet points.
- Professional language."""),
        ("human", "{text}"),
    ])
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    return reduce_chain.invoke({"text": combined})


# ── Hindi summarisation via Sarvam AI ────────────────────────────────────────

def _summarize_hindi(transcript: str) -> str:
    """
    Uses Sarvam AI's chat completions endpoint to summarise in Hindi.
    Set SARVAM_API_KEY in your .env / Streamlit secrets.
    Docs: https://docs.sarvam.ai
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        # Graceful fallback: summarise in English if key missing
        return _summarize_english(transcript) + "\n\n_(Hindi summarisation unavailable — SARVAM_API_KEY not set)_"

    # Sarvam supports OpenAI-compatible chat completions
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Truncate very long transcripts to stay within Sarvam context limits
    truncated = transcript[:6000] if len(transcript) > 6000 else transcript

    payload = {
        "model": "sarvam-m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "आप एक विशेषज्ञ मीटिंग विश्लेषक हैं। "
                    "नीचे दी गई मीटिंग ट्रांसक्रिप्ट का सारांश हिंदी में बुलेट पॉइंट्स में दें। "
                    "केवल मुख्य चर्चा बिंदु, निर्णय और कार्य सूची शामिल करें। "
                    "अधिकतम 8 बुलेट पॉइंट्स।"
                ),
            },
            {"role": "user", "content": truncated},
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback to English on any Sarvam error
        return _summarize_english(transcript) + f"\n\n_(Hindi summarisation failed: {e})_"


# ── Public API ────────────────────────────────────────────────────────────────

def summarize(transcript: str, language: str = "english") -> str:
    if language == "hindi":
        return _summarize_hindi(transcript)
    return _summarize_english(transcript)


def generate_title(transcript: str) -> str:
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    keywords = kw_extractor.extract_keywords(transcript)
    for keyword, score in keywords:
        if len(keyword.split()) >= 2:
            return keyword.title()
    return "Meeting Transcript"
