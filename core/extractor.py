from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import os


def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, max_tokens=500)


# ── English extraction ────────────────────────────────────────────────────────

def _extract_english(transcript: str) -> dict:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract three things from this meeting transcript.\n"
         "Use EXACTLY these headers (no spaces, no bold, no extra chars):\n\n"
         "##ACTION_ITEMS\n"
         "Numbered list. Include owner and deadline ('Not specified' if missing). Max 6.\n"
         "If none: 'No action items found.'\n\n"
         "##KEY_DECISIONS\n"
         "Numbered list of decisions made. Max 6.\n"
         "If none: 'No key decisions found.'\n\n"
         "##OPEN_QUESTIONS\n"
         "Numbered list of unresolved questions or follow-ups. Max 6.\n"
         "If none: 'No open questions found.'"),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"transcript": transcript})
    return _parse(raw)


# ── Hindi extraction via Sarvam AI ───────────────────────────────────────────

def _extract_hindi(transcript: str) -> dict:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        result = _extract_english(transcript)
        result["_note"] = "_(Hindi extraction unavailable — SARVAM_API_KEY not set)_"
        return result

    truncated = transcript[:6000] if len(transcript) > 6000 else transcript

    payload = {
        "model": "sarvam-m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "मीटिंग ट्रांसक्रिप्ट से तीन चीजें निकालें। "
                    "EXACTLY इन headers का उपयोग करें:\n\n"
                    "##ACTION_ITEMS\n"
                    "कार्यों की क्रमांकित सूची (मालिक और समय सीमा)। अधिकतम 6।\n"
                    "यदि कोई नहीं: 'कोई कार्य सूची नहीं मिली।'\n\n"
                    "##KEY_DECISIONS\n"
                    "मुख्य निर्णयों की क्रमांकित सूची। अधिकतम 6।\n"
                    "यदि कोई नहीं: 'कोई मुख्य निर्णय नहीं मिला।'\n\n"
                    "##OPEN_QUESTIONS\n"
                    "अनसुलझे प्रश्नों की क्रमांकित सूची। अधिकतम 6।\n"
                    "यदि कोई नहीं: 'कोई खुले प्रश्न नहीं मिले।'"
                ),
            },
            {"role": "user", "content": truncated},
        ],
        "max_tokens": 600,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return _parse(raw, hindi=True)
    except Exception as e:
        result = _extract_english(transcript)
        result["_note"] = f"_(Hindi extraction failed: {e})_"
        return result


# ── Parser — normalizes headers before matching ───────────────────────────────

def _parse(raw: str, hindi: bool = False) -> dict:
    if hindi:
        sections = {
            "action_items":   "कोई कार्य सूची नहीं मिली।",
            "key_decisions":  "कोई मुख्य निर्णय नहीं मिला।",
            "open_questions": "कोई खुले प्रश्न नहीं मिले।",
        }
    else:
        sections = {
            "action_items":   "No action items found.",
            "key_decisions":  "No key decisions found.",
            "open_questions": "No open questions found.",
        }

    markers = {
        "ACTION_ITEMS":   "action_items",
        "KEY_DECISIONS":  "key_decisions",
        "OPEN_QUESTIONS": "open_questions",
    }

    current_key = None
    buffer = []

    for line in raw.splitlines():
        # Normalize: strip spaces, leading #, *, _ so variants like
        # "## **ACTION_ITEMS**", "###ACTION_ITEMS", "**##ACTION_ITEMS**"
        # all resolve to "ACTION_ITEMS"
        normalized = (
            line.strip()
                .lstrip("#")
                .strip()
                .strip("*")
                .strip("_")
                .strip()
                .upper()
        )

        if normalized in markers:
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = markers[normalized]
            buffer = []
        else:
            if current_key:
                buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

def extract_all(transcript: str, language: str = "english") -> dict:
    if language == "hindi":
        return _extract_hindi(transcript)
    return _extract_english(transcript)
