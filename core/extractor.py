from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import os


def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, max_tokens=480)


# ── English extraction ────────────────────────────────────────────────────────

def _extract_english(transcript: str) -> dict:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Extract from this meeting transcript. Output ALL three sections.\n"
         "Use EXACTLY these headers (no spaces, bold, or extra chars):\n\n"
         "##ACTION_ITEMS\n"
         "Numbered tasks with owner + deadline ('Not specified' if missing). Max 5.\n"
         "If none exist, infer the most likely next steps from context.\n\n"
         "##KEY_DECISIONS\n"
         "Numbered decisions or conclusions reached. Max 5.\n"
         "If none explicit, infer from what was agreed or settled.\n\n"
         "##OPEN_QUESTIONS\n"
         "Numbered unresolved topics or follow-ups needed. Max 5.\n"
         "If none explicit, list topics that need further discussion.\n\n"
         "Every section MUST have at least one item."),
        ("human", "{transcript}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"transcript": transcript})
    result = _parse(raw)

    # Targeted retry for any section that came back with default fallback text
    empty = [k for k, v in result.items() if v.startswith("No ") or not v.strip()]
    if empty:
        result = _retry_empty(transcript, result, empty)

    return result


def _retry_empty(transcript: str, existing: dict, missing_keys: list) -> dict:
    """Single targeted call for only the sections that failed to parse."""
    llm = get_llm()
    key_prompts = {
        "action_items":   "##ACTION_ITEMS\nList tasks or next steps (infer if needed). Min 1 item.",
        "key_decisions":  "##KEY_DECISIONS\nList decisions or conclusions (infer if needed). Min 1 item.",
        "open_questions": "##OPEN_QUESTIONS\nList unresolved questions or follow-ups (infer if needed). Min 1 item.",
    }
    sections_text = "\n\n".join(key_prompts[k] for k in missing_keys)
    snippet = transcript[:3000]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"From this meeting transcript, fill ONLY these sections:\n\n"
         f"{sections_text}\n\n"
         "Use the exact headers shown. Infer from context if not explicit."),
        ("human", "{transcript}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"transcript": snippet})
    retry = _parse(raw)

    for k in missing_keys:
        val = retry.get(k, "")
        if val and not val.startswith("No "):
            existing[k] = val

    return existing


# ── Hindi extraction via Sarvam AI ───────────────────────────────────────────

def _extract_hindi(transcript: str) -> dict:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return _extract_english(transcript)

    truncated = transcript[:6000] if len(transcript) > 6000 else transcript
    payload = {
        "model": "sarvam-m",
        "messages": [
            {
                "role": "system",
                "content": (
                    "मीटिंग ट्रांसक्रिप्ट से तीनों sections भरें। हर section में कम से कम एक item हो।\n\n"
                    "##ACTION_ITEMS\nकार्यों की सूची (मालिक और समय सीमा)। अधिकतम 5।\n\n"
                    "##KEY_DECISIONS\nमुख्य निर्णयों की सूची। अधिकतम 5।\n\n"
                    "##OPEN_QUESTIONS\nअनसुलझे प्रश्नों की सूची। अधिकतम 5।"
                ),
            },
            {"role": "user", "content": truncated},
        ],
        "max_tokens": 550,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse(raw, hindi=True)
    except Exception:
        return _extract_english(transcript)


# ── Parser — normalizes every LLM header variant ─────────────────────────────

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
        # Normalize so "## **Action_Items**", "###ACTION ITEMS", "**##action-items**"
        # all resolve to the canonical marker key
        normalized = (
            line.strip()
                .lstrip("#").strip()
                .strip("*").strip()
                .strip("_").strip()
                .upper()
                .replace(" ", "_")
                .replace("-", "_")
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