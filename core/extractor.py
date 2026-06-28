from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, max_tokens=600)

def extract_all(transcript: str) -> dict:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze this meeting transcript and extract three things.
Use EXACTLY these section headers:

##ACTION_ITEMS
Numbered list of tasks with owner and deadline (write 'Not specified' if missing).
If none, write 'No action items found.'

##KEY_DECISIONS
Numbered list of decisions made.
If none, write 'No key decisions found.'

##OPEN_QUESTIONS
Numbered list of unresolved questions or follow-up topics.
If none, write 'No open questions found.'

Be concise. Max 6 points per section."""),
        ("human", "{transcript}"),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"transcript": transcript})
    return _parse(raw)


def _parse(raw: str) -> dict:
    sections = {
        "action_items":   "No action items found.",
        "key_decisions":  "No key decisions found.",
        "open_questions": "No open questions found.",
    }
    markers = {
        "##ACTION_ITEMS":   "action_items",
        "##KEY_DECISIONS":  "key_decisions",
        "##OPEN_QUESTIONS": "open_questions",
    }

    current_key = None
    buffer = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped in markers:
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = markers[stripped]
            buffer = []
        else:
            if current_key:
                buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    return sections