from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_map_llm():
    # Per-chunk: short bullets only, hard cap
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, max_tokens=120)


def get_reduce_llm():
    # Final combined summary
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, max_tokens=280)


def get_title_llm():
    # Title only — tiny cap, costs almost nothing
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.3, max_tokens=18)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=80)
    return splitter.split_text(transcript)


def summarize(transcript: str, language: str = "english") -> str:
    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Summarize this meeting chunk as bullet points. "
         "Key points, decisions, action items only. Max 4 bullets."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | get_map_llm() | StrOutputParser()
    chunk_summaries = map_chain.batch([{"text": c} for c in chunks])
    combined = "\n\n".join(chunk_summaries)

    if language == "hindi":
        reduce_sys = (
            "इन सारांशों को मिलाएं। डुप्लीकेट हटाएं। "
            "हिंदी में उत्तर दें। अधिकतम 7 बुलेट पॉइंट्स।"
        )
    else:
        reduce_sys = (
            "Merge these summaries. Remove duplicates. "
            "Max 7 bullet points. Professional tone."
        )

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", reduce_sys),
        ("human", "{text}"),
    ])
    reduce_chain = reduce_prompt | get_reduce_llm() | StrOutputParser()
    return reduce_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    # Only send the first 600 chars — enough to capture topic, minimal tokens
    snippet = transcript[:600]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Generate a short, specific meeting/video title (4-7 words). "
         "Return ONLY the title. No quotes, no punctuation at end."),
        ("human", "{text}"),
    ])
    chain = prompt | get_title_llm() | StrOutputParser()
    title = chain.invoke({"text": snippet}).strip().strip('"').strip("'")
    return title if title else "Meeting Recording"