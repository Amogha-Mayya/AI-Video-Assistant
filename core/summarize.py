from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yake


def get_map_llm():
    # Per-chunk summaries — short bullets only, hard cap at 150 tokens
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, max_tokens=150)


def get_reduce_llm():
    # Final combined summary — slightly more room
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, max_tokens=300)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=80,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str, language: str = "english") -> str:
    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this meeting chunk as bullet points. "
                   "Include key points, decisions, action items. "
                   "No repetition. Max 5 bullets."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | get_map_llm() | StrOutputParser()
    chunk_summaries = map_chain.batch([{"text": c} for c in chunks])
    combined = "\n\n".join(chunk_summaries)

    if language == "hindi":
        reduce_system = (
            "इन आंशिक सारांशों को मिलाएं। "
            "डुप्लीकेट हटाएं, समान बिंदु मिलाएं, केवल महत्वपूर्ण जानकारी रखें। "
            "हिंदी में उत्तर दें। अधिकतम 8 बुलेट पॉइंट्स।"
        )
    else:
        reduce_system = (
            "Combine these partial summaries. Remove duplicates, merge similar points, "
            "keep only important info. Max 8 bullet points. Professional language."
        )

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", reduce_system),
        ("human", "{text}"),
    ])
    reduce_chain = reduce_prompt | get_reduce_llm() | StrOutputParser()
    return reduce_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
    keywords = kw_extractor.extract_keywords(transcript)
    for keyword, score in keywords:
        if len(keyword.split()) >= 2:
            return keyword.title()
    return "Meeting Transcript"
