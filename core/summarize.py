from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yake

def get_llm():
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.1,
        max_tokens = 350
    )


# Split transcript into manageable chunks
def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=80,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Summarize this meeting transcript chunk.

Return ONLY concise bullet points.

Include:
- Key discussion points
- Decisions made
- Action items

Avoid repetition.
Maximum 6 bullet points.""",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    # Parallel summarization
    chunk_summaries = map_chain.batch(
        [{"text": chunk} for chunk in chunks]
    )

    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Combine these partial meeting summaries.

Requirements:
- Remove duplicate information.
- Merge similar points.
- Keep only the important information.
- Maximum 10 bullet points.
- Professional language.""",
            ),
            ("human", "{text}"),
        ]
    )

    reduce_chain = reduce_prompt | llm | StrOutputParser()

    return reduce_chain.invoke({"text": combined})



def generate_title(summary: str) -> str:
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        top=10
    )

    keywords = kw_extractor.extract_keywords(summary)

    for keyword, score in keywords:

        # Ignore very short phrases
        if len(keyword.split()) >= 2:
            return keyword.title()

    return "Meeting Transcript"
# def generate_title(summary: str) -> str:
#     llm = get_llm()

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """Generate a professional meeting title.

# Rules:
# - Maximum 8 words.
# - Return ONLY the title.
# - No quotation marks.
# - No explanation.""",
#             ),
#             ("human", "{text}"),
#         ]
#     )

#     chain = prompt | llm | StrOutputParser()

#     return chain.invoke({"text": summary})