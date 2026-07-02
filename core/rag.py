from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever
from langchain_openai import ChatOpenAI


def get_llm():
    # Hard cap: chat answers should be concise, 250 tokens is plenty
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0.2, max_tokens=200)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=3)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using ONLY the meeting transcript context below. "
         "Be concise. If not found, say: "
         "'I could not find this information in the meeting transcript.' "
         "If quoting someone, mention them by name.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer
