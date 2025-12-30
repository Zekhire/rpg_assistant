import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from settings import CHROMA_DIR, MODEL, EMBEDDING_MODEL

# Format docs WITH source + page
def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"(Source: {d.metadata['source']}, Page: {d.metadata['page'] + 1})\n{d.page_content}"
        for d in docs
    )

st.set_page_config(page_title="Ollama RPG Assistant", layout="wide")
st.title("Ollama RPG Assistant Chat (RAG)")

# Load vector store
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

template = """
You are an expert assistant helping users find information about the content of the RPG rule book.

Here is some relevant context from the document: {context}

Using the provided context, answer the following question: {question}
"""

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Load LLM
model = OllamaLLM(model=MODEL)
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
)

# Chat UI
question = st.text_input("Ask a question about the PDF")

if question:
    with st.spinner("Thinking..."):
        result = chain.invoke({"question": question})
        docs = retriever.invoke(question)
        st.markdown("### Answer")
        st.write(result)

        st.markdown("### Sources")
        seen = set()
        for d in docs:
            key = (d.metadata["source"], d.metadata["page"])
            if key not in seen:
                seen.add(key)
                st.write(f"- {d.metadata['source']} — page {d.metadata['page'] + 1}")