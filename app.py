import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
import base64
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def get_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If answer is not in the context, say "Answer not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # model = ChatOpenAI(model_name="gpt-4", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def chat_with_pdf(user_question, vector_store, conversation_history):
    docs = vector_store.similarity_search(user_question, k=5)
    chain = get_qa_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conversation_history.append({"question": user_question, "answer": answer, "timestamp": timestamp})
    return answer


def display_conversation(conversation_history):
    for turn in conversation_history:
        st.markdown(f"<div style='margin-bottom:8px;'><b>You:</b> {turn['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-bottom:15px; color:#555;'><b>Bot:</b> {turn['answer']}</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with PDFs (HuggingFace+OpenAI)", page_icon=":books:")
    st.title("Chat with PDFs using Hugging Face Embeddings & OpenAI")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    pdf_docs = st.file_uploader("Upload PDF files (multiple)", accept_multiple_files=True, type=['pdf'])

    if pdf_docs:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs and indexing..."):
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                vector_store = get_vector_store(chunks)
            st.success("PDFs processed and indexed. You can start asking questions now.")
            st.session_state.vector_store = vector_store

    if "vector_store" in st.session_state:
        user_question = st.text_input("Ask a question about the uploaded PDFs:")
        if user_question:
            with st.spinner("Generating answer..."):
                answer = chat_with_pdf(user_question, st.session_state.vector_store, st.session_state.conversation_history)
            display_conversation(st.session_state.conversation_history)
    
    # Option to download conversation history
    if st.session_state.conversation_history:
        df_data = [
            [turn["question"], turn["answer"], turn["timestamp"]]
            for turn in st.session_state.conversation_history
        ]
        import pandas as pd
        df = pd.DataFrame(df_data, columns=["Question", "Answer", "Timestamp"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">Download conversation history as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
