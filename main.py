import os

import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Track the project through LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# init vector embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# sidebar
st.sidebar.title("Configurations")
uploaded_files = st.sidebar.file_uploader("Upload your PDFs", accept_multiple_files=True)

# main window
st.title('Document specific Q&A')
## Input the Groq API Key
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")

if uploaded_files and api_key:
    llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It")
    user_session_id = st.text_input("Session ID",value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # read the uploaded pdf and save it
    for file in uploaded_files:
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as ip:
            ip.write(file.getvalue())
            file_name = file.name
        
        loader = PyPDFLoader(temp_pdf)

        # load the document using document loader
        doc = loader.load()

        # split documents and save it to ChromaDB
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_docs = splitter.split_documents(doc)

        # save the split docs to Chroma after converting them to vectors
        vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")

        # vector db can't be used directly so use as_retriever() to access it
        retriever = vectordb.as_retriever()

        # create the question context and prompt
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, q_prompt)

        # create answer prompt
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        # connect the chains
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if user_session_id not in st.session_state.store:
                st.session_state.store[user_session_id] = ChatMessageHistory()
            return st.session_state.store[user_session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(user_session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":user_session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your GROQ API key.")
