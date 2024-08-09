import streamlit as st
from langchain.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import pickle
from configure import groq_api_key
import os

st.title("PDF Q&A with LangChain")

# Load the precomputed FAISS index and documents
embeddings_file = "./embeddings.pkl"

if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        db = pickle.load(f)

    # Initialize the LLM using ChatGroq
    GROQ_API_KEY = groq_api_key
    llm = ChatGroq(temperature=0, model_name='llama3-70b-8192', groq_api_key=GROQ_API_KEY)

    # Initialize conversation memory
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=2,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
        )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(),
        return_source_documents=True,
        memory = st.session_state.memory,
    ) 

    # Query input from the user
    query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        # Prepare the inputs
        response = qa_chain({'question': query, 'chat_history': st.session_state.chat_history})
        user_answer = response['answer']

        # Update chat history
        st.session_state.chat_history.append((query, user_answer))

        # Display the answer
        st.write("**Answer:**", user_answer)

    # Display the chat history
    if st.session_state.chat_history:
        st.write("## Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {answer}")
            st.write("---")
else:
    st.write("Embeddings file not found. Please process the PDFs first.")
