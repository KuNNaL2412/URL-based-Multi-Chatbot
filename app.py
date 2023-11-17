import os
import pickle
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

with st.sidebar:
        st.title('🗨️ URL Based Chatbot')
        st.markdown("## Conversation History: ")

        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
        
        # New Chat button
        if "active_session" not in st.session_state or st.sidebar.button("New Chat +", use_container_width=True):
            # Create a new chat session and set it as active
            chat_id = len(st.session_state.chat_sessions) + 1
            session_key = f"Chat {chat_id}"
            st.session_state.chat_sessions[session_key] = []
            st.session_state.active_session = session_key

        # Buttons for previous chat sessions
        for session in st.session_state.chat_sessions:
            if st.sidebar.button(session, key=session):
                st.session_state.active_session = session
        st.markdown('''
        ## About App:

        The app's primary resource is utilised to create:

        - [Streamlit](https://streamlit.io/)
        - [Langchain](https://docs.langchain.com/docs/)
        - [OpenAI](https://openai.com/)

        ## About me:

        - [Linkedin](https://www.linkedin.com/in/kunal-pamu-710674230/)
        
        ''')
        st.write("Made by Kunal Shripati Pamu")

def main():
    # Api Key Input
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        # Url user input
        st.header("Chat with your URL")
        url = st.text_input("Enter the URL: ")

        if url:
            # extract text from the URL
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()

            # split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)
            
            # Open AI embeddings and vector store 
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            # Open AI LLM and initializing a Conversation Chain using langchain
            llm = OpenAI(temperature=0)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())

            if "active_session" in st.session_state:
                for message in st.session_state.chat_sessions[st.session_state.active_session]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Read user input prompt
            query = st.chat_input("Ask your questions from PDF ")

            if query:
                # using chat message to initiate User conversation
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                # Generate response using qa chain with the help of query and previous messages
                result = qa_chain({"question": query, "chat_history": [(message["role"], message["content"]) for message in st.session_state.chat_sessions[st.session_state.active_session]]})
                response = result["answer"]

                # using chat message to initiate Bot conversation
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_sessions[st.session_state.active_session].append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()