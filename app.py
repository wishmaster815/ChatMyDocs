# import os 
# import time
# from dotenv import load_dotenv
# import streamlit as st
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain.document_loaders import PyPDFLoader

# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# load_dotenv()

# os.environ['HF_KEY']=os.getenv("HF_KEY")
# embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



# st.title("CONVERSATIONAL CHATBOT WITH GROQ AND HUGGINGFACE")
# st.title("Upload api_key and pdf to start chatting with it")

# api_key = st.text_input("Enter your GROQ API key: ",type="password")

# if api_key:
#     llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
#     session_id = st.text_input("Enter Session ID", value="Default_session")

#     if 'store' not in st.session_state:
#         st.session_state.store = {}
#     uploaded_files = st.file_uploader("Upload pdf files",type="pdf",accept_multiple_files=True)
#     if uploaded_files:
#         documents = []
#         for i, uploaded_file in enumerate(uploaded_files):
#             temppdf = f"./temp_{i}.pdf"
#             with open(temppdf, "wb") as file:
#                 file.write(uploaded_file.getvalue())
#                 file_name=uploaded_file.name
#             loader = PyPDFLoader(temppdf)    
#             docs = loader.load()
#             documents.extend(docs)
#             os.remove(temppdf)
            
#         # splitting docs
#         splitter = RecursiveCharacterTextSplitter(chunk_size = 4000,chunk_overlap=300)
#         splits = splitter.split_documents(documents)
#         vectorstore = Chroma.from_documents(splits,embeddings)
#         retriever = vectorstore.as_retriever()

#         # contextual embedding feature
#         contextualize_q_system_prompt = (
#         "Given a chat history and the latest user question"
#         "which might reference context in the chat history, "
#         "formulate a standalone question which can be understood "
#         "without the chat history. Do NOT answer the question, "
#         "just reformulate it if needed and otherwise return it as is."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system",contextualize_q_system_prompt),
#                 MessagesPlaceholder('chat_history'),
#                 ("user","{input}")
#             ]
#         )
#         history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

#         # answer quesiton
#         system_prompt = (
#             "You are an assistant for question-answering tasks. "
#                 "Use the following pieces of retrieved context to answer "
#                 "the question. If you don't know the answer, say that you "
#                 "don't know. Use three sentences maximum and keep the "
#                 "answer concise."
#                 "\n\n"
#                 "{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system",system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}")
#             ]
#         )
#         question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
#         def get_session_history(session:str)->BaseChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id]=ChatMessageHistory()
#             return st.session_state.store[session_id]
        
#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer"
#         )
#         user_input = st.text_input("Your question: ")
#         if user_input:
#             session_history=get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={
#                     "configurable": {"session_id":session_id}
#                 },
#             )
#             # st.write(st.session_state.store)
#             st.write("Assistant:", response['answer'])
#             # st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please enter GROQ API key to proceed")

# st.chat_input("your message")

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

os.environ['HF_KEY']=os.getenv("HF_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="Conversational PDF Chatbot")
st.title("📚 Conversational PDF Chatbot using GROQ + HuggingFace")

api_key = st.text_input("Enter your GROQ API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
    session_id = st.text_input("Enter Session ID:", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for i, uploaded_file in enumerate(uploaded_files):
            temppdf = f"./temp_{i}.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            documents.extend(loader.load())
            os.remove(temppdf)

        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. Do NOT answer the question."),
            MessagesPlaceholder('chat_history'),
            ("user", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following retrieved context to answer "
                       "the question. If you don't know the answer, say 'I don't know'. Keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Show past messages in the chat UI
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle new user input
        prompt = st.chat_input("Ask something about your PDFs...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )
            answer = response["answer"]
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.warning("🔐 Please enter your GROQ API key to begin.")
