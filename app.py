import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()  # Loads environment variables (e.g., API keys) from a .env file.

# Page configuration
st.set_page_config(
    page_title="UIDAI Assistance",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .resizable-container {
        width: 80%;
        height: auto;
        margin: auto;
        padding: 20px;
    }
    .chat-container {
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f8;
    }
    .user-message {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }
    .user-message p {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
    }
    .bot-message {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .bot-message p {
        background-color: #e3e3e3;
        padding: 10px;
        border-radius: 15px;
        margin: 5px;
        max-width: 70%;
    }
    .user-icon, .bot-icon {
        width: 40px;
        height: 40px;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a container with the custom class
with st.container():
    st.markdown('<div class="resizable-container">', unsafe_allow_html=True)
    
    # Title with icon
    st.markdown(
        """
        <h1 style="text-align: center; display: flex; justify-content: center; align-items: center;">
            <img src="https://indiadesignsystem.bombaydc.com/assets/india-designs/display/Aadhar/black.svg" alt="Fingerprint Logo" width="50" height="50"> UIDAI <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/virtual-assistant-avatar-9430722-7676417.png?f=webp&w=256" alt="Fingerprint Logo" width="60" height="60">
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Input field
    query = st.chat_input("Ask issue here...")

    # Define system and user prompts
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Load URLs
    from langchain_community.document_loaders import UnstructuredURLLoader
    urls = [
        'https://www.wikihow.life/Get-an-e%E2%80%90Aadhaar-Card#:~:text=Applying%20for%20an%20Aadhaar%20Card%201%201%20Gather,in%20the%20mail%20in%20about%203%20months.%20',
        "https://www.hindustantimes.com/india-news/link-aadhaar-with-mobile-number-in-five-easy-steps-a-step-by-step-guide-101622522292918.html",
        "https://www.squareyards.com/blog/common-problems-with-aadhaar-cards-aadhpan#:~:text=Common%20Problems%20with%20%F0%9D%91%A8%F0%9D%92%82%F0%9D%92%85%F0%9D%92%89%F0%9D%92%82%F0%9D%92%82%F0%9D%92%93%20%F0%9D%91%AA%F0%9D%92%82%F0%9D%92%93%F0%9D%92%85%20and%20Ways%20To,FAQ%E2%80%99s%20about%20Common%20Issues%20with%20Aadhar%20Card%20",
        "https://www.squareyards.com/blog/how-to-link-aadhaar-to-pm-kisan-samman-nidhi-aadhpan",
        "https://www.bankbazaar.com/aadhar-card/common-problems-with-aadhar-card.html"
    ]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Process documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
    docs = text_splitter.split_documents(data)

    # Create vectorstore and retriever
    vectorstore = FAISS.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Load the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=None, timeout=None)

    # Process query
    if query:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})

        # Chat-like UI
        # User message
        user_col1, user_col2 = st.columns([4, 1])
        with user_col1:
            st.markdown(
                f"""
                <div class="chat-container user-message">
                    <img class="user-icon" src="https://img.icons8.com/color/48/000000/user.png" alt="User">
                    <p>{query}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Chatbot response
        bot_col1, bot_col2 = st.columns([1, 4])
        with bot_col2:
            st.markdown(
                f"""
                <div class="chat-container bot-message">
                    <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/virtual-assistant-avatar-9430722-7676417.png?f=webp&w=256" alt="Fingerprint Logo" width="60" height="60">
                    <p>{response['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
    st.markdown('</div>', unsafe_allow_html=True)


