import argparse
import torch
import os
import chromadb
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import streamlit as st
import tempfile
import time
from utils.prompt_utils import promptManager
from operator import itemgetter
from utils.logging_utils import logger

# Load base environment variables first
load_dotenv()
# Load local overrides with explicit override
load_dotenv(".env.local", override=True)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_file_names" not in st.session_state:
    st.session_state.pdf_file_names = []


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )


@st.cache_resource
def load_llm(use_quantization=True):
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    model = None
    if use_quantization:
        # Use BitsAndBytesConfig for quantization
        logger.info("Loading model with quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Hoặc load_in_8bit=True
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",  # nf4 là lựa chọn tốt cho mô hình lớn
        )

        # Load model với quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
    else:
        # Load model without quantization
        logger.info("Loading model without quantization")

        # We are using float16 for better performance and lower memory usage
        # This requires a GPU with at least 12GB VRAM for Vicuna-7B
        # If you have a GPU with less VRAM, consider using quantization
        # Good read: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", torch_dtype=torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )
    return HuggingFacePipeline(pipeline=model_pipeline)


def get_chroma_client(allow_reset=False):
    """Get a Chroma client for vector database operations."""
    # Use PersistentClient for persistent storage
    return chromadb.PersistentClient(
        settings=chromadb.Settings(allow_reset=allow_reset)
    )


def process_pdf(uploaded_files):
    """Process multiple uploaded PDF files, combine their docs, and build a single retriever and RAG chain."""
    all_docs = []
    file_names = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False, prefix=uploaded_file.name, suffix=".pdf"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        semantic_splitter = SemanticChunker(
            embeddings=st.session_state.embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True,
        )
        docs = semantic_splitter.split_documents(documents)
        # for doc in docs:
        #     # Add file name to doc metadata for citation
        #     doc.metadata["source"] = uploaded_file.name
        all_docs.extend(docs)
        file_names.append(uploaded_file.name)
        os.unlink(tmp_file_path)

    chroma_client = get_chroma_client(allow_reset=True)
    chroma_client.reset()  # empties and completely resets the database. This is destructive and not reversible.

    vector_db = Chroma.from_documents(
        documents=all_docs, embedding=st.session_state.embeddings, client=chroma_client
    )
    retriever = vector_db.as_retriever()
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = promptManager.load_prompt_template_from_file("rag_with_memory.v1.txt")

    # Build the RAG chain
    # This use  LangChain Expression Language (LCEL) 
    # where  LangChain overrides the Python pipe operator (|) for its Runnable objects
    # Ref: https://python.langchain.com/docs/concepts/lcel/
    rag_chain = (
        {
            "context": itemgetter("question")
            | retriever
            | promptManager.format_docs_chunks,
            "question": itemgetter("question"),
            "chat_history": lambda x: promptManager.format_conversation_history(
                x["chat_history"]
            ),
        }
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    return rag_chain, len(all_docs), file_names


def add_message(role, content):
    """Thêm tin nhắn vào lịch sử chat"""
    st.session_state.chat_history.append(
        {"role": role, "content": content, "timestamp": time.time()}
    )


def clear_chat():
    """Xóa lịch sử chat"""
    st.session_state.chat_history = []


def display_chat():
    """Hiển thị lịch sử chat"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(
                "Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊"
            )


def retrieve_chat_history():
    # Retrieve the last x messages from chat history
    message_threshold = 10  # Number of messages to retrieve
    return st.session_state.chat_history[-message_threshold:-1]


def main(use_quantization = True):
    if use_quantization:
        logger.info("Using quantization for model loading")
    else:
        logger.info("Using full precision for model loading")
    
    # Streamlit app configuration
    st.set_page_config(
        page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("PDF RAG Assistant")

    # Trong streamlit v-1.38 không khỗ trợ param size
    st.logo("./assets/logo.png", size="large")

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")

        # Load models
        if not st.session_state.models_loaded:
            st.warning("⏳ Đang tải models...")
            with st.spinner("Đang tải AI models..."):
                st.session_state.embeddings = load_embeddings()
                st.session_state.llm = load_llm(use_quantization=use_quantization)
                st.session_state.models_loaded = True
            st.success("✅ Models đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ Models đã sẵn sàng!")

        st.markdown("---")

        # Upload PDF
        st.subheader("📄 Upload tài liệu")
        uploaded_files = st.file_uploader(
            "Chọn file PDF", type="pdf", accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("🔄 Xử lý PDF", use_container_width=True):
                with st.spinner("Đang xử lý PDF..."):
                    st.session_state.rag_chain, num_chunks, file_names = process_pdf(
                        uploaded_files
                    )
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_file_names = file_names
                    # Reset chat history khi upload PDF mới
                    clear_chat()
                    add_message(
                        "assistant",
                        f"✅ Đã xử lý thành công {len(file_names)} file PDF: **{', '.join(file_names)}**!\n\n📊 Tổng cộng {num_chunks} phần. Bạn có thể bắt đầu đặt câu hỏi về nội dung tài liệu.",
                    )
                st.rerun()

        # PDF status
        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {'; '.join(st.session_state.pdf_file_names)}")
        else:
            st.info("📄 Chưa có tài liệu")

        # Chat controls
        st.divider()
        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()

        # Instructions
        st.divider()
        st.subheader("📋 Hướng dẫn")
        st.markdown(
            """
        **Cách sử dụng:**
        1. **Upload PDF:** Chọn file PDF và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi:** Nhập câu hỏi trong ô chat
        3. **Nhận câu trả lời:** AI sẽ trả lời dựa trên nội dung PDF
        """
        )

    # Main content
    st.markdown("*Trò chuyện với chabot để trao đổi về nội dung tài liệu PDF của bạn*")
    chat_container = st.container()

    # Chat container
    with chat_container:
        # Display chat history
        display_chat()

    # Chat input
    if st.session_state.models_loaded:
        if st.session_state.pdf_processed:
            # User input
            if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
                add_message("user", user_input)

                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        answer = ""
                        try:
                            # Invoke the RAG chain with user input and chat history
                            output = st.session_state.rag_chain.invoke(
                                {
                                    "question": user_input,
                                    "chat_history": retrieve_chat_history(),
                                }
                            )

                            # Clean up the response
                            if "Answer:" in output:
                                answer = output.split("Answer:")[1].strip()
                            else:
                                answer = output.strip()

                            # Display response
                            st.write(answer)

                            # Add assistant message to history
                            add_message("assistant", answer)
                        except Exception as e:
                            logger.error(e, exc_info=True)
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF RAG Chatbot")
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable quantization (requires more GPU memory)",
    )

    args = parser.parse_args()
    use_quantization = not args.no_quantization
    
    main(use_quantization=use_quantization)
