import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0d2240 0%, #0a0e1a 60%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
}

[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
}

.hero-wrap {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid rgba(99,179,237,0.12);
}

.hero-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #1a56db 0%, #06b6d4 100%);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 8px 32px rgba(6,182,212,0.35);
    flex-shrink: 0;
}

.hero-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e2e8f0 30%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; line-height: 1.1;
}

.hero-text p {
    font-size: 0.88rem;
    color: #64748b;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
}

[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8) !important;
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(6,182,212,0.6) !important;
    background: rgba(15,23,42,0.95) !important;
}

[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    font-size: 0.85rem !important;
}

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #67e8f9;
    font-weight: 500;
}

.status-dot {
    width: 8px; height: 8px;
    background: #06b6d4;
    border-radius: 50%;
    box-shadow: 0 0 8px #06b6d4;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.95) !important;
    border: 1.5px solid rgba(99,179,237,0.2) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(6,182,212,0.6) !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.1) !important;
}

.stButton > button {
    background: rgba(6,182,212,0.1) !important;
    color: #67e8f9 !important;
    border: 1px solid rgba(6,182,212,0.3) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: rgba(6,182,212,0.2) !important;
    border-color: rgba(6,182,212,0.5) !important;
}

.file-badge {
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.6rem;
    font-size: 0.78rem;
    color: #94a3b8;
    word-break: break-word;
}

.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.7rem 0;
}

hr { border-color: rgba(99,179,237,0.1) !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.25); border-radius: 10px; }

@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem !important; }
    .hero-text h1 { font-size: 1.4rem; }
    .hero-icon { width: 44px; height: 44px; }
}
</style>
""", unsafe_allow_html=True)

# ─── API KEY & LLM ──────────────────────────────────────────────────────────────

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Add it to Streamlit Secrets or .env file")
    st.stop()

# ─── SESSION STATE ───────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}

if "processing" not in st.session_state:
    st.session_state.processing = False

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def process_pdfs(uploaded_files):
    """Process uploaded PDFs and create vector store"""
    if not uploaded_files:
        return None, {}
    
    all_docs = []
    metadata = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        try:
            status_text.text(f"📖 Processing: {file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            for page in pages:
                page.metadata["source"] = file.name
                all_docs.append(page)
            
            metadata[file.name] = {
                "pages": len(pages),
                "uploaded": True
            }
            
            os.unlink(tmp_path)
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            continue
    
    if not all_docs:
        st.error("❌ No text found in PDF. Try uploading a different file.")
        st.stop()
        return None, metadata
    
    status_text.text("✂️ Splitting documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_docs)
    
    status_text.text("🧠 Creating embeddings...")
    
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    status_text.text("✅ Documents processed successfully!")
    progress_bar.empty()
    status_text.empty()
    
    return vectorstore, metadata

def create_qa_chain(vectorstore):
    """Create RAG chain using LCEL (LangChain Expression Language)"""
    # Verify model name - MUST be exactly llama-3.3-70b-versatile
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.3,
        max_tokens=1000
    )
    
    prompt_template = """You are an intelligent research assistant. Use the provided context to answer the user's question accurately.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, detailed answer based on the context
- If information is not in the context, say so clearly
- Cite specific parts when relevant
- Keep your answer focused and organized

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    def format_docs(docs):
        """Format retrieved documents with context truncation"""
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in docs
        )
        # Truncate context to avoid 400 errors from Groq
        context = context[:12000]
        return context
    
    # Build LCEL chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #e2e8f0, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
        ">
            ResearchMind
        </div>
        <div style="font-size: 0.75rem; color: #64748b; letter-spacing: 0.08em;">
            🧠 AI · 📚 PDF Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ─── ACTIONS SECTION ─────────────────────────────────────
    st.markdown('<p class="sidebar-label">⚙️ Actions</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.uploaded_files = []
            st.session_state.doc_metadata = {}
            st.rerun()
    
    st.divider()
    
    # ─── LOADED DOCUMENTS ────────────────────────────────────
    st.markdown('<p class="sidebar-label">📄 Documents</p>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            doc_name = file.name if hasattr(file, 'name') else str(file)
            metadata = st.session_state.doc_metadata.get(doc_name, {})
            pages = metadata.get("pages", "?")
            
            st.markdown(f"""
            <div class="file-badge">
                📋 {doc_name[:20]}... <br/>
                <span style="color: #64748b; font-size: 0.7rem;">{pages} pages</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📭 No documents yet")
    
    st.divider()
    
    # ─── STATS ──────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">📊 Stats</p>', unsafe_allow_html=True)
    
    total_docs = len(st.session_state.uploaded_files)
    total_messages = len(st.session_state.messages)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", total_docs, delta=None)
    with col2:
        st.metric("Messages", total_messages, delta=None)
    
    st.divider()
    
    # ─── CHAT PREVIEW ───────────────────────────────────────
    st.markdown('<p class="sidebar-label">💬 Recent Chat</p>', unsafe_allow_html=True)
    
    if st.session_state.messages:
        for msg in st.session_state.messages[-3:]:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            preview = msg["content"][:35]
            st.caption(f"{role_icon} {preview}...")
    else:
        st.caption("No messages yet")
    
    st.divider()
    
    # ─── ABOUT ──────────────────────────────────────────────
    st.markdown('<p class="sidebar-label">ℹ️ About</p>', unsafe_allow_html=True)
    st.caption("⚡ Powered by Groq LLaMA 3.3 70B")
    st.caption("🔗 Embeddings via MiniLM-L6-v2")
    st.caption("🗂️ Vector Search via FAISS")
    st.caption("🚀 Built with Streamlit")

# ─── MAIN CONTENT ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🧠</div>
    <div class="hero-text">
        <h1>ResearchMind</h1>
        <p>Chat with your PDFs. Ask anything. Get instant answers.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── FILE UPLOADER ──────────────────────────────────────────────────────────────

st.markdown('<p class="sidebar-label" style="margin-top: 0;">📤 Upload PDFs</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop your PDFs here",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ─── PROCESS PDFs ───────────────────────────────────────────────────────────────

if uploaded_files and uploaded_files != st.session_state.uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    st.session_state.processing = True
    
    with st.spinner("🔄 Processing your documents..."):
        vectorstore, metadata = process_pdfs(uploaded_files)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_metadata = metadata
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("✅ Documents ready! Ask your questions.")
        else:
            st.error("Failed to process documents")
    
    st.session_state.processing = False
    st.rerun()

# ─── CHAT INTERFACE ─────────────────────────────────────────────────────────────

if st.session_state.vectorstore:
    st.markdown('<p class="sidebar-label">💬 Chat with Your PDFs</p>', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask anything about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Create chain and retriever
                    qa_chain, retriever = create_qa_chain(st.session_state.vectorstore)
                    
                    # Invoke chain with explicit error handling
                    try:
                        answer = qa_chain.invoke(user_input)
                    except Exception as e:
                        st.error(f"❌ LLM Error: {str(e)}")
                        raise
                    
                    # Get source documents
                    docs = retriever.get_relevant_documents(user_input)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if docs:
                        st.divider()
                        st.markdown("**📌 Sources:**")
                        source_set = set()
                        for doc in docs:
                            source = doc.metadata.get("source", "Unknown")
                            if source not in source_set:
                                source_set.add(source)
                                st.markdown(f"• {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.markdown("Please try again or rephrase your question.")

else:
    if uploaded_files:
        st.info("⏳ Processing your documents...")
    else:
        st.info("👆 Upload PDF documents to get started!")
        st.markdown("""
        ### How to use:
        1. **Upload PDFs** - Click the upload area above
        2. **Wait** - The app will process your documents
        3. **Ask Questions** - Type your question in the chat box
        4. **Get Answers** - Get instant answers with source citations
        """)
        
    