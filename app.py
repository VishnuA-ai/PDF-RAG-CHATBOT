import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

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

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #070b14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0d2240 0%, #070b14 60%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #070b14 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.1) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Main container ── */
.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

/* ── Hero header ── */
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
    letter-spacing: 0.02em;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.7) !important;
    border: 1.5px dashed rgba(99,179,237,0.25) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(6,182,212,0.5) !important;
}
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Success / Error banners ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
    font-size: 0.85rem !important;
}

/* ── Status bar after processing ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.18);
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #67e8f9;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.status-dot {
    width: 8px; height: 8px;
    background: #06b6d4;
    border-radius: 50%;
    box-shadow: 0 0 8px #06b6d4;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

.stChatMessage .stMarkdown p {
    font-size: 0.9rem;
    line-height: 1.65;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.9) !important;
    border: 1.5px solid rgba(99,179,237,0.18) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(6,182,212,0.5) !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.08) !important;
}
[data-testid="stChatInputSubmitButton"] svg {
    fill: #06b6d4 !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: rgba(6,182,212,0.08) !important;
    color: #67e8f9 !important;
    border: 1px solid rgba(6,182,212,0.25) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    padding: 0.45rem 1rem !important;
}
.stButton > button:hover {
    background: rgba(6,182,212,0.18) !important;
    border-color: rgba(6,182,212,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #06b6d4 !important;
}

/* ── Source citation pill ── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #67e8f9;
    font-weight: 500;
    margin: 0.2rem;
    letter-spacing: 0.03em;
}

/* ── Sidebar file badge ── */
.file-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(6,182,212,0.06);
    border: 1px solid rgba(6,182,212,0.15);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: #94a3b8;
    word-break: break-all;
}

/* ── Sidebar section label ── */
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.6rem 0;
}

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.08) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 10px; }

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem !important; }
    .hero-text h1 { font-size: 1.4rem; }
    .hero-icon { width: 44px; height: 44px; font-size: 1.4rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── API KEY ─────────────────────────────────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("⚠️  GROQ_API_KEY missing in your .env file.")
    st.stop()

# ─── SESSION STATE ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(90deg,#e2e8f0,#06b6d4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">ResearchMind</div>
        <div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;letter-spacing:0.04em;">
            AI · PDF Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Actions</div>', unsafe_allow_html=True)

    if st.button("🗑️  Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="sidebar-label">Loaded Documents</div>', unsafe_allow_html=True)
    doc_placeholder = st.container()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;line-height:1.6;">
        Powered by <b style="color:#475569;">Groq LLaMA-3.3-70B</b><br>
        Embeddings via <b style="color:#475569;">MiniLM-L6-v2</b><br>
        Vector search via <b style="color:#475569;">FAISS</b>
    </div>
    """, unsafe_allow_html=True)

# ─── HERO HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🧠</div>
    <div class="hero-text">
        <h1>ResearchMind AI</h1>
        <p>Upload PDFs — ask anything. Answers grounded in your documents.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── FILE UPLOADER ────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Drop your PDF files here or click to browse",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="visible"
)

# ─── PROCESS PDFs ────────────────────────────────────────────────────────────────
if uploaded_files:
    with st.spinner("Indexing documents…"):
        all_docs = []
        file_names = []

        for uploaded_file in uploaded_files:
            file_names.append(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vectorstore/db_faiss")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    # Populate sidebar badges
    with doc_placeholder:
        for name in file_names:
            st.markdown(f'<div class="file-badge">📄 {name}</div>', unsafe_allow_html=True)

    # Status bar
    total_chunks = len(docs)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot"></div>
        {len(file_names)} document{"s" if len(file_names)>1 else ""} indexed
        &nbsp;·&nbsp; {total_chunks} chunks
        &nbsp;·&nbsp; Ready to answer
    </div>
    """, unsafe_allow_html=True)

    # ─── CHAT HISTORY ────────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                pills = "".join(
                    f'<span class="source-pill">📌 {s}</span>'
                    for s in message["sources"]
                )
                st.markdown(f'<div style="margin-top:0.6rem">{pills}</div>', unsafe_allow_html=True)

    # ─── CHAT INPUT ──────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your documents…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in relevant_docs])

                sources = list(set(
                    f"Page {d.metadata.get('page', '?')}" for d in relevant_docs
                ))

                history = "\n".join(
                    f"{m['role'].capitalize()}: {m['content']}"
                    for m in st.session_state.messages
                )

                prompt = f"""You are ResearchMind, a precise and helpful AI research assistant.
Use the context and conversation history below to answer accurately.
Be concise yet thorough. If the answer isn't in the context, say so clearly.

Conversation History:
{history}

Context from documents:
{context}

Question: {question}
"""
                response = llm.invoke(prompt)
                answer = response.content

            st.markdown(answer)
            pills = "".join(f'<span class="source-pill">📌 {s}</span>' for s in sources)
            st.markdown(f'<div style="margin-top:0.6rem">{pills}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

else:
    # ─── EMPTY STATE ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        margin-top: 1rem;
    ">
        <div style="
            font-size: 3.5rem;
            margin-bottom: 1rem;
            filter: drop-shadow(0 0 24px rgba(6,182,212,0.4));
        ">📂</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #475569;
            margin-bottom: 0.5rem;
        ">No documents loaded yet</div>
        <div style="
            font-size: 0.82rem;
            color: #334155;
            max-width: 320px;
            margin: 0 auto;
            line-height: 1.6;
        ">
            Upload one or more PDF files above to start asking<br>intelligent questions about your research.
        </div>
    </div>
    """, unsafe_allow_html=True)