<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=SYNE&weight=800&size=34&pause=1000&color=00F7FF&center=true&vCenter=true&multiline=true&width=1400&height=180&lines=PDF+RAG+Chatbot+%F0%9F%9A%80;AI+Research+Assistant+%F0%9F%93%9A;Chat+with+Multiple+PDFs+%F0%9F%92%AC)](https://git.io/typing-svg)
<br/>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>

  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-00A67E?style=for-the-badge&logo=groq&logoColor=white"/>

  <img src="https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge&logo=databricks&logoColor=white"/>

  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>

  <img src="https://img.shields.io/badge/LangChain-RAG_Framework-00C853?style=for-the-badge&logo=chainlink&logoColor=white"/>

</p>

<p align="center">
  <b>Upload PDFs. Ask anything. Get grounded, intelligent answers instantly.</b>
  <br/><br/>
  <sub>
    A modern Multi-PDF Conversational RAG application built with
    Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq LLaMA 3.3 70B.
  </sub>
</p>

</div>

---
---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Multi-PDF Upload** | Drop multiple PDFs at once — all indexed together |
| 🧠 **LLaMA-3.3 70B via Groq** | Ultra-fast inference, no GPU needed |
| 🔍 **FAISS Vector Search** | Semantic retrieval across all your documents |
| 📌 **Source Citations** | Every answer shows exact page references |
| 💬 **Chat History** | Full conversation memory within session |
| 🌑 **Premium Dark UI** | Deep-space theme, Syne + DM Sans fonts |
| 📱 **Mobile Responsive** | Works on any screen size |
| ⚡ **Groq Speed** | Sub-second LLM responses |

---

## 🖥️ Demo

```
Upload paper.pdf  →  "What is the main contribution of this paper?"
                  ←  "The paper proposes... [📌 Page 3, Page 7]"
```

---

## 🛠️ Tech Stack

```
Frontend    →  Streamlit + custom CSS (dark theme)
LLM         →  Groq API → LLaMA-3.3-70B-Versatile
Embeddings  →  sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
Vector DB   →  FAISS (local)
PDF Loader  →  LangChain PyPDFLoader
Chunking    →  RecursiveCharacterTextSplitter (1000 tokens, 200 overlap)
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/researchmind-ai.git
cd researchmind-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free Groq API key at → [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Done. 🎉

---

## 📁 Project Structure

```
researchmind-ai/
│
├── app.py                  # Main Streamlit app
├── .env                    # API keys (never commit this)
├── .gitignore
├── requirements.txt
├── README.md
│
└── vectorstore/
    └── db_faiss/           # Auto-created on first PDF upload
        ├── index.faiss
        └── index.pkl
```

---

## 📦 requirements.txt

```txt
streamlit
python-dotenv
langchain
langchain-community
langchain-groq
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
huggingface-hub
```

Generate it directly:

```bash
pip freeze > requirements.txt
```

---

## ⚙️ How It Works

```
User uploads PDFs
        ↓
PyPDFLoader reads each page
        ↓
RecursiveCharacterTextSplitter chunks text (1000 tokens, 200 overlap)
        ↓
MiniLM-L6-v2 converts chunks → embeddings
        ↓
FAISS stores embeddings locally
        ↓
User asks a question
        ↓
FAISS retrieves top-4 relevant chunks (semantic search)
        ↓
Chunks + chat history → prompt → Groq LLaMA-3.3-70B
        ↓
Answer + page citations displayed in chat UI
```

---

## 🔐 Security Notes

- Never commit your `.env` file
- Add `vectorstore/` to `.gitignore` if documents are private
- The app stores no data externally — everything runs locally

```gitignore
# .gitignore
.env
vectorstore/
__pycache__/
*.pyc
.venv/
venv/
```

---

## 🗺️ Roadmap

- [ ] Persistent chat history across sessions
- [ ] Document management (delete individual PDFs)
- [ ] Export chat as PDF/Markdown
- [ ] Support DOCX, TXT, and web URLs
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Switch between multiple LLM providers

---

## 🤝 Contributing

Pull requests are welcome.

```bash
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

Open a PR and I'll review it.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Built by **[Vishnu A](https://linkedin.com/in/vishnuaitech)** · [GitHub](https://github.com/VishnuA-ai)

<sub>If this helped you, drop a ⭐ on the repo!</sub>

</div>
