ResearchMind AI - Setup Guide 🚀
Prerequisites

Python 3.10+
Git
GitHub account
Groq API key (free)


LOCAL SETUP (5 minutes)
Step 1: Clone & Setup
bashgit clone https://github.com/YOUR_USERNAME/researchmind-ai.git
cd researchmind-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Step 2: Install Dependencies
bashpip install -r requirements.txt
Step 3: Add API Key
Create .env file:
GROQ_API_KEY=your_key_here
Get free key at: https://console.groq.com
Step 4: Run
bashstreamlit run app.py
Visit: http://localhost:8501

STREAMLIT CLOUD DEPLOYMENT (5 minutes)
Step 1: Prepare GitHub Repository
bash# Make sure all files are committed
git add .
git commit -m "Ready for Streamlit Cloud"
git push origin main
Files needed:

✅ app.py
✅ requirements.txt
✅ .streamlit/config.toml
✅ .gitignore
✅ README.md

Step 2: Deploy on Streamlit Cloud

Go to: https://share.streamlit.io
Click "New app"
Fill in:

Repository: YOUR_USERNAME/researchmind-ai
Branch: main
Main file path: app.py


Click "Deploy"

Step 3: Add Secrets

Wait for deployment (2-3 minutes)
Click "☰" (menu) → "Settings"
Go to "Secrets"
Paste:

GROQ_API_KEY = your_groq_api_key_here

Save
App will auto-reboot

Step 4: Share Your App
Your app is live at:
https://researchmind-ai-YOUR_USERNAME.streamlit.app

GET GROQ API KEY (FREE)

Visit: https://console.groq.com/keys
Sign up or login
Click "Create API Key"
Copy the key
Add to your .env file (local) or Secrets (cloud)


VERIFY IT'S WORKING
Local

Upload a PDF
Ask a question
You should get an answer within 2-3 seconds

Cloud

Same as above
Share your app link with others
They can use it without installation


TROUBLESHOOTING
"GROQ_API_KEY not found"

Local: Create .env with your key
Cloud: Add to Secrets, then reboot

App crashes on upload

Check PDF file is valid
Try smaller PDF first
Check logs in Streamlit Cloud

Slow responses

Groq may be rate-limited
Try rephrasing question
Refresh the page


NEXT STEPS

 Test with your own PDFs
 Share with friends/colleagues
 Customize styling (edit CSS in app.py)
 Add more features from roadmap


FILE CHECKLIST
Before pushing to GitHub, make sure you have:
✅ app.py
✅ requirements.txt
✅ .streamlit/config.toml
✅ .gitignore
✅ README.md
✅ .env (LOCAL ONLY - in .gitignore)

NEED HELP?

Streamlit: https://docs.streamlit.io
LangChain: https://python.langchain.com/docs
Groq: https://console.groq.com/docs
GitHub Issues: Create an issue on your repo


Happy chatting! 🎉