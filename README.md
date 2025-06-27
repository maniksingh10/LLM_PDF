
# 🧠 Chat with Your PDFs using Gemini and Pinecone

This project is an interactive **Retrieval-Augmented Generation (RAG)** app that lets you **upload PDFs, embed their content, store in Pinecone, and ask questions**. It uses **Google Gemini LLMs** for embeddings and answering queries.

Built with:
- [LangChain](https://python.langchain.com)
- [Pinecone](https://www.pinecone.io/)
- [Google Generative AI](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)



## ✨ Features

✅ **PDF Upload**  
✅ **Chunking**  
✅ **Embeddings via Gemini**  
✅ **Vector Storage using Pinecone**  
✅ **RAG-based Q&A**  
✅ **Interactive Streamlit UI**  



## 🚀 Getting Started


### Install Dependencies
```bash
pip install -r requirements.txt
```


###  Create .env File
```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

###  Create Pinecone Index
- Name: `pdf`
- Dimension: `768`
- Metric: `cosine`

###  Run the App
```bash
streamlit run app.py
```


## 🖥️ Usage

1. Upload a PDF
2. Ask a question about it
3. Get an answer and see relevant content

---
