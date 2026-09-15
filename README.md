# Simple AI RAG Application

A simple **Retrieval-Augmented Generation (RAG)** application built with **Python, Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq LLM**.

The application loads information from a text file, splits it into smaller chunks, converts the chunks into vector embeddings, stores them in a FAISS vector database, retrieves the most relevant information based on the user's question, and generates a short answer using an LLM.

---

## 🚀 Features

* Load information from a `.txt` file
* Split large documents into smaller chunks
* Generate embeddings using Hugging Face
* Store embeddings in a FAISS vector database
* Retrieve the top relevant document chunks
* Generate answers using Groq LLM
* Simple Streamlit web interface
* Returns only information available in the retrieved context
* Returns `Not found` when the required information is unavailable

---

## 🏗️ Project Architecture

```text
                 ┌──────────────────┐
                 │     data.txt     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   TextLoader     │
                 └────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Recursive Text Splitter │
              │ chunk_size = 600        │
              │ overlap = 100           │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Hugging Face Embeddings │
              │ all-MiniLM-L6-v2        │
              └────────────┬────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │   FAISS Vector   │
                 │     Database     │
                 └────────┬─────────┘
                          │
                    User Question
                          │
                          ▼
                 ┌──────────────────┐
                 │    Retriever     │
                 │    Top 3 chunks  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Context + Query  │
                 │     Prompt       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Groq LLM      │
                 │ gpt-oss-120b     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Streamlit Answer │
                 └──────────────────┘
```

---

## 🧠 How It Works

The application follows a basic RAG pipeline:

### 1. Load Environment Variables

The application loads API keys from the `.env` file.

```python
load_dotenv()

os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
```

The required API keys are:

* `HF_API_KEY`
* `GROQ_API_KEY`

---

### 2. Initialize the LLM

Groq is used as the Large Language Model provider.

```python
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
)
```

A temperature of `0.0` is used to make the generated responses more deterministic.

---

### 3. Load the Document

The application reads information from `data.txt`.

```python
data = TextLoader("../data.txt")
docs = data.load()
```

The text file acts as the knowledge source for the RAG system.

---

### 4. Split the Document

Large documents are divided into smaller chunks using `RecursiveCharacterTextSplitter`.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

final_splitter = splitter.split_documents(docs)
```

#### Configuration

| Parameter     | Value | Purpose                                           |
| ------------- | ----: | ------------------------------------------------- |
| Chunk Size    |   600 | Maximum size of each chunk                        |
| Chunk Overlap |   100 | Keeps some information between neighboring chunks |

Chunking helps the retrieval system find relevant pieces of information efficiently.

---

### 5. Generate Embeddings

The application uses the Hugging Face model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

```python
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
```

The text chunks are converted into numerical vectors called **embeddings**.

These vectors allow the application to perform semantic similarity searches.

---

### 6. Create the FAISS Vector Database

FAISS is used to store and search the generated embeddings.

```python
db = FAISS.from_documents(
    final_splitter,
    embeddings
)
```

The vector database is saved locally:

```python
db.save_local("faiss_index")
```

It can later be loaded using:

```python
new_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

### 7. Create the Retriever

The FAISS database is converted into a retriever.

```python
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)
```

The value `k=3` means that the application retrieves the **3 most relevant chunks** for a user's question.

---

### 8. User Question

The Streamlit application provides a simple text input:

```python
user_input = st.text_input("Enter something:")
```

For example:

```text
Who won the IPL 2024 final?
```

---

### 9. Retrieve Relevant Information

The user's question is sent to the retriever:

```python
results = retriever.invoke(user_input)
```

The retrieved documents are combined into a context:

```python
context = "\n".join(
    doc.page_content for doc in results
)
```

---

### 10. Generate the Final Answer

The retrieved context and user's question are passed to the LLM.

The prompt instructs the model to:

* Answer only from the provided context
* Return a short answer
* Avoid explanations
* Avoid adding external information
* Return `Not found` if the answer isn't available

This helps reduce hallucination and keeps the response grounded in the source document.

---

## 📁 Project Structure

Recommended project structure:

```text
project/
│
├── app.py
├── data.txt
├── .env
├── .gitignore
├── requirements.txt
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

> Adjust the structure according to where your `app.py` and `data.txt` are located.

---

## 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
HF_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```

**Do not commit your `.env` file to GitHub.**

Add this to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt`, the main dependencies are:

```text
python-dotenv
streamlit
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

You can then enter a question into the input field.

---

## 💡 Example

Suppose `data.txt` contains information about IPL.

User enters:

```text
Explain about IPL
```

The system performs:

```text
User Question
      ↓
FAISS Retriever
      ↓
Top 3 Relevant Chunks
      ↓
Context
      ↓
Prompt
      ↓
Groq LLM
      ↓
Short Answer
```

If the requested information exists in `data.txt`, the application returns the relevant answer.

If the information does not exist in the retrieved context:

```text
Not found
```

---

## 🛠️ Technologies Used

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Application development         |
| Streamlit                      | Web interface                   |
| LangChain                      | RAG application framework       |
| TextLoader                     | Loading text documents          |
| RecursiveCharacterTextSplitter | Document chunking               |
| Hugging Face                   | Text embedding model            |
| all-MiniLM-L6-v2               | Embedding generation            |
| FAISS                          | Vector database                 |
| Groq                           | LLM inference                   |
| python-dotenv                  | Environment variable management |

---

## 🔄 RAG Pipeline

The complete pipeline can be summarized as:

```text
Document
   ↓
Load
   ↓
Split into Chunks
   ↓
Generate Embeddings
   ↓
Store in FAISS
   ↓
User Query
   ↓
Similarity Search
   ↓
Retrieve Top 3 Chunks
   ↓
Build Context
   ↓
Send Context + Question to LLM
   ↓
Generate Grounded Answer
   ↓
Display in Streamlit
```

---

## 🎯 Purpose of the Project

This project demonstrates the fundamental implementation of a **Retrieval-Augmented Generation (RAG)** system.

Instead of asking the LLM to answer directly from its general knowledge, the application first retrieves relevant information from a custom knowledge source and then provides that information to the LLM as context.

This approach is useful for building:

* Document Q&A systems
* Company knowledge assistants
* FAQ chatbots
* Internal knowledge bots
* Educational assistants
* Custom AI assistants
* Private document search systems

---

## 🚧 Future Improvements

Possible improvements include:

* Support for PDF documents
* Support for multiple documents
* Chat history
* Conversation memory
* Better UI design
* Streaming responses
* Metadata filtering
* Source document citations
* Upload documents directly through Streamlit
* Multiple vector database support
* Hybrid search
* Reranking retrieved documents
* Authentication
* Deployment to cloud platforms
* More advanced Agentic RAG workflows

---

## ⚠️ Important Notes

The current implementation creates and saves the FAISS index, but it also loads the saved index into `new_db` and then continues using `db` for retrieval.

For a cleaner implementation, you can use the loaded database:

```python
new_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = new_db.as_retriever(
    search_kwargs={"k": 3}
)
```

Also, the current `query="explain about IPL"` variable is not used by the application. The actual query comes from:

```python
user_input
```

---

## 👨‍💻 Project Summary

**Project:** Simple AI RAG Application

**Type:** Retrieval-Augmented Generation (RAG)

**Frontend:** Streamlit

**LLM:** Groq

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Vector Database:** FAISS

**Framework:** LangChain

**Language:** Python

---

## 📌 Learning Outcomes

By building this project, you can understand:

1. How documents are loaded into a RAG system
2. Why document chunking is required
3. What embeddings are
4. How semantic search works
5. How FAISS stores and retrieves vectors
6. How LangChain connects the RAG components
7. How retrieved context is passed to an LLM
8. How RAG can reduce unsupported answers
9. How to build a basic AI application with Streamlit
10. How custom knowledge can be connected to an LLM

---

## ⭐ Project Flow in One Line

**Document → Chunking → Embeddings → FAISS → Retrieval → Context → Groq LLM → Answer**
