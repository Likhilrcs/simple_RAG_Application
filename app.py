import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_groq import ChatGroq

# loading the .env file
load_dotenv()

os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    
)



query="explain about IPL"
# document loader
data = TextLoader("../data.txt")
docs = data.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=100)

# Text splitter
final_splitter = splitter.split_documents(docs)


# embedding

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# save the data in FAISS

db = FAISS.from_documents(final_splitter, embeddings)

db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# streamlit web application
st.title("Simple AI App")

user_input = st.text_input("Enter something:")


retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

results = retriever.invoke(user_input)


context = "\n".join(
    doc.page_content for doc in results
)

prompt = f"""
You are an information extraction system.

Read the context and answer the question.

Rules:
- Return ONLY the exact answer.
- Return one short sentence or phrase.
- Do not explain.
- Do not add information that is not in the context.
- If the answer is not present, return "Not found".

Context:
{context}

Question:
{user_input}

Answer:
"""

response = llm.invoke(prompt)

answer = response.content.strip()

st.write(answer)