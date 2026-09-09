import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Data Ingestion | Study RAG", page_icon="🏠", layout="wide")

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

try:
    embedder = load_embedder()
except Exception as e:
    st.error(f"Failed to load embedder: {e}")

st.title("🏠 Document Ingestion System")
st.markdown("Upload your study materials here to build the knowledge base for the Study Guider and Chat Assistant.")

st.header("1. API Configuration")
api_key = st.text_input("Groq API Key", type="password", value=st.session_state.api_key)
if api_key:
    st.session_state.api_key = api_key
elif "GROQ_API_KEY" in st.secrets:
    st.session_state.api_key = st.secrets["GROQ_API_KEY"]
    st.success("API Key loaded from Streamlit secrets.")

st.header("2. Upload Materials")
# Updated to only accept PDF files
uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)

if st.button("Process Documents") and uploaded_files:
    with st.spinner("Extracting text and building vector space..."):
        all_text = ""
        for file in uploaded_files:
            # Removed the else block for images; processes PDFs only
            if file.name.lower().endswith('.pdf'):
                doc = fitz.open(stream=file.read(), filetype="pdf")
                for page in doc:
                    all_text += page.get_text() + "\n"
        
        chunk_size = 600
        overlap = 100
        chunks = []
        for i in range(0, len(all_text), chunk_size - overlap):
            chunks.append(all_text[i:i+chunk_size].strip())
        
        chunks = [c for c in chunks if len(c) > 10]
        
        if chunks:
            st.session_state.chunks = chunks
            embeddings = embedder.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))
            st.session_state.faiss_index = index
            st.success(f"Successfully processed {len(chunks)} text chunks! You can now use the Study Guider or Chat.")
        else:
            st.error("No readable text could be extracted from the PDFs.")
