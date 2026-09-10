import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Chat Assistant | AI RAG", page_icon="💬", layout="wide")

st.title("💬 RAG Chat Assistant")

if not st.session_state.get("faiss_index") or not st.session_state.get("api_key"):
    st.warning("Please configure your API Key and upload documents on the Home page first.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_web" not in st.session_state:
    st.session_state.awaiting_web = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_embedder()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.awaiting_web:
    with st.chat_message("assistant"):
        st.warning("I couldn't find the answer in your uploaded documents. Would you like me to search the web?")
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("🌐 Yes, Search Web", use_container_width=True):
                with st.spinner("Searching live sources..."):
                    results = DDGS().text(st.session_state.pending_query, max_results=3)
                    web_context = "\n\n".join([f"Source: {r.get('href')}\n{r.get('body')}" for r in results])
                    
                    client = Groq(api_key=st.session_state.api_key)
                    prompt = f"Web Results:\n{web_context}\n\nQuestion: {st.session_state.pending_query}\n\nAnswer thoroughly."
                    res = client.chat.completions.create(model=openai/gpt-oss-120b, messages=[{"role": "user", "content": prompt}])
                    
                    display_text = f"{res.choices[0].message.content}\n\n*(Sourced from live web search)*"
                    st.session_state.chat_history.append({"role": "assistant", "content": display_text})
                    st.session_state.awaiting_web = False
                    st.rerun()
        with col2:
            if st.button("❌ No, Cancel", use_container_width=True):
                st.session_state.chat_history.append({"role": "assistant", "content": "*Web search cancelled.*"})
                st.session_state.awaiting_web = False
                st.rerun()

query = st.chat_input("Ask a question about your study materials...", disabled=st.session_state.awaiting_web)

if query and not st.session_state.awaiting_web:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching local documents..."):
            query_embed = embedder.encode([query])
            D, I = st.session_state.faiss_index.search(np.array(query_embed), k=4)
            context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)])
            
            client = Groq(api_key=st.session_state.api_key)
            system_prompt = """You are a precise Study Assistant. Answer ONLY using the <context>. If missing, output EXACTLY: CONTEXT_MISSING"""
            user_prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}"
            
            res = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            
            response_text = res.choices[0].message.content.strip()
            
            if response_text == "CONTEXT_MISSING":
                st.session_state.awaiting_web = True
                st.session_state.pending_query = query
                st.rerun()
            else:
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
