import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from duckduckgo_search import DDGS

st.set_page_config(page_title="Chat Assistant | AI RAG", page_icon="💬", layout="wide")

st.title("💬 RAG Chat Assistant")

# Document upload is now OPTIONAL, so we only check for the API Key
    st.warning("Please configure your API Key on the Home page first.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_embedder()

# Define your model string here
MODEL_NAME = "openai/gpt-oss-120b"

# Render previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        client = Groq(api_key=st.session_state.api_key)
        
        # ---------------------------------------------------------
        # SCENARIO A: User uploaded a document
        # ---------------------------------------------------------
        if st.session_state.get("faiss_index") is not None:
            with st.spinner("Searching local documents..."):
                query_embed = embedder.encode([query])
                D, I = st.session_state.faiss_index.search(np.array(query_embed), k=4)
                context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)])
                
                system_prompt = """You are a precise Study Assistant. Answer ONLY using the <context>. If missing, output EXACTLY: CONTEXT_MISSING"""
                user_prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}"
                
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                )
                
                response_text = res.choices[0].message.content.strip()
                
                # Check if the document had the answer
                if response_text == "CONTEXT_MISSING":
                    st.info("💡 The answer was not found in your PDF. The AI is using its own knowledge to answer.")
                    
                    fallback_prompt = f"Answer this question comprehensively based on your general knowledge: {query}"
                    fallback_res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": fallback_prompt}]
                    )
                    
                    final_response = fallback_res.choices[0].message.content.strip()
                    st.markdown(final_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                else:
                    st.markdown(response_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # ---------------------------------------------------------
        # SCENARIO B: User did NOT upload any document
        # ---------------------------------------------------------
        else:
            with st.spinner("No documents found. Searching the internet..."):
                try:
                    results = DDGS().text(query, max_results=3)
                    web_context = "\n\n".join([f"Source: {r.get('href')}\n{r.get('body')}" for r in results])
                    
                    # Force the model to use ONLY the web results to prevent hallucination
                    prompt = f"Web Results:\n{web_context}\n\nQuestion: {query}\n\nAnswer thoroughly based STRICTLY on these web results."
                    res = client.chat.completions.create(
                        model=MODEL_NAME, 
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    display_text = f"{res.choices[0].message.content}\n\n*(Sourced from live web search)*"
                    st.markdown(display_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": display_text})
                except Exception as e:
                    st.error(f"Web search failed: {e}")
