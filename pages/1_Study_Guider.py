import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Study Guider | AI RAG", page_icon="🧠", layout="wide")

st.title("🧠 AI Study Guider")
st.markdown("Generate personalized study materials from your uploaded documents.")

if not st.session_state.get("faiss_index") or not st.session_state.get("api_key"):
    st.warning("Please configure your API Key and upload documents on the Home page first.")
    st.stop()

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedder = load_embedder()
client = Groq(api_key=st.session_state.api_key)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        goals = st.text_input("Learning Goals", placeholder="e.g., Understand twin-screw compounding parameters")
    with col2:
        duration = st.text_input("Study Duration", placeholder="e.g., 7 days, 4 hours")

    st.subheader("Select Deliverables")
    deliverables = st.multiselect("What would you like to generate?", 
                                 ["Study Plan", "Notes", "Flashcards", "Exam Tips", "Quiz"])
    
    notes_len = "Medium"
    mcq_count = 5
    
    if "Notes" in deliverables:
        notes_len = st.selectbox("Notes Length", ["Short (Summary)", "Medium (Outline)", "Long (Detailed)"])
    if "Quiz" in deliverables:
        mcq_count = st.slider("Number of Quiz Questions", min_value=3, max_value=20, value=5)

if st.button("Generate Study Materials", type="primary"):
    if not goals:
        st.error("Please enter your learning goals.")
        st.stop()
        
    with st.spinner("Retrieving relevant context..."):
        query_embed = embedder.encode([goals])
        D, I = st.session_state.faiss_index.search(np.array(query_embed), k=6)
        context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)])

    tabs = st.tabs(deliverables) if deliverables else []
    
    for i, item in enumerate(deliverables):
        with tabs[i]:
            with st.spinner(f"Generating {item}..."):
                if item == "Study Plan":
                    prompt = f"Context:\n{context}\n\nCreate a structured study plan for '{goals}' spanning '{duration}'. Use Markdown tables where appropriate."
                    res = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
                    st.markdown(res.choices[0].message.content)
                    
                elif item == "Notes":
                    prompt = f"Context:\n{context}\n\nGenerate {notes_len} notes for the topic: '{goals}'."
                    res = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
                    st.markdown(res.choices[0].message.content)
                    
                elif item == "Exam Tips":
                    prompt = f"Context:\n{context}\n\nProvide top exam preparation tips and common pitfalls regarding: '{goals}'."
                    res = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
                    st.markdown(res.choices[0].message.content)
                    
                elif item == "Flashcards":
                    prompt = f"Context:\n{context}\n\nCreate 5-10 flashcards for '{goals}'. ONLY output valid JSON like this: {{\"flashcards\": [{{\"term\": \"X\", \"definition\": \"Y\"}}]}}"
                    res = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
                    try:
                        fc_data = json.loads(res.choices[0].message.content.strip())
                        for fc in fc_data.get("flashcards", []):
                            with st.expander(f"**{fc.get('term', 'Term')}**"):
                                st.write(fc.get('definition', 'Definition'))
                    except:
                        st.error("Failed to parse JSON.")
                        st.write(res.choices[0].message.content)
                        
                elif item == "Quiz":
                    prompt = f"Context:\n{context}\n\nGenerate a multiple-choice quiz with {mcq_count} questions about '{goals}'. ONLY output valid JSON. Format: {{\"quiz\": [{{\"question\": \"Q\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"A\"}}]}}. DO NOT wrap in markdown block."
                    res = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
                    try:
                        quiz_data = json.loads(res.choices[0].message.content.strip())
                        for q_idx, q in enumerate(quiz_data.get("quiz", [])):
                            st.markdown(f"**Q{q_idx + 1}: {q.get('question')}**")
                            ans = st.radio("Select an answer:", options=q.get('options', []), key=f"q_{q_idx}", index=None)
                            with st.expander("Show Answer"):
                                st.success(f"Correct Answer: {q.get('answer')}")
                    except Exception as e:
                        st.error("Failed to parse JSON. Ensure the LLM outputs strict JSON.")
                        st.write(res.choices[0].message.content)
