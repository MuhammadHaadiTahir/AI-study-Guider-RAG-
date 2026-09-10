import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Study Guider | AI RAG", page_icon="🧠", layout="wide")

st.title("🧠 AI Study Guider")
st.markdown("Generate personalized study materials from your uploaded documents. If a topic isn't in your PDFs, the AI will use its general knowledge.")

if not st.session_state.get("faiss_index") or not st.session_state.get("api_key"):
    st.warning("Please configure your API Key and upload documents on the Home page first.")
    st.stop()

# Initialize Session State for preserving generated materials across reruns
if "study_materials" not in st.session_state:
    st.session_state.study_materials = {}
if "current_tabs" not in st.session_state:
    st.session_state.current_tabs = []

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()
client = Groq(api_key=st.session_state.api_key)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        goals = st.text_input("Learning Goals", placeholder="e.g., Polycarbonate co-extrusion optimization or LAT essay structure")
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

MODEL_NAME = "openai/gpt-oss-120b"

# --- PHASE 1: DATA FETCHING ---
if st.button("Generate Study Materials", type="primary"):
    if not goals:
        st.error("Please enter your learning goals.")
        st.stop()
        
    with st.spinner("Retrieving relevant context..."):
        query_embed = embedder.encode([goals])
        D, I = st.session_state.faiss_index.search(np.array(query_embed), k=6)
        context = "\n\n---\n\n".join([st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)])

    # Clear previous memory and prep for new generation
    st.session_state.study_materials = {}
    st.session_state.current_tabs = deliverables
    
    for item in deliverables:
        with st.spinner(f"Evaluating documents for {item}..."):
            try:
                strict_task = ""
                fallback_task = ""
                
                if item == "Study Plan":
                    strict_task = f"Create a structured study plan for '{goals}' spanning '{duration}'. Use Markdown tables."
                    fallback_task = f"Create a structured study plan for '{goals}' spanning '{duration}'. Use your general AI knowledge and format with Markdown tables."
                elif item == "Notes":
                    strict_task = f"Generate {notes_len} notes for the topic: '{goals}'."
                    fallback_task = f"Generate {notes_len} notes for the topic: '{goals}' using your general AI knowledge."
                elif item == "Exam Tips":
                    strict_task = f"Provide top exam preparation tips and common pitfalls regarding: '{goals}'."
                    fallback_task = f"Provide top exam preparation tips and common pitfalls regarding: '{goals}' using your general AI knowledge."
                elif item == "Flashcards":
                    strict_task = f"Create 5-10 flashcards for '{goals}'. ONLY output valid JSON format: {{\"flashcards\": [{{\"term\": \"X\", \"definition\": \"Y\"}}]}}"
                    fallback_task = f"Create 5-10 flashcards for '{goals}' using general AI knowledge. ONLY output valid JSON format: {{\"flashcards\": [{{\"term\": \"X\", \"definition\": \"Y\"}}]}}"
                elif item == "Quiz":
                    strict_task = f"Generate a multiple-choice quiz with {mcq_count} questions about '{goals}'. ONLY output valid JSON format: {{\"quiz\": [{{\"question\": \"Q\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"A\"}}]}}. Do NOT wrap in markdown."
                    fallback_task = f"Generate a multiple-choice quiz with {mcq_count} questions about '{goals}' using general AI knowledge. ONLY output valid JSON format: {{\"quiz\": [{{\"question\": \"Q\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"A\"}}]}}. Do NOT wrap in markdown."

                eval_prompt = f"Context:\n{context}\n\nTask: {strict_task}\n\nCRITICAL INSTRUCTION: If the Context does not contain enough information about '{goals}' to fulfill this task, you MUST output exactly this string and nothing else: CONTEXT_MISSING"
                
                res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}])
                output = res.choices[0].message.content.strip()

                if output == "CONTEXT_MISSING":
                    st.info(f"💡 '{goals}' was not found in your uploaded PDFs. Generating {item} using general AI knowledge.")
                    fallback_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": fallback_task}])
                    output = fallback_res.choices[0].message.content.strip()

                # Store the API output into session state memory instead of printing immediately
                st.session_state.study_materials[item] = output

            except Exception as api_error:
                st.error(f"API Error during {item} generation: {api_error}")

# --- PHASE 2: UI RENDERING ---
# This block runs automatically every time the app reruns, preserving your quiz state.
if st.session_state.study_materials:
    tabs = st.tabs(st.session_state.current_tabs)
    
    for i, item in enumerate(st.session_state.current_tabs):
        with tabs[i]:
            # Retrieve the saved text for this specific tab
            output = st.session_state.study_materials.get(item, "")
            
            if item in ["Study Plan", "Notes", "Exam Tips"]:
                st.markdown(output)
                
            elif item == "Flashcards":
                try:
                    fc_data = json.loads(output)
                    for fc in fc_data.get("flashcards", []):
                        with st.expander(f"**{fc.get('term', 'Term')}**"):
                            st.write(fc.get('definition', 'Definition'))
                except Exception as e:
                    st.error("Failed to parse JSON for Flashcards.")
                    st.code(output)
                    
            elif item == "Quiz":
                try:
                    quiz_data = json.loads(output)
                    for q_idx, q in enumerate(quiz_data.get("quiz", [])):
                        st.markdown(f"**Q{q_idx + 1}: {q.get('question')}**")
                        
                        # The radio button starts empty (index=None)
                        ans = st.radio("Select an answer:", options=q.get('options', []), key=f"q_{item}_{q_idx}", index=None)
                        
                        # Only show the result AFTER the user clicks an option
                        if ans is not None:
                            if ans == q.get('answer'):
                                st.success("✅ Correct!")
                            else:
                                st.error(f"❌ Incorrect. The correct answer is: **{q.get('answer')}**")
                        
                        st.divider() # Adds a clean line between questions
                        
                except Exception as e:
                    st.error("Failed to parse JSON for Quiz. Ensure the LLM outputs strict JSON.")
                    st.code(output)
