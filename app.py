import streamlit as st
import fitz  # PyMuPDF
import json
import numpy as np
from groq import Groq

st.set_page_config(page_title="AI Study Guider", page_icon="🧠", layout="wide")

# --- Initialize Global Session States ---
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "study_materials" not in st.session_state:
    st.session_state.study_materials = {}
if "current_tabs" not in st.session_state:
    st.session_state.current_tabs = []

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

MODEL_NAME = "openai/gpt-oss-120b"

# ==========================================
# SIDEBAR: CONFIGURATION & DATA INGESTION
# ==========================================
with st.sidebar:
    st.title("⚙️ Setup & Documents")
    
    st.header("1. API Configuration")
    api_key = st.text_input("Groq API Key", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key
    elif "GROQ_API_KEY" in st.secrets:
        st.session_state.api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key loaded from Streamlit secrets.")

    st.header("2. Upload Materials (Optional)")
    st.markdown("Upload PDFs to base your study materials on specific documents. If left empty, the AI will use its general knowledge.")
    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)

    if st.button("Process Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Initializing AI embedding model..."):
            embedder = load_embedder()
            import faiss 
            
            all_text = ""
            for file in uploaded_files:
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
                st.success(f"Successfully processed {len(chunks)} text chunks!")
            else:
                st.error("No readable text could be extracted from the PDFs.")

# ==========================================
# MAIN PAGE: AI STUDY GUIDER
# ==========================================
st.title("🧠 AI Study Guider")
st.markdown("Generate personalized study materials from your uploaded documents or general AI knowledge.")

if not st.session_state.get("api_key"):
    st.warning("Please configure your API Key in the sidebar first.")
    st.stop()

client = Groq(api_key=st.session_state.api_key)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        goals = st.text_input("Learning Goals", placeholder="e.g., Polycarbonate co-extrusion parameters or LAT essay structure")
    with col2:
        duration = st.text_input("Study Duration", placeholder="e.g., 7 days, 4 hours")

    st.subheader("Generation Settings")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        tech_level = st.selectbox("Technicality Level", ["Beginner", "Intermediate", "Advanced"])
    with col4:
        response_size = st.selectbox("Response Size", ["Short", "Medium", "Detail", "Very Detailed"])
    with col5:
        language = st.selectbox("Answer Language", ["English", "Urdu", "Persian Urdu"])

    # Document-specific settings
    min_similarity = 0
    if st.session_state.get("faiss_index") is not None:
        st.markdown("**Document Search Settings**")
        min_similarity = st.slider("Minimum Context Similarity (%)", min_value=0, max_value=100, value=25, 
                                   help="Higher percentage requires a stricter match to your PDFs. If no chunks meet this threshold, the AI falls back to general knowledge.")

    st.subheader("Select Deliverables")
    deliverables = st.multiselect("What would you like to generate?", 
                                 ["Study Plan", "Notes", "Flashcards", "Exam Tips", "Quiz"])
    
    mcq_count = 5
    if "Quiz" in deliverables:
        mcq_count = st.slider("Number of Quiz Questions", min_value=3, max_value=20, value=5)

# --- PHASE 1: DATA FETCHING ---
if st.button("Generate Study Materials", type="primary"):
    if not goals:
        st.error("Please enter your learning goals.")
        st.stop()
        
    context = ""
    
    if st.session_state.get("faiss_index") is not None:
        with st.spinner("Retrieving relevant context from PDFs..."):
            embedder = load_embedder()
            query_embed = embedder.encode([goals])
            D, I = st.session_state.faiss_index.search(np.array(query_embed), k=8)
            
            # Convert percentage to FAISS L2 distance threshold
            min_cos_sim = min_similarity / 100.0
            l2_threshold = 2.0 - (2.0 * min_cos_sim)
            
            valid_chunks = []
            for i in range(len(I[0])):
                idx = I[0][i]
                dist = D[0][i]
                if idx < len(st.session_state.chunks) and dist <= l2_threshold:
                    valid_chunks.append(st.session_state.chunks[idx])
            
            context = "\n\n---\n\n".join(valid_chunks)
            if not context:
                st.info("No document content met the strict similarity threshold. Proceeding with general AI knowledge.")

    st.session_state.study_materials = {}
    st.session_state.current_tabs = deliverables
    
    for item in deliverables:
        with st.spinner(f"Generating {item} in {language}..."):
            try:
                # Dynamic modifier string for the prompt
                modifiers = f"Target Audience: {tech_level}.\nLanguage: {language}.\n"
                if item in ["Study Plan", "Notes", "Exam Tips"]:
                    modifiers += f"Required Length/Detail: {response_size}.\n"

                if item == "Study Plan":
                    strict_task = f"Create a structured study plan for '{goals}' spanning '{duration}'. Use Markdown tables.\n{modifiers}"
                    fallback_task = f"Create a structured study plan for '{goals}' spanning '{duration}'. Use your general AI knowledge and format with Markdown tables.\n{modifiers}"
                
                elif item == "Notes":
                    strict_task = f"Generate notes for the topic: '{goals}'.\n{modifiers}"
                    fallback_task = f"Generate notes for the topic: '{goals}' using your general AI knowledge.\n{modifiers}"
                
                elif item == "Exam Tips":
                    strict_task = f"Provide top exam preparation tips and common pitfalls regarding: '{goals}'.\n{modifiers}"
                    fallback_task = f"Provide top exam preparation tips and common pitfalls regarding: '{goals}' using your general AI knowledge.\n{modifiers}"
                
                elif item == "Flashcards":
                    strict_task = f"Create 5-10 flashcards for '{goals}'.\n{modifiers} ONLY output valid JSON format: {{\"flashcards\": [{{\"term\": \"X\", \"definition\": \"Y\"}}]}}"
                    fallback_task = f"Create 5-10 flashcards for '{goals}' using general AI knowledge.\n{modifiers} ONLY output valid JSON format: {{\"flashcards\": [{{\"term\": \"X\", \"definition\": \"Y\"}}]}}"
                
                elif item == "Quiz":
                    strict_task = f"Generate a multiple-choice quiz with {mcq_count} questions about '{goals}'.\n{modifiers} ONLY output valid JSON format: {{\"quiz\": [{{\"question\": \"Q\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"A\"}}]}}. Do NOT wrap in markdown."
                    fallback_task = f"Generate a multiple-choice quiz with {mcq_count} questions about '{goals}' using general AI knowledge.\n{modifiers} ONLY output valid JSON format: {{\"quiz\": [{{\"question\": \"Q\", \"options\": [\"A\", \"B\", \"C\", \"D\"], \"answer\": \"A\"}}]}}. Do NOT wrap in markdown."

                output = ""

                if context:
                    eval_prompt = f"Context:\n{context}\n\nTask: {strict_task}\n\nCRITICAL INSTRUCTION: If the Context does not contain enough information about '{goals}' to fulfill this task, you MUST output exactly this string and nothing else: CONTEXT_MISSING"
                    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}])
                    output = res.choices[0].message.content.strip()

                    if output == "CONTEXT_MISSING":
                        st.info(f"💡 Uploaded documents lacked specific details for '{item}'. Generating using general AI knowledge.")
                        fallback_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": fallback_task}])
                        output = fallback_res.choices[0].message.content.strip()
                
                else:
                    fallback_res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": fallback_task}])
                    output = fallback_res.choices[0].message.content.strip()

                st.session_state.study_materials[item] = output

            except Exception as api_error:
                st.error(f"API Error during {item} generation: {api_error}")

# --- PHASE 2: UI RENDERING ---
if st.session_state.study_materials:
    tabs = st.tabs(st.session_state.current_tabs)
    
    for i, item in enumerate(st.session_state.current_tabs):
        with tabs[i]:
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
                        
                        ans = st.radio("Select an answer:", options=q.get('options', []), key=f"q_{item}_{q_idx}", index=None)
                        
                        if ans is not None:
                            if ans == q.get('answer'):
                                st.success("✅ Correct!")
                            else:
                                st.error(f"❌ Incorrect. The correct answer is: **{q.get('answer')}**")
                        
                        st.divider()
                        
                except Exception as e:
                    st.error("Failed to parse JSON for Quiz. Ensure the LLM outputs strict JSON.")
                    st.code(output)
