import streamlit as st
import os
import json
import google.generativeai as genai
from pypdf import PdfReader # Using pypdf
from io import BytesIO # To handle uploaded file in memory

# --- Configuration & Gemini API Setup ---
# Attempt to get API key from Streamlit secrets first, then environment variable
try:
    # For Streamlit Cloud deployment, set GOOGLE_API_KEY in an st.secrets.toml file
    # or directly in the app's secrets settings on Streamlit Community Cloud.
    # Example st.secrets.toml:
    # GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
    API_KEY = "AIzaSyAddh04FxC9YPL6LfDGiVEgJcU1khxKLpA"
    # if not API_KEY: # Fallback to environment variable if not in secrets
    #     API_KEY = os.environ.get("GOOGLE_API_KEY")
except AttributeError: # If st.secrets is not available (e.g. local run without secrets file)
    API_KEY = os.environ.get("GOOGLE_API_KEY")


if not API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your Streamlit secrets or environment variables.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"🚨 Error configuring Gemini API: {e}")
    st.stop()


GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS
)

# --- Helper Functions (adapted from previous script) ---
def extract_text_from_pdf_bytes(pdf_bytes):
    """Extracts text from PDF bytes."""
    try:
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def call_gemini_api(prompt_text):
    """Generic function to call the Gemini API and parse JSON output."""
    try:
        response = model.generate_content(prompt_text)
        raw_response_text = response.text
        
        if raw_response_text.strip().startswith("```json"):
            cleaned_text = raw_response_text.strip()[7:-3].strip()
        elif raw_response_text.strip().startswith("```"):
             cleaned_text = raw_response_text.strip()[3:-3].strip()
        else:
            cleaned_text = raw_response_text.strip()
            
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        st.error(f"🚨 Error decoding JSON from LLM response: {e}")
        st.error(f"Raw response was:\n---\n{raw_response_text}\n---")
        return {"error": "Failed to parse LLM response as JSON", "details": str(e), "raw_response": raw_response_text}
    except Exception as e:
        st.error(f"🚨 Error calling Gemini API or processing response: {e}")
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            st.error(f"Content blocked due to: {response.prompt_feedback.block_reason}")
            return {"error": f"Content blocked: {response.prompt_feedback.block_reason}"}
        return {"error": "An unexpected error occurred with the Gemini API.", "details": str(e)}


def screen_resume_llm(resume_text, job_description_text):
    """
    Uses Gemini to screen a resume against a job description.
    (Renamed from screen_resume to avoid conflict if you also want the sentiment part)
    """
    if not resume_text:
        return {"error": "Resume text is empty."}
    if not job_description_text:
        return {"error": "Job description text is empty."}

    prompt = f"""
    You are an expert HR AI assistant specializing in screening resumes for technical roles.
    Your task is to analyze the provided resume text against the given job description for a "Software Engineer" position.

    **Job Description:**
    ---
    {job_description_text}
    ---

    **Resume Text:**
    ---
    {resume_text}
    ---

    Based *only* on the information present in the resume text and its relevance to the job description, provide the following in JSON format:
    1.  `extracted_skills`: A list of key technical skills mentioned in the resume relevant to the job description (e.g., "Python", "Java", "React", "AWS", "Docker", "Kubernetes", "SQL", "Git").
    2.  `years_of_experience`: Estimate the total years of relevant professional software engineering experience. If not explicitly stated, infer based on work history dates. State "Not specified" if unclear.
    3.  `education_match`: Briefly state if the education level (e.g., Bachelor's, Master's in CS or related) aligns with typical expectations for a Software Engineer, if mentioned.
    4.  `key_qualifications_match`: A list of 3-5 key qualifications or accomplishments from the resume that directly match requirements or desired skills in the job description.
    5.  `missing_critical_skills`: A list of critical skills mentioned in the job description that appear to be missing or not emphasized in the resume.
    6.  `overall_match_score_percentage`: An estimated percentage (0-100) indicating how well the candidate's profile matches the job description. Provide a brief justification for this score.
    7.  `summary_for_recruiter`: A concise 2-3 sentence summary highlighting the candidate's fit and any red flags for the recruiter.

    Output *only* the JSON object. Do not include any other explanatory text before or after the JSON.
    """
    return call_gemini_api(prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI-Powered Resume Screener")
st.markdown("Upload a resume (PDF) and paste a job description to get an AI-driven analysis.")

# Default Job Description (can be edited by user)
default_jd = """
**Job Title: Software Engineer**

We are looking for a passionate Software Engineer to design, develop, and install software solutions.
The successful candidate will be able to build high-quality, innovative, and fully performing software
in compliance with coding standards and technical design.

**Responsibilities:**
- Execute full lifecycle software development (SDLC).
- Write well-designed, testable, efficient code.
- Produce specifications and determine operational feasibility.
- Integrate software components into a fully functional software system.
- Develop software verification plans and quality assurance procedures.
- Document and maintain software functionality.
- Troubleshoot, debug and upgrade existing systems.
- Deploy programs and evaluate user feedback.
- Comply with project plans and industry standards.
- Ensure software is updated with latest features.

**Requirements:**
- Proven work experience as a Software Engineer or Software Developer.
- Minimum 3 years of experience in software development.
- Experience with test-driven development.
- Proficiency in software engineering tools.
- Ability to develop software in Python, Java, or C++.
- Experience with databases such as SQL, PostgreSQL.
- Experience with cloud platforms like AWS or Azure is a plus.
- BSc degree in Computer Science, Engineering or relevant field.
- Excellent problem-solving skills and attention to detail.
- Good communication skills.
"""

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Job Description")
    job_description_text = st.text_area("Paste the Job Description here:", value=default_jd, height=400)
    
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF resume...", type="pdf")

analyze_button = st.button("✨ Analyze Resume", type="primary", use_container_width=True)

if analyze_button and uploaded_file and job_description_text:
    with st.spinner("Analyzing resume... This may take a moment. 🧠"):
        resume_bytes = uploaded_file.getvalue()
        resume_text_content = extract_text_from_pdf_bytes(resume_bytes)

        if resume_text_content:
            # st.subheader("Extracted Resume Text (First 500 chars):") # For debugging
            # st.text(resume_text_content[:500] + "...")
            
            screening_result = screen_resume_llm(resume_text_content, job_description_text)
            
            st.subheader("🔍 Analysis Result:")
            if screening_result and "error" not in screening_result:
                st.json(screening_result)

                # Optionally, display key fields more prominently
                st.markdown("---")
                st.markdown(f"**Overall Match Score:** `{screening_result.get('overall_match_score_percentage', 'N/A')}%`")
                st.markdown(f"**Summary for Recruiter:**")
                st.info(screening_result.get('summary_for_recruiter', 'N/A'))
                
                if screening_result.get('extracted_skills'):
                    st.markdown(f"**Extracted Skills:**")
                    st.markdown(f"`{', '.join(screening_result.get('extracted_skills'))}`")

            elif screening_result and "error" in screening_result:
                st.error(f"Analysis Error: {screening_result.get('error')}")
                if screening_result.get('details'):
                    st.error(f"Details: {screening_result.get('details')}")
                if screening_result.get('raw_response'):
                     st.expander("Show Raw LLM Response").warning(screening_result.get('raw_response'))

        else:
            st.error("Could not extract text from the uploaded PDF.")
elif analyze_button:
    if not uploaded_file:
        st.warning("Please upload a resume PDF.")
    if not job_description_text:
        st.warning("Please provide a job description.")

st.markdown("---")
st.caption("Powered by Google Gemini Pro & Streamlit")