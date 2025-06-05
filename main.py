# main.py

import streamlit as st
import os
import json
import google.generativeai as genai
from pypdf import PdfReader
from io import BytesIO

# --- Configuration & Gemini API Setup ---
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

# --- Helper Functions ---
def extract_text_from_pdf_bytes(pdf_bytes):
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

# --- Core Logic Functions ---
def screen_resume_llm(resume_text, job_description_text):
    if not resume_text: return {"error": "Resume text is empty."}
    if not job_description_text: return {"error": "Job description text is empty."}
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

def analyze_employee_sentiment_llm(feedback_text):
    if not feedback_text: return {"error": "Feedback text is empty."}
    prompt = f"""
    You are an expert HR AI Analyst specializing in employee engagement and sentiment analysis.
    Analyze the following employee feedback text.

    **Employee Feedback:**
    ---
    {feedback_text}
    ---

    Based *only* on the provided feedback, provide the following in JSON format:
    1.  `overall_sentiment`: Classify the sentiment as "Positive", "Negative", or "Neutral".
    2.  `sentiment_score`: A score from -1.0 (very negative) to 1.0 (very positive), reflecting the intensity of the sentiment.
    3.  `key_themes`: A list of 2-4 key themes or topics mentioned in the feedback (e.g., "Work-life balance", "Management", "Salary", "Growth opportunities", "Company culture").
    4.  `potential_attrition_risk`: Estimate the attrition risk based on this feedback as "Low", "Medium", or "High". Provide a brief justification.
    5.  `suggested_engagement_strategies`: Based *only* on the issues or positive points raised, suggest 1-2 actionable, specific engagement strategies HR could consider. If sentiment is positive, suggest how to reinforce it.

    Output *only* the JSON object. Do not include any other explanatory text before or after the JSON.
    """
    return call_gemini_api(prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="AI HR Tools", layout="wide")
st.title("🤖 AI-Powered HR Automation Suite")
st.markdown("Tools for Resume Screening and Employee Sentiment Analysis using Google Gemini.")

# --- 1. Resume Screening Section ---
with st.expander("📄 AI Resume Screener", expanded=True):
    st.header("Resume Screening")
    st.markdown("Upload a resume (PDF) and paste a job description to get an AI-driven analysis.")

    default_jd = """
    **Job Title: Software Engineer**

    We are looking for a passionate Software Engineer to design, develop, and install software solutions.
    The successful candidate will be able to build high-quality, innovative, and fully performing software
    in compliance with coding standards and technical design.

    **Requirements:**
    - Minimum 3 years of experience in software development using Python or Java.
    - Experience with cloud platforms like AWS or Azure.
    - BSc degree in Computer Science or relevant field.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Job Description")
        job_description_text = st.text_area("Paste the Job Description here:", value=default_jd, height=300, key="jd_input")
    with col2:
        st.subheader("📄 Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF resume...", type="pdf", key="resume_uploader")

    analyze_resume_button = st.button("✨ Analyze Resume", type="primary", key="analyze_resume_btn", use_container_width=True)

    if analyze_resume_button and uploaded_file and job_description_text:
        with st.spinner("Analyzing resume... 🧠"):
            resume_bytes = uploaded_file.getvalue()
            resume_text_content = extract_text_from_pdf_bytes(resume_bytes)

            if resume_text_content:
                screening_result = screen_resume_llm(resume_text_content, job_description_text)
                st.subheader("🔍 Resume Analysis Result:")
                if screening_result and "error" not in screening_result:
                    st.json(screening_result)
                    st.markdown("---")
                    st.markdown(f"**Overall Match Score:** `{screening_result.get('overall_match_score_percentage', 'N/A')}%`")
                    st.markdown(f"**Summary for Recruiter:**")
                    st.info(screening_result.get('summary_for_recruiter', 'N/A'))
                elif screening_result and "error" in screening_result:
                    st.error(f"Analysis Error: {screening_result.get('error')}")
            else:
                st.error("Could not extract text from the uploaded PDF.")
    elif analyze_resume_button:
        if not uploaded_file: st.warning("Please upload a resume PDF.")
        if not job_description_text: st.warning("Please provide a job description.")

st.markdown("---") # Separator

# --- 2. Employee Sentiment Analysis Section ---
with st.expander("💬 Employee Sentiment Analyzer", expanded=True):
    st.header("Employee Sentiment Analysis")
    st.markdown("Paste employee feedback (e.g., from surveys, exit interviews) to analyze sentiment and identify potential attrition risks.")

    default_feedback_positive = "I love working here! My team is fantastic, and I feel challenged and supported. The recent training on new technologies was excellent, and I see clear paths for growth."
    default_feedback_negative = "The workload has been insane lately, and I don't feel like management is listening to our concerns. I'm seriously considering looking for other opportunities if things don't change."
    
    feedback_examples = {
        "Positive Example": default_feedback_positive,
        "Negative Example": default_feedback_negative,
        "Neutral Example": "The job is fine. It pays the bills. Some days are good, some are okay.",
        "Custom": ""
    }
    selected_example = st.selectbox("Load Example Feedback or Enter Custom:", options=list(feedback_examples.keys()), key="feedback_example_select")
    
    if selected_example == "Custom":
        employee_feedback_text = st.text_area("Enter Employee Feedback Text:", height=200, key="feedback_input_custom")
    else:
        employee_feedback_text = st.text_area("Employee Feedback Text:", value=feedback_examples[selected_example], height=200, key="feedback_input_example")


    analyze_sentiment_button = st.button("📊 Analyze Sentiment", type="primary", key="analyze_sentiment_btn", use_container_width=True)

    if analyze_sentiment_button and employee_feedback_text:
        with st.spinner("Analyzing sentiment... 🧠"):
            sentiment_result = analyze_employee_sentiment_llm(employee_feedback_text)
            st.subheader("📈 Sentiment Analysis Result:")
            if sentiment_result and "error" not in sentiment_result:
                st.json(sentiment_result)
                # Display key fields more prominently
                st.markdown("---")
                st.markdown(f"**Overall Sentiment:** `{sentiment_result.get('overall_sentiment', 'N/A')}` (Score: `{sentiment_result.get('sentiment_score', 'N/A')}`)")
                st.markdown(f"**Potential Attrition Risk:**")
                risk = sentiment_result.get('potential_attrition_risk', 'N/A')
                if risk == "High":
                    st.error(f"🚨 {risk}")
                elif risk == "Medium":
                    st.warning(f"⚠️ {risk}")
                else:
                    st.success(f"✅ {risk}")
                
                if sentiment_result.get('key_themes'):
                    st.markdown(f"**Key Themes:**")
                    st.markdown(f"`{', '.join(sentiment_result.get('key_themes'))}`")
                
                if sentiment_result.get('suggested_engagement_strategies'):
                    st.markdown(f"**Suggested Engagement Strategies:**")
                    for strategy in sentiment_result.get('suggested_engagement_strategies'):
                        st.info(f"- {strategy}")

            elif sentiment_result and "error" in sentiment_result:
                st.error(f"Analysis Error: {sentiment_result.get('error')}")
    elif analyze_sentiment_button:
        st.warning("Please enter some employee feedback text.")


st.markdown("---")
st.caption("Powered by Google Gemini Pro & Streamlit")
