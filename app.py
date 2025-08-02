import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import tempfile
import pymupdf

model_name = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def extract_questions_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    questions = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            for line in text.split("\n"):
                if "?" in line and len(line.strip()) > 10:
                    questions.append(line.strip())
    return questions

def generate_answer(question):
    prompt = f"""You are a helpful educational tutor. Provide a clear, detailed answer to the following question:\n{question}"""
    output = qa_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"].strip()

    if output.lower().strip() == question.lower().strip() or len(output) < 10:
        return "❌ Unable to generate a clear answer. Please rephrase or try again."
    return output

def chat_with_ai(user_input):
    prompt = f"""You are a friendly educational assistant. Explain clearly:\n{user_input}"""
    output = qa_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"].strip()

    # Fallback check
    if output.lower().strip() == user_input.lower().strip() or len(output) < 10:
        return "🤔 I'm not sure about that. Could you please rephrase your question?"
    return output

# Streamlit UI
st.set_page_config(page_title="StudyMate AI", layout="centered")
st.title("📖 STUDYMATE AI - Smart Question Bank with AI Answers")

uploaded_file = st.file_uploader("📄 Upload your Questions PDF (No Answers)", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded. Extracting questions and generating answers...")
    questions = extract_questions_from_pdf(tmp_path)

    if questions:
        st.markdown("### 🤖 AI Generated Answers:")
        for i, question in enumerate(questions):
            with st.spinner(f"Answering Question {i+1}..."):
                answer = generate_answer(question)
            st.markdown(f"**{i+1}. {question}**\n\n**Answer**: {answer}\n")
    else:
        st.warning("⚠️ No questions detected in the PDF.")

# AI Chatbot section
st.markdown("---")
st.markdown("### 🤖 Ask StudyMate AI Anything")
user_query = st.text_input("Type your question here:")
if user_query:
    with st.spinner("Thinking..."):
        bot_response = chat_with_ai(user_query)
    st.markdown(f"**AI Answer:** {bot_response}")
