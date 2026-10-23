
import os
import json
import re
from io import BytesIO
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
from docx import Document


# Load the API key from the local .env file
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL = "openai/gpt-oss-120b"

SECTIONS = [
    "Document A Summary",
    "Document B Summary",
    "Similarities",
    "Differences",
    "Key Details Comparison",
    "Overall Conclusion",
    "Limitations",
]


def extract_pdf(uploaded_file):
    """Extract text and page count from a PDF file."""
    reader = PdfReader(uploaded_file)

    if reader.is_encrypted:
        raise ValueError(
            f"{uploaded_file.name} is password-protected."
        )

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError(
            f"No extractable text was found in {uploaded_file.name}. "
            "The PDF may be scanned or image-based."
        )

    return full_text, len(reader.pages)


def compare_documents(text_a, text_b):
    """Ask Groq AI to compare two documents and return structured results."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Set it as an environment variable in your hosting settings."
        )

    client = Groq(api_key=api_key)

    # Limit input size to keep requests manageable.
    text_a = text_a[:40000]
    text_b = text_b[:40000]

    prompt = f"""
Compare the following two documents carefully.

Return ONLY a valid JSON object with exactly these seven keys:
{json.dumps(SECTIONS)}

Every value must be a clear, useful string.
If information is unavailable, explain that briefly.
Do not invent facts. Distinguish actual differences from missing
information. Use simple, professional English.

DOCUMENT A:
{text_a}

DOCUMENT B:
{text_b}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document comparison assistant. "
                    "Follow the requested JSON format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content or ""
    raw = raw.strip()

    # Support responses wrapped in Markdown code fences.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Extract the JSON object if extra text is returned.
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "The AI did not return valid JSON. Please try again."
        )

    result = json.loads(raw[start:end + 1])

    # Ensure every section is available.
    cleaned = {}
    for section in SECTIONS:
        value = result.get(section, "Not returned by the AI.")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        cleaned[section] = str(value)

    return cleaned


def create_report(results, file_a, file_b):
    """Create a downloadable Word comparison report."""
    doc = Document()

    doc.add_heading("AI Document Comparison Report", level=0)
    doc.add_paragraph(f"Document A: {file_a}")
    doc.add_paragraph(f"Document B: {file_b}")

    for section in SECTIONS:
        doc.add_heading(section, level=1)
        doc.add_paragraph(results.get(section, ""))

    output = BytesIO()
    doc.save(output)
    output.seek(0)

    return output.getvalue()


# ---------------- Streamlit interface ----------------

st.set_page_config(
    page_title="AI Document Comparison Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("AI Document Comparison Assistant")
st.write(
    "Upload two PDF documents to compare their content, "
    "identify similarities and differences, and generate a report."
)

col_a, col_b = st.columns(2)

with col_a:
    file_a = st.file_uploader(
        "Upload Document A (PDF)",
        type=["pdf"],
        key="document_a",
    )

with col_b:
    file_b = st.file_uploader(
        "Upload Document B (PDF)",
        type=["pdf"],
        key="document_b",
    )

text_a = None
text_b = None
pages_a = 0
pages_b = 0

if file_a is not None:
    try:
        text_a, pages_a = extract_pdf(file_a)
        st.success(f"Document A loaded: {pages_a} page(s)")
        with st.expander("Preview Document A text"):
            st.text(text_a[:3000])
    except Exception as exc:
        st.error(f"Could not read Document A: {exc}")

if file_b is not None:
    try:
        text_b, pages_b = extract_pdf(file_b)
        st.success(f"Document B loaded: {pages_b} page(s)")
        with st.expander("Preview Document B text"):
            st.text(text_b[:3000])
    except Exception as exc:
        st.error(f"Could not read Document B: {exc}")

if st.button(
    "Compare Documents",
    type="primary",
    use_container_width=True,
):
    if file_a is None or file_b is None:
        st.warning("Please upload both PDF documents first.")
    elif not text_a or not text_b:
        st.warning("Please upload two readable PDFs containing text.")
    else:
        try:
            with st.spinner("Analyzing both documents..."):
                results = compare_documents(text_a, text_b)

            st.session_state["comparison_results"] = results
            st.session_state["document_a_name"] = file_a.name
            st.session_state["document_b_name"] = file_b.name
            st.success("Document comparison completed!")

        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

if "comparison_results" in st.session_state:
    results = st.session_state["comparison_results"]

    st.subheader("Comparison Results")

    tabs = st.tabs(SECTIONS)

    for tab, section in zip(tabs, SECTIONS):
        with tab:
            st.markdown(results.get(section, "No result available."))

    report_bytes = create_report(
        results,
        st.session_state.get("document_a_name", "Document A"),
        st.session_state.get("document_b_name", "Document B"),
    )

    st.download_button(
        label="Download Comparison Report (Word)",
        data=report_bytes,
        file_name="document_comparison_report.docx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        use_container_width=True,
    )

    if st.button("Start New Comparison"):
        st.session_state.pop("comparison_results", None)
        st.session_state.pop("document_a_name", None)
        st.session_state.pop("document_b_name", None)
        st.rerun()
