# AI Document Comparison Assistant

## Project Overview
The AI Document Comparison Assistant is a Python-based web application that uses Artificial Intelligence to compare two PDF documents. It identifies similarities, differences, important details, and provides an overall conclusion.

## Features
- Upload two PDF documents.
- Extract and preview PDF text.
- Compare documents using the Groq AI API.
- Display results in seven sections.
- Download the comparison report as a Word document.

## Technologies Used
- Python
- Streamlit
- Groq API
- pypdf
- python-dotenv
- python-docx

## How It Works
1. Upload two PDF documents.
2. Extract text from both documents.
3. Send the document content to the AI model for comparison.
4. Display summaries, similarities, differences, key details, conclusion, and limitations.
5. Download the generated Word report.

## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration
Create a local `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace `your_groq_api_key` with your own API key. Never upload the `.env` file or reveal your actual API key.

## Run the Application
Run the following command in your Python environment:

```bash
streamlit run app.py
```

## Conclusion
This project simplifies document comparison by combining Python, a web interface, and AI-powered text analysis to generate an organized, downloadable report.
