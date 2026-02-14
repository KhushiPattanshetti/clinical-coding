**Clinical Coding System – Classical & LLM-Based Pipelines**

A hybrid clinical coding system that automatically extracts ICD-10 medical codes from unstructured doctors' notes using two approaches:

1. Classical Rule-Based Pipeline

2. LLM-Powered Pipeline (Groq) with Streamlit UI

This project demonstrates how traditional NLP + medical ontologies and modern large language models (LLMs) can be combined to build accurate and scalable clinical decision-support systems.


**1. Classical Pipeline**
Approach

Uses:

Rule-based NLP

ICD-10 hierarchy traversal

Keyword matching

Ontology-based expansion

Strengths

Deterministic

Explainable

No API cost

Fast execution

Limitations

Lower accuracy for complex notes

Cannot generalize beyond handcrafted rules

**2. LLM Pipeline (Main Contribution)**
Approach

Uses:

Large Language Models (via Groq / OpenAI APIs)

Prompt engineering

Multi-step verification pipeline

Pipeline Stages:

Generator Stage – Extract ICD codes

Expansion Stage – Generate nearby ICD candidates

Verifier Stage – Choose best matching code

Strengths

High accuracy

Understands context

Handles complex language

Medical reasoning capability

Installation & Setup
1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate     # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Setup Environment Variables

Create .env file inside llm_pipeline/

GROQ_API_KEY=your_groq_api_key_here

Get free Groq API key: https://console.groq.com

4️⃣ Run LLM Pipeline
python complete_example.py

5️⃣ Run Web UI
streamlit run app.py

**Sample Input**
Patient presents with left knee pain. MRI confirms osteoarthritis of left knee.
Given NSAIDs and physiotherapy.

**Output**
M17.12: Unilateral primary osteoarthritis, left knee
M25.562: Pain in left knee
