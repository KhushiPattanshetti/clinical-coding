import streamlit as st
from llm_interface import LLMInterface
from pipeline import ClinicalCodingPipeline

# Page config
st.set_page_config(
    page_title="Clinical ICD-10 Coding Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Clinical ICD-10 Coding Assistant")
st.caption("LLM-powered medical coding using Groq + ICD expansion + verification")

# Initialize pipeline (cached for performance)
@st.cache_resource
def load_pipeline():
    llm = LLMInterface()
    return ClinicalCodingPipeline(llm)

pipeline = load_pipeline()

# Input box
note = st.text_area(
    "📝 Paste Clinical Note:",
    height=200,
    placeholder="Example:\nPatient presents with left knee pain. MRI confirms osteoarthritis of left knee. Given NSAIDs and physiotherapy."
)

# Generate button
if st.button("🔍 Generate ICD-10 Codes"):
    if not note.strip():
        st.warning("Please enter a clinical note.")
    else:
        with st.spinner("Analyzing clinical note..."):
            results = pipeline.run(note)

        st.success("ICD-10 Codes Generated Successfully")

        st.subheader("📌 Final ICD-10 Codes")
        for r in results:
            st.write("•", r)

# Footer
st.markdown("---")
st.caption("Built with Groq LLaMA 3.3 + Clinical Coding Pipeline")
