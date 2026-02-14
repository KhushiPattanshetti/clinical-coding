from llm_interface import LLMInterface
from pipeline import ClinicalCodingPipeline

note = """
Patient presents with left knee pain. MRI confirms osteoarthritis of left knee.
Given NSAIDs and physiotherapy.
"""

llm = LLMInterface()   # uses Groq + LLaMA 3.1 by default

print("Testing Groq connection...")
print(llm.generate("Say hello in one word."))

pipeline = ClinicalCodingPipeline(llm)
results = pipeline.run(note)

print("\nFinal ICD-10 Codes:")
for r in results:
    print("•", r)
