# Clinical Coding Pipeline

Implementation of the paper: **"Leveraging LLMs for Clinical Coding with Structured Verification"**

## Overview

This repository implements a complete pipeline for automated ICD-10-CM clinical coding using Large Language Models (LLMs) with hierarchical verification. The system addresses the challenge of assigning correct billing codes from a set of 72,000 ICD-10-CM codes based on clinical notes.

## Key Features

### 1. **Generate-Expand-Verify Pipeline**
- **Generate**: Multiple prompt engineering strategies for code generation
- **Expand**: Hierarchical expansion using ICD-10-CM structure (siblings, cousins, neighbors)
- **Verify**: LLM-based verification to select the most appropriate code

### 2. **Prompt Engineering Strategies**
- Single-line baseline
- Detailed instructions with MEAT principle
- Chain-of-thought reasoning
- Prompt decomposition
- Combined strategies

### 3. **Hierarchical Code Expansion**
Based on ICD-10-CM structure:
- **Siblings S(c)**: Codes sharing the same parent
- **Cousins C(c)**: Codes sharing the same grandparent
- **1-hop neighbors N₁(c)**: Direct cross-references
- **2-hop neighbors N₂(c)**: Two-step cross-references

### 4. **Comprehensive Evaluation Metrics**
- **Exact Match**: Standard precision, recall, F1
- **Prefix-n Match**: Hierarchical matching at different levels
- **Prefix Overlap Ratio (POR)**: Weighted hierarchical similarity
- **Verification Accuracy**: Standalone verification performance

## Project Structure

```
clinical_coding_pipeline/
├── utils/
│   └── icd_hierarchy.py          # ICD-10-CM hierarchy management
├── models/
│   └── code_generator.py          # Prompt strategies & code generation
├── verification/
│   └── verifier.py                # Expand-verify pipeline
├── evaluation/
│   └── metrics.py                 # Evaluation metrics
├── data/
│   └── sample_icd_hierarchy.json  # Sample hierarchy data
├── demo.py                        # Comprehensive demo
└── README.md                      # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run the Demo

```bash
python demo.py
```

This runs all demonstrations including:
1. Prompt engineering strategies
2. Candidate expansion
3. Code verification
4. Full pipeline evaluation
5. Metric comparisons

### Use Individual Components

#### 1. ICD-10-CM Hierarchy

```python
from utils.icd_hierarchy import ICDHierarchy, create_sample_hierarchy

# Create hierarchy
hierarchy = create_sample_hierarchy()

# Expand a code
code = "M25.561"  # Pain in right knee
siblings = hierarchy.get_siblings(code)
cousins = hierarchy.get_cousins(code)
all_candidates = hierarchy.expand_candidates(code)

print(f"Siblings: {siblings}")
print(f"Cousins: {cousins}")
```

#### 2. Code Generation with Prompts

```python
from models.code_generator import (
    ClinicalNote, PromptGenerator, PromptType
)

# Create clinical note
note = ClinicalNote(
    text="Patient presents with left knee pain...",
    note_id="NOTE001"
)

# Generate different prompts
simple_prompt = PromptGenerator.get_prompt(note, PromptType.SINGLE_LINE)
detailed_prompt = PromptGenerator.get_prompt(note, PromptType.DETAILED_COT)

print(simple_prompt)
```

#### 3. Verification Pipeline

```python
from verification.verifier import (
    ClinicalCodingPipeline, ExpansionType, VerificationPromptType
)

# Create pipeline
pipeline = ClinicalCodingPipeline(hierarchy)

# Run verification
initial_predictions = ["M25.561"]  # Wrong: right knee
verified_codes = pipeline.run_pipeline(
    note=note,
    initial_predictions=initial_predictions,
    expansion_type=ExpansionType.ALL,
    verification_prompt=VerificationPromptType.DESCRIPTION_ONLY
)

print(f"Initial: {initial_predictions}")
print(f"Verified: {verified_codes}")
```

#### 4. Evaluation Metrics

```python
from evaluation.metrics import ClinicalCodingMetrics

# Create metrics calculator
metrics = ClinicalCodingMetrics(hierarchy)

# Evaluate predictions
predicted = [["M25.561", "I11.0"]]
gold = [["M25.562", "I11.0"]]

results = metrics.comprehensive_evaluation(predicted, gold)
print(f"Exact Match F1: {results['exact_match_f1']:.3f}")
print(f"Prefix-1 F1: {results['prefix_1_f1']:.3f}")
print(f"POR: {results['prefix_overlap_ratio']:.3f}")
```

## Key Results from Paper

The paper reports significant improvements using the verification pipeline:

| Model | Generation (F1) | + Verification (F1) | Improvement |
|-------|----------------|-------------------|-------------|
| Haiku-3 | 41.6 | 47.2 | +5.6 |
| Haiku-3 (Fine-tuned) | 56.9 | 57.6 | +0.7 |
| Sonnet-3.5v1 | 55.6 | 55.5 | -0.1 |
| PLM-ICD | 24.8 | 29.4 | +4.6 |

**Key Findings:**
- Description-only prompts perform best for verification (88-90% accuracy)
- Siblings are hardest to distinguish (similar semantics)
- Most errors occur at leaf/leaf-1 levels (laterality, complications)

## Extending the Implementation

### 1. Load Full ICD-10-CM Data

Replace the sample hierarchy with complete ICD-10-CM codes:

```python
# Load from official CMS files
hierarchy = ICDHierarchy()
# Parse ICD-10-CM tabular and index files
# hierarchy.add_code(...) for each code
hierarchy.save_to_file("icd10cm_full.json")
```

### 2. Integrate with LLM APIs

```python
def model_generate_fn(prompt: str) -> str:
    """Call your LLM API here."""
    # Example with Anthropic Claude
    import anthropic
    client = anthropic.Anthropic(api_key="...")
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Use in pipeline
verified = pipeline.run_pipeline(
    note=note,
    initial_predictions=predictions,
    model_fn=model_generate_fn
)
```

### 3. Fine-tuning

```python
# Prepare fine-tuning data
training_data = [
    {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Code: M25.562\nDescription: Pain in left knee"}
        ]
    }
    # ... more examples
]

# Fine-tune using your LLM provider's API
# Then use fine-tuned model in pipeline
```

### 4. Add Real Clinical Notes

```python
# Load from your dataset
import pandas as pd

notes_df = pd.read_csv("clinical_notes.csv")

for _, row in notes_df.iterrows():
    note = ClinicalNote(
        text=row['note_text'],
        note_id=row['note_id']
    )
    
    # Generate codes
    predictions = generate_codes(note)
    
    # Verify
    verified = pipeline.run_pipeline(note, predictions)
    
    # Evaluate
    if 'gold_codes' in row:
        metrics.evaluate(verified, row['gold_codes'])
```

## Common Error Patterns

Based on the paper's clinical analysis:

1. **Symptom vs. Etiology**: Models code symptoms (e.g., R26.2 - difficulty walking) when underlying cause is documented (e.g., M51.27 - disc displacement)

2. **Suspected vs. Confirmed**: Models code "possible" conditions in outpatient settings (should only code confirmed diagnoses)

3. **Laterality**: Confusion between left/right (e.g., M25.561 vs M25.562)

4. **Complication Status**: With vs. without complications (e.g., I11.0 vs I11.9)

The verification pipeline specifically addresses these by:
- Expanding to include siblings (catches laterality)
- Expanding to include family members (catches complication status)
- Using description-based verification (improves semantic understanding)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{clinical_coding_2024,
  title={Leveraging LLMs for Clinical Coding with Structured Verification},
  author={[Authors from paper]},
  year={2024}
}
```

