"""
Code Generation Module
Implements various prompting strategies for clinical code generation.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Different prompt engineering strategies."""
    SINGLE_LINE = "single_line"
    DETAILED = "detailed"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DETAILED_COT = "detailed_cot"
    PROMPT_DECOMPOSITION = "prompt_decomposition"


@dataclass
class ClinicalNote:
    """Represents a clinical note."""
    text: str
    note_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class CodePrediction:
    """Represents a predicted ICD-10-CM code."""
    code: str
    description: str
    confidence: Optional[float] = None


class PromptGenerator:
    """Generates prompts for clinical coding based on different strategies."""
    
    @staticmethod
    def single_line_prompt(note: ClinicalNote) -> str:
        """
        Simple baseline prompt from prior work (Boyle et al., 2023).
        """
        return f"""Given the following clinical note, predict the ICD-10-CM billing codes.

Clinical Note:
{note.text}

Predicted ICD-10-CM Codes (comma-separated):"""
    
    @staticmethod
    def detailed_prompt(note: ClinicalNote) -> str:
        """
        Detailed instructions incorporating MEAT principle.
        MEAT = Monitor, Evaluate, Assess, Treat
        """
        return f"""You are an expert medical coder. Given the clinical note below, assign appropriate ICD-10-CM billing codes.

Instructions:
1. Read the clinical note carefully
2. Identify all diagnoses mentioned
3. Apply the MEAT principle:
   - Monitor: What is being tracked?
   - Evaluate: What assessments were done?
   - Assess: What is the clinical interpretation?
   - Treat: What treatment is planned or provided?
4. Only assign codes for confirmed diagnoses, not suspected conditions
5. Do not code symptoms if the underlying cause is documented
6. Assign only billable (leaf-level) ICD-10-CM codes
7. For each code, provide both the code and its description

Clinical Note:
{note.text}

Output Format:
For each diagnosis, provide:
Code: [ICD-10-CM code]
Description: [Full description]

Your response:"""
    
    @staticmethod
    def chain_of_thought_prompt(note: ClinicalNote) -> str:
        """
        Chain-of-thought reasoning prompt.
        """
        return f"""You are an expert medical coder. Analyze the following clinical note step by step.

Clinical Note:
{note.text}

Please think through this step by step:
1. First, identify all medical conditions mentioned in the note
2. For each condition, determine if it is:
   - A confirmed diagnosis (should be coded)
   - A symptom that has an underlying cause (do not code the symptom)
   - A suspected/possible condition (do not code in outpatient setting)
3. Search your knowledge for the appropriate ICD-10-CM code
4. Verify the code is billable (leaf-level in the hierarchy)
5. Provide your final answer

Your reasoning and final codes:"""
    
    @staticmethod
    def detailed_cot_prompt(note: ClinicalNote) -> str:
        """
        Combination of detailed instructions with chain-of-thought.
        Best performing prompt for Sonnet-3.5v1 in the paper.
        """
        return f"""You are an expert medical coder. Analyze this clinical note using structured reasoning.

Clinical Note:
{note.text}

Instructions:
1. Apply the MEAT principle (Monitor, Evaluate, Assess, Treat)
2. Only code confirmed diagnoses
3. Do not code symptoms when the underlying etiology is documented
4. Use only billable ICD-10-CM codes

Step-by-step analysis:

Step 1: Extract Clinical Information
List all diagnoses, symptoms, and procedures mentioned.

Step 2: Apply Coding Rules
For each item, determine:
- Is this a confirmed diagnosis? (Required for coding)
- Is this a symptom with a documented cause? (Do not code symptom)
- Is this the most specific code available? (Must be billable/leaf-level)

Step 3: Assign Codes
For each diagnosis to be coded:

Code: [ICD-10-CM code]
Description: [Full description]
Justification: [Why this code was selected]

Your response:"""
    
    @staticmethod
    def decomposition_prompt(note: ClinicalNote) -> str:
        """
        Two-stage prompt: first extract key phrases, then assign codes.
        """
        return f"""You are an expert medical coder. Let's break this task into two steps.

Clinical Note:
{note.text}

STEP 1: Extract Key Clinical Phrases
List the key medical terms, diagnoses, and clinical findings from the note.

Key phrases:

STEP 2: Assign ICD-10-CM Codes
For each key phrase that represents a codeable diagnosis, assign the appropriate ICD-10-CM code.

Code: [ICD-10-CM code]
Description: [Full description]

Your response:"""
    
    @staticmethod
    def get_prompt(note: ClinicalNote, prompt_type: PromptType) -> str:
        """Get prompt based on type."""
        if prompt_type == PromptType.SINGLE_LINE:
            return PromptGenerator.single_line_prompt(note)
        elif prompt_type == PromptType.DETAILED:
            return PromptGenerator.detailed_prompt(note)
        elif prompt_type == PromptType.CHAIN_OF_THOUGHT:
            return PromptGenerator.chain_of_thought_prompt(note)
        elif prompt_type == PromptType.DETAILED_COT:
            return PromptGenerator.detailed_cot_prompt(note)
        elif prompt_type == PromptType.PROMPT_DECOMPOSITION:
            return PromptGenerator.decomposition_prompt(note)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


class CodeParser:
    """Parses model outputs to extract ICD-10-CM codes."""
    
    @staticmethod
    def parse_output(text: str) -> List[CodePrediction]:
        """
        Parse model output to extract codes and descriptions.
        Handles various output formats.
        """
        predictions = []
        lines = text.strip().split('\n')
        
        current_code = None
        current_desc = None
        
        for line in lines:
            line = line.strip()
            
            # Look for code patterns
            if line.startswith('Code:') or line.startswith('code:'):
                if current_code and current_desc:
                    predictions.append(CodePrediction(current_code, current_desc))
                current_code = line.split(':', 1)[1].strip()
                current_desc = None
            
            elif line.startswith('Description:') or line.startswith('description:'):
                current_desc = line.split(':', 1)[1].strip()
                if current_code and current_desc:
                    predictions.append(CodePrediction(current_code, current_desc))
                    current_code = None
                    current_desc = None
            
            # Handle comma-separated codes (single-line format)
            elif ',' in line and not any(x in line.lower() for x in ['step', 'note:', 'clinical']):
                codes = [c.strip() for c in line.split(',')]
                for code in codes:
                    if code and code[0].isalnum():  # Basic code validation
                        predictions.append(CodePrediction(code, ""))
        
        # Add any remaining code
        if current_code:
            predictions.append(CodePrediction(current_code, current_desc or ""))
        
        return predictions


# Example usage and testing
if __name__ == "__main__":
    # Create a sample clinical note
    note = ClinicalNote(
        text="""
        Patient presents with chronic left knee pain. 
        X-ray shows degenerative changes. 
        Difficulty walking due to pain.
        Assessment: Osteoarthritis of left knee.
        Plan: Physical therapy and NSAIDs.
        """,
        note_id="NOTE001"
    )
    
    print("Clinical Coding Prompt Examples")
    print("=" * 70)
    
    # Test different prompt types
    for prompt_type in PromptType:
        print(f"\n{'='*70}")
        print(f"Prompt Type: {prompt_type.value.upper()}")
        print(f"{'='*70}")
        prompt = PromptGenerator.get_prompt(note, prompt_type)
        print(prompt)
        print()
    
    # Test parsing
    print("\n" + "="*70)
    print("Testing Code Parser")
    print("="*70)
    
    sample_outputs = [
        """Code: M17.12
Description: Unilateral primary osteoarthritis, left knee
Code: R26.2
Description: Difficulty in walking""",
        
        """M17.12, M25.562, R26.2""",
        
        """Based on the note, I would assign:
        Code: M17.12
        Description: Unilateral primary osteoarthritis, left knee"""
    ]
    
    for i, output in enumerate(sample_outputs, 1):
        print(f"\nSample Output {i}:")
        print(output)
        print("\nParsed Codes:")
        predictions = CodeParser.parse_output(output)
        for pred in predictions:
            print(f"  - {pred.code}: {pred.description}")