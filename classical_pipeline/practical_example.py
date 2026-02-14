"""
Practical Example: End-to-End Clinical Coding with Mock LLM
This demonstrates how to integrate the pipeline with an actual LLM API.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.icd_hierarchy import ICDHierarchy, create_sample_hierarchy
from models.code_generator import (
    ClinicalNote, PromptGenerator, PromptType, CodeParser
)
from verification.verifier import (
    ClinicalCodingPipeline, ExpansionType, VerificationPromptType
)
from evaluation.metrics import ClinicalCodingMetrics, print_metrics_table


class MockLLM:
    """
    Mock LLM for demonstration purposes.
    In practice, replace this with actual API calls to Claude, GPT, etc.
    """
    
    def __init__(self, hierarchy: ICDHierarchy):
        self.hierarchy = hierarchy
        
        # Mock responses for demo
        self.generation_responses = {
            "left knee pain": ["M25.562"],  # Correct
            "right knee pain": ["M25.561"],  # Correct
            "knee pain": ["M25.561"],  # Incorrect (should ask which knee)
            "hypertensive heart disease with heart failure": ["I11.0"],
            "hypertensive heart disease": ["I11.9"],
        }
        
        # Mock verification responses
        self.verification_responses = {
            ("M25.561", "M25.562"): "B",  # Choose left knee
            ("M25.562", "M25.561"): "A",  # Choose left knee
            ("I11.0", "I11.9"): "A",  # Choose with heart failure
            ("I11.9", "I11.0"): "B",  # Choose with heart failure
        }
    
    def generate(self, prompt: str) -> str:
        """
        Mock code generation.
        In practice: Replace with actual LLM API call.
        """
        # Simple keyword matching for demo
        prompt_lower = prompt.lower()
        
        if "left knee" in prompt_lower:
            return "Code: M25.562\nDescription: Pain in left knee"
        elif "right knee" in prompt_lower:
            return "Code: M25.561\nDescription: Pain in right knee"
        elif "hypertensive" in prompt_lower and "with heart failure" in prompt_lower:
            return "Code: I11.0\nDescription: Hypertensive heart disease with heart failure"
        elif "hypertensive" in prompt_lower:
            return "Code: I11.9\nDescription: Hypertensive heart disease without heart failure"
        else:
            return "Code: R69\nDescription: Illness, unspecified"
    
    def verify(self, prompt: str, candidates: list) -> str:
        """
        Mock verification.
        In practice: Replace with actual LLM API call.
        """
        prompt_lower = prompt.lower()
        
        # Simple rule-based verification for demo
        if "left" in prompt_lower and any("left" in self.hierarchy.get_code_description(c).lower() 
                                          for c in candidates):
            # Find and return the letter for left knee code
            for i, code in enumerate(candidates):
                if "left" in self.hierarchy.get_code_description(code).lower():
                    return chr(65 + i)
        
        if "right" in prompt_lower and any("right" in self.hierarchy.get_code_description(c).lower() 
                                           for c in candidates):
            for i, code in enumerate(candidates):
                if "right" in self.hierarchy.get_code_description(code).lower():
                    return chr(65 + i)
        
        if "with heart failure" in prompt_lower or "with hf" in prompt_lower:
            for i, code in enumerate(candidates):
                if "with heart failure" in self.hierarchy.get_code_description(code).lower():
                    return chr(65 + i)
        
        # Default to first candidate
        return "A"


class RealLLMExample:
    """
    Template for integrating with real LLM APIs.
    """
    
    @staticmethod
    def claude_api_example():
        """
        Example integration with Anthropic Claude API.
        Uncomment and add your API key to use.
        """
        code = '''
# Uncomment to use with actual API key
"""
import anthropic

def generate_with_claude(prompt: str) -> str:
    client = anthropic.Anthropic(
        api_key="your-api-key-here"
    )
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text

# Use in pipeline
model_fn = generate_with_claude
"""
        '''
        print(code)
    
    @staticmethod
    def openai_api_example():
        """
        Example integration with OpenAI API.
        """
        code = '''
# Uncomment to use with actual API key
"""
import openai

def generate_with_gpt(prompt: str) -> str:
    client = openai.OpenAI(
        api_key="your-api-key-here"
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024
    )
    
    return response.choices[0].message.content

# Use in pipeline
model_fn = generate_with_gpt
"""
        '''
        print(code)


def run_practical_example():
    """
    Run a complete practical example with mock LLM.
    """
    print("="*70)
    print("PRACTICAL EXAMPLE: CLINICAL CODING PIPELINE")
    print("="*70)
    
    # Setup
    hierarchy = create_sample_hierarchy()
    mock_llm = MockLLM(hierarchy)
    pipeline = ClinicalCodingPipeline(hierarchy)
    metrics = ClinicalCodingMetrics(hierarchy)
    
    # Test cases
    test_cases = [
        {
            'note': ClinicalNote(
                text="""
                Patient: 58F with chronic left knee pain x6 months.
                Worse with stairs and prolonged standing.
                PE: Tenderness over medial joint line, small effusion.
                XR: Joint space narrowing, osteophytes.
                Dx: Primary osteoarthritis, left knee
                Plan: NSAIDs, PT, weight loss
                """,
                note_id="CASE001"
            ),
            'gold': ["M17.12"],  # Osteoarthritis left knee (not in our sample)
            'alternative_gold': ["M25.562"],  # Pain in left knee
            'description': "Left knee osteoarthritis"
        },
        {
            'note': ClinicalNote(
                text="""
                Patient: 65M with HTN and recent SOB.
                Echo shows reduced EF 35%, LVH.
                Diagnosis: Hypertensive heart disease with heart failure
                Plan: ACE inhibitor, diuretic, cardiology follow-up
                """,
                note_id="CASE002"
            ),
            'gold': ["I11.0"],
            'description': "Hypertensive heart disease with HF"
        },
        {
            'note': ClinicalNote(
                text="""
                Patient presents with right knee pain after fall.
                Swelling and limited ROM noted.
                X-ray negative for fracture.
                Diagnosis: Contusion and pain in right knee
                """,
                note_id="CASE003"
            ),
            'gold': ["M25.561"],
            'description': "Right knee pain post-trauma"
        }
    ]
    
    print("\n" + "="*70)
    print("STEP 1: CODE GENERATION")
    print("="*70)
    
    all_predictions = []
    all_gold = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Test Case {i}: {test['description']}")
        print(f"{'-'*70}")
        print(f"Note (excerpt): {test['note'].text[:100]}...")
        
        # Generate prompt
        prompt = PromptGenerator.get_prompt(test['note'], PromptType.DETAILED)
        
        # Get LLM response (mock)
        response = mock_llm.generate(prompt)
        print(f"\nLLM Response:\n{response}")
        
        # Parse codes
        predictions = CodeParser.parse_output(response)
        pred_codes = [p.code for p in predictions]
        
        print(f"\nExtracted Codes: {pred_codes}")
        print(f"Gold Standard: {test['gold']}")
        
        all_predictions.append(pred_codes)
        all_gold.append(test['gold'])
    
    print("\n" + "="*70)
    print("STEP 2: EVALUATION (Before Verification)")
    print("="*70)
    
    results_before = metrics.comprehensive_evaluation(all_predictions, all_gold)
    print_metrics_table(results_before, "Before Verification")
    
    print("\n" + "="*70)
    print("STEP 3: VERIFICATION PIPELINE")
    print("="*70)
    
    all_verified = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Verifying Case {i}: {test['description']}")
        print(f"{'-'*70}")
        
        initial_preds = all_predictions[i-1]
        print(f"Initial Predictions: {initial_preds}")
        
        # Create verification function that uses mock LLM
        def verify_fn(prompt: str) -> str:
            return mock_llm.verify(prompt, initial_preds)
        
        # Run verification
        verified = pipeline.run_pipeline(
            note=test['note'],
            initial_predictions=initial_preds,
            expansion_type=ExpansionType.ALL,
            verification_prompt=VerificationPromptType.DESCRIPTION_ONLY,
            model_fn=verify_fn
        )
        
        print(f"After Verification: {verified}")
        print(f"Gold Standard: {test['gold']}")
        
        # Check if improved
        if verified[0] == test['gold'][0]:
            print("✓ Verification corrected the prediction!")
        elif initial_preds[0] == test['gold'][0]:
            print("✓ Initial prediction was already correct")
        else:
            print("✗ Still incorrect after verification")
        
        all_verified.append(verified)
    
    print("\n" + "="*70)
    print("STEP 4: FINAL EVALUATION")
    print("="*70)
    
    results_after = metrics.comprehensive_evaluation(all_verified, all_gold)
    print_metrics_table(results_after, "After Verification")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nExact Match F1:")
    print(f"  Before: {results_before['exact_match_f1']:.3f}")
    print(f"  After:  {results_after['exact_match_f1']:.3f}")
    print(f"  Change: {results_after['exact_match_f1'] - results_before['exact_match_f1']:+.3f}")
    
    if 'prefix_1_f1' in results_before:
        print(f"\nPrefix-1 Match F1:")
        print(f"  Before: {results_before['prefix_1_f1']:.3f}")
        print(f"  After:  {results_after['prefix_1_f1']:.3f}")
        print(f"  Change: {results_after['prefix_1_f1'] - results_before['prefix_1_f1']:+.3f}")
    
    print("\n" + "="*70)
    print("INTEGRATION WITH REAL LLM APIs")
    print("="*70)
    
    print("\n1. Anthropic Claude API:")
    print("-" * 70)
    RealLLMExample.claude_api_example()
    
    print("\n2. OpenAI GPT API:")
    print("-" * 70)
    RealLLMExample.openai_api_example()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("1. The pipeline can be easily integrated with any LLM API")
    print("2. Mock responses allow for testing without API costs")
    print("3. Verification improves accuracy on hierarchical near-misses")
    print("4. The system is modular and extensible")
    
    print("\nNext Steps:")
    print("1. Replace MockLLM with actual API calls")
    print("2. Load full ICD-10-CM hierarchy")
    print("3. Test on real clinical notes")
    print("4. Fine-tune models on your dataset")


if __name__ == "__main__":
    run_practical_example()