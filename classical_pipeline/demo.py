"""
Clinical Coding Pipeline Demo
End-to-end demonstration of the generate-expand-verify pipeline.
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


class ClinicalCodingDemo:
    """
    Comprehensive demo of the clinical coding pipeline.
    """
    
    def __init__(self, hierarchy: ICDHierarchy):
        self.hierarchy = hierarchy
        self.pipeline = ClinicalCodingPipeline(hierarchy)
        self.metrics = ClinicalCodingMetrics(hierarchy)
    
    def demo_prompt_engineering(self):
        """Demonstrate different prompt strategies."""
        print("\n" + "="*70)
        print("DEMO 1: PROMPT ENGINEERING STRATEGIES")
        print("="*70)
        
        note = ClinicalNote(
            text="""
            58-year-old female presents with chronic left knee pain for 6 months.
            Pain worsens with activity and stairs. No recent trauma.
            Physical exam: tenderness over medial joint line, mild effusion.
            X-ray shows joint space narrowing and osteophytes.
            Assessment: Primary osteoarthritis, left knee
            Plan: NSAIDs, physical therapy, weight management
            """,
            note_id="DEMO001"
        )
        
        print("\nClinical Note:")
        print(note.text)
        
        print("\n" + "-"*70)
        print("Generated Prompts:")
        print("-"*70)
        
        for prompt_type in [PromptType.SINGLE_LINE, PromptType.DETAILED, 
                           PromptType.DETAILED_COT]:
            print(f"\n{'='*70}")
            print(f"Prompt Type: {prompt_type.value.upper()}")
            print(f"{'='*70}")
            prompt = PromptGenerator.get_prompt(note, prompt_type)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    def demo_expansion(self):
        """Demonstrate candidate expansion."""
        print("\n" + "="*70)
        print("DEMO 2: CANDIDATE EXPANSION")
        print("="*70)
        
        # Use a wrong prediction to show expansion
        wrong_code = "M25.561"  # Pain in RIGHT knee
        correct_code = "M25.562"  # Pain in LEFT knee
        
        print(f"\nInitial (Incorrect) Prediction:")
        print(f"  {wrong_code}: {self.hierarchy.get_code_description(wrong_code)}")
        print(f"\nCorrect Code:")
        print(f"  {correct_code}: {self.hierarchy.get_code_description(correct_code)}")
        
        print("\n" + "-"*70)
        print("Expansion Results:")
        print("-"*70)
        
        from verification.verifier import CandidateExpander
        expander = CandidateExpander(self.hierarchy)
        
        for exp_type in ExpansionType:
            if exp_type == ExpansionType.ALL:
                continue
                
            expanded = expander.expand(wrong_code, exp_type)
            print(f"\n{exp_type.value.upper()}: {len(expanded)} candidates")
            
            for code in list(expanded)[:5]:  # Show first 5
                desc = self.hierarchy.get_code_description(code)
                marker = " ✓ CORRECT!" if code == correct_code else ""
                print(f"  - {code}: {desc}{marker}")
            
            if len(expanded) > 5:
                print(f"  ... and {len(expanded) - 5} more")
    
    def demo_verification(self):
        """Demonstrate verification process."""
        print("\n" + "="*70)
        print("DEMO 3: CODE VERIFICATION")
        print("="*70)
        
        note = ClinicalNote(
            text="""
            Patient complains of left knee pain.
            Exam shows tenderness and swelling.
            Diagnosis: Pain in left knee.
            """,
            note_id="DEMO003"
        )
        
        # Simulate wrong prediction
        wrong_prediction = "M25.561"  # Right knee
        candidates = ["M25.561", "M25.562"]  # Right and Left
        
        print("\nClinical Note:")
        print(note.text)
        print(f"\nInitial Prediction: {wrong_prediction} (Pain in RIGHT knee) ❌")
        print("\nCandidates after expansion:")
        for i, code in enumerate(candidates):
            desc = self.hierarchy.get_code_description(code)
            print(f"  {chr(65+i)}. {code}: {desc}")
        
        print("\n" + "-"*70)
        print("Verification Prompts:")
        print("-"*70)
        
        from verification.verifier import CodeVerifier
        verifier = CodeVerifier(self.hierarchy)
        
        for prompt_type in [VerificationPromptType.CODE_ONLY,
                           VerificationPromptType.DESCRIPTION_ONLY]:
            print(f"\n{prompt_type.value.upper()}:")
            print("-" * 40)
            prompt = verifier.generate_verification_prompt(
                note, candidates, prompt_type
            )
            print(prompt)
    
    def demo_full_pipeline(self):
        """Demonstrate complete pipeline with evaluation."""
        print("\n" + "="*70)
        print("DEMO 4: FULL PIPELINE WITH EVALUATION")
        print("="*70)
        
        # Create test cases
        test_cases = [
            {
                'note': ClinicalNote(
                    text="Patient has pain in left knee. Diagnosis: osteoarthritis, left knee.",
                    note_id="TEST001"
                ),
                'predicted': ["M25.561"],  # Wrong: right knee
                'gold': ["M25.562"],  # Correct: left knee
                'description': "Laterality error (right vs left)"
            },
            {
                'note': ClinicalNote(
                    text="Hypertensive heart disease with heart failure documented.",
                    note_id="TEST002"
                ),
                'predicted': ["I11.9"],  # Wrong: without heart failure
                'gold': ["I11.0"],  # Correct: with heart failure
                'description': "Complication status (with vs without)"
            }
        ]
        
        print("\n" + "-"*70)
        print("Test Cases:")
        print("-"*70)
        
        all_predicted_before = []
        all_predicted_after = []
        all_gold = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['description']}")
            print(f"Note: {test['note'].text[:70]}...")
            print(f"Initial Prediction: {test['predicted'][0]} ❌")
            print(f"Gold Standard: {test['gold'][0]} ✓")
            
            # Run pipeline
            verified = self.pipeline.run_pipeline(
                note=test['note'],
                initial_predictions=test['predicted'],
                expansion_type=ExpansionType.ALL,
                verification_prompt=VerificationPromptType.DESCRIPTION_ONLY,
                model_fn=None  # Mock for demo
            )
            
            print(f"After Verification: {verified[0]}")
            
            # Collect for evaluation
            all_predicted_before.append(test['predicted'])
            all_predicted_after.append(verified)
            all_gold.append(test['gold'])
        
        # Evaluate
        print("\n" + "-"*70)
        print("Evaluation Results:")
        print("-"*70)
        
        print("\nBEFORE VERIFICATION:")
        results_before = self.metrics.comprehensive_evaluation(
            all_predicted_before, all_gold
        )
        print_metrics_table(results_before, "Before Verification")
        
        print("\nAFTER VERIFICATION:")
        results_after = self.metrics.comprehensive_evaluation(
            all_predicted_after, all_gold
        )
        print_metrics_table(results_after, "After Verification")
        
        print("\n" + "-"*70)
        print("Improvement:")
        print("-"*70)
        print(f"F1 Score: {results_before['exact_match_f1']:.3f} → "
              f"{results_after['exact_match_f1']:.3f} "
              f"(+{results_after['exact_match_f1'] - results_before['exact_match_f1']:.3f})")
    
    def demo_metrics(self):
        """Demonstrate evaluation metrics."""
        print("\n" + "="*70)
        print("DEMO 5: EVALUATION METRICS")
        print("="*70)
        
        test_scenarios = [
            {
                'name': "Perfect Match",
                'predicted': [["M25.562", "I11.0"]],
                'gold': [["M25.562", "I11.0"]]
            },
            {
                'name': "Complete Miss",
                'predicted': [["R26.2"]],
                'gold': [["M25.562"]]
            },
            {
                'name': "Hierarchical Near-Miss (siblings)",
                'predicted': [["M25.561"]],  # Right knee
                'gold': [["M25.562"]]  # Left knee
            },
            {
                'name': "Hierarchical Near-Miss (parent-child)",
                'predicted': [["I11"]],  # Parent
                'gold': [["I11.0"]]  # Child
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n{'-'*70}")
            print(f"Scenario: {scenario['name']}")
            print(f"{'-'*70}")
            
            results = self.metrics.comprehensive_evaluation(
                scenario['predicted'], scenario['gold']
            )
            
            print(f"Predicted: {scenario['predicted'][0]}")
            print(f"Gold: {scenario['gold'][0]}")
            print(f"\nExact Match F1: {results['exact_match_f1']:.3f}")
            if 'prefix_1_f1' in results:
                print(f"Prefix-1 Match F1: {results['prefix_1_f1']:.3f}")
                print(f"Prefix Overlap Ratio: {results['prefix_overlap_ratio']:.3f}")
    
    def run_all_demos(self):
        """Run all demonstrations."""
        print("\n" + "="*70)
        print(" CLINICAL CODING PIPELINE - COMPREHENSIVE DEMO")
        print("="*70)
        print("\nThis demo implements the paper:")
        print("'Leveraging LLMs for Clinical Coding with Structured Verification'")
        print("\nKey Components:")
        print("1. Prompt Engineering Strategies")
        print("2. Hierarchical Candidate Expansion")
        print("3. LLM-based Code Verification")
        print("4. Comprehensive Evaluation Metrics")
        
        self.demo_prompt_engineering()
        self.demo_expansion()
        self.demo_verification()
        self.demo_full_pipeline()
        self.demo_metrics()
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nNext Steps:")
        print("1. Load real ICD-10-CM hierarchy data")
        print("2. Integrate with actual LLM (Claude, GPT, etc.)")
        print("3. Train on annotated clinical notes")
        print("4. Evaluate on test set")
        print("\nFiles created:")
        print("- utils/icd_hierarchy.py: ICD-10-CM structure")
        print("- models/code_generator.py: Prompt strategies")
        print("- verification/verifier.py: Expand-verify pipeline")
        print("- evaluation/metrics.py: Evaluation metrics")


def main():
    """Main entry point."""
    # Create sample hierarchy
    hierarchy = create_sample_hierarchy()
    
    # Run demo
    demo = ClinicalCodingDemo(hierarchy)
    demo.run_all_demos()


if __name__ == "__main__":
    main()