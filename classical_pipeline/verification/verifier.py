"""
Code Verification Module
Implements the generate-expand-verify pipeline from the paper.
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.icd_hierarchy import ICDHierarchy
from models.code_generator import ClinicalNote, CodePrediction


class ExpansionType(Enum):
    """Types of candidate expansion."""
    SIBLINGS = "siblings"
    COUSINS = "cousins"
    ONE_HOP = "1hop"
    TWO_HOP = "2hop"
    ALL = "all"


class VerificationPromptType(Enum):
    """Types of verification prompts."""
    CODE_ONLY = "code_only"
    CODE_DESCRIPTION = "code_description"
    DESCRIPTION_ONLY = "description_only"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class VerificationTask:
    """Represents a verification task."""
    clinical_note: ClinicalNote
    candidate_codes: List[str]
    gold_codes: Optional[List[str]] = None


@dataclass
class VerificationResult:
    """Result of verification."""
    selected_code: str
    confidence: float
    all_candidates: List[str]
    reasoning: Optional[str] = None


class CandidateExpander:
    """
    Expands candidate codes using ICD-10-CM hierarchy.
    Implements the expansion step: Expand(c) = S(c) ∪ C(c) ∪ N1(c) ∪ N2(c)
    """
    
    def __init__(self, hierarchy: ICDHierarchy):
        self.hierarchy = hierarchy
    
    def expand(self, code: str, expansion_type: ExpansionType) -> Set[str]:
        """
        Expand a single code based on expansion type.
        
        Args:
            code: ICD-10-CM code to expand
            expansion_type: Type of expansion to perform
            
        Returns:
            Set of expanded candidate codes
        """
        if expansion_type == ExpansionType.SIBLINGS:
            return self.hierarchy.get_siblings(code)
        
        elif expansion_type == ExpansionType.COUSINS:
            return self.hierarchy.get_cousins(code)
        
        elif expansion_type == ExpansionType.ONE_HOP:
            return self.hierarchy.get_1hop_neighbors(code)
        
        elif expansion_type == ExpansionType.TWO_HOP:
            return self.hierarchy.get_2hop_neighbors(code)
        
        elif expansion_type == ExpansionType.ALL:
            return self.hierarchy.expand_candidates(code)
        
        else:
            raise ValueError(f"Unknown expansion type: {expansion_type}")
    
    def expand_multiple(self, codes: List[str], 
                       expansion_type: ExpansionType) -> Set[str]:
        """
        Expand multiple codes and combine results.
        
        Args:
            codes: List of ICD-10-CM codes
            expansion_type: Type of expansion
            
        Returns:
            Combined set of all expanded candidates
        """
        all_candidates = set(codes)  # Include original codes
        
        for code in codes:
            expanded = self.expand(code, expansion_type)
            all_candidates.update(expanded)
        
        return all_candidates
    
    def get_expansion_statistics(self, code: str) -> Dict[str, int]:
        """Get statistics about expansion for a code."""
        return {
            'siblings': len(self.hierarchy.get_siblings(code)),
            'cousins': len(self.hierarchy.get_cousins(code)),
            '1hop': len(self.hierarchy.get_1hop_neighbors(code)),
            '2hop': len(self.hierarchy.get_2hop_neighbors(code)),
            'total': len(self.hierarchy.expand_candidates(code))
        }


class CodeVerifier:
    """
    Verifies codes using LLM-based contextual revision.
    Implements the verification step: c_best = argmax_{c in C(c)} f_verify(N, c)
    """
    
    def __init__(self, hierarchy: ICDHierarchy):
        self.hierarchy = hierarchy
    
    def generate_verification_prompt(self,
                                    note: ClinicalNote,
                                    candidates: List[str],
                                    prompt_type: VerificationPromptType) -> str:
        """
        Generate verification prompt based on type.
        
        Args:
            note: Clinical note
            candidates: List of candidate codes to verify
            prompt_type: Type of verification prompt
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""Given the following clinical note, select the most appropriate ICD-10-CM code from the candidates.

Clinical Note:
{note.text}

"""
        
        if prompt_type == VerificationPromptType.CODE_ONLY:
            # Present only codes
            options = "\n".join([f"{chr(65+i)}. {code}" 
                                for i, code in enumerate(candidates)])
            
            return base_prompt + f"""Candidates:
{options}

Select the letter (A, B, C, etc.) of the most appropriate code based on the clinical note.

Your answer (letter only):"""
        
        elif prompt_type == VerificationPromptType.CODE_DESCRIPTION:
            # Present codes with descriptions
            options = "\n".join([
                f"{chr(65+i)}. {code} - {self.hierarchy.get_code_description(code)}"
                for i, code in enumerate(candidates)
            ])
            
            return base_prompt + f"""Candidates:
{options}

Select the letter (A, B, C, etc.) of the most appropriate code based on the clinical note.

Your answer (letter only):"""
        
        elif prompt_type == VerificationPromptType.DESCRIPTION_ONLY:
            # Present only descriptions (best performing in paper)
            options = "\n".join([
                f"{chr(65+i)}. {self.hierarchy.get_code_description(code)}"
                for i, code in enumerate(candidates)
            ])
            
            return base_prompt + f"""Candidates:
{options}

Select the letter (A, B, C, etc.) of the most appropriate diagnosis based on the clinical note.

Your answer (letter only):"""
        
        elif prompt_type == VerificationPromptType.CHAIN_OF_THOUGHT:
            # Chain-of-thought reasoning
            options = "\n".join([
                f"{chr(65+i)}. {code} - {self.hierarchy.get_code_description(code)}"
                for i, code in enumerate(candidates)
            ])
            
            return base_prompt + f"""Candidates:
{options}

Think step by step:
1. What are the key clinical findings in the note?
2. Which candidate best matches these findings?
3. Are there any subtle distinctions (e.g., laterality, with/without complications)?

Your reasoning and final answer (letter):"""
        
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    def parse_verification_response(self, response: str, 
                                   candidates: List[str]) -> Optional[str]:
        """
        Parse model response to extract selected code.
        
        Args:
            response: Model's text response
            candidates: List of candidate codes
            
        Returns:
            Selected code or None if parsing fails
        """
        response = response.strip().upper()
        
        # Look for letter choices (A, B, C, etc.)
        for i, candidate in enumerate(candidates):
            letter = chr(65 + i)
            if letter in response[:10]:  # Check first 10 chars
                return candidate
        
        # Fallback: look for code directly in response
        for candidate in candidates:
            if candidate in response:
                return candidate
        
        return None
    
    def verify_codes(self,
                    note: ClinicalNote,
                    predicted_codes: List[str],
                    expansion_type: ExpansionType,
                    prompt_type: VerificationPromptType,
                    model_generate_fn=None) -> List[VerificationResult]:
        """
        Full verification pipeline for multiple predicted codes.
        
        Args:
            note: Clinical note
            predicted_codes: Initial predicted codes
            expansion_type: How to expand candidates
            prompt_type: Type of verification prompt
            model_generate_fn: Function that takes prompt and returns model output
                              If None, returns mock results for testing
            
        Returns:
            List of verification results
        """
        expander = CandidateExpander(self.hierarchy)
        results = []
        
        for pred_code in predicted_codes:
            # Step 1: Expand candidates
            if expansion_type == ExpansionType.ALL:
                candidates = list(expander.expand(pred_code, ExpansionType.ALL))
            else:
                candidates = list(expander.expand(pred_code, expansion_type))
            
            # Add original prediction
            if pred_code not in candidates:
                candidates.insert(0, pred_code)
            
            # Limit candidates to reasonable number (e.g., 10)
            candidates = candidates[:10]
            
            # Step 2: Generate verification prompt
            prompt = self.generate_verification_prompt(note, candidates, prompt_type)
            
            # Step 3: Get model response
            if model_generate_fn:
                response = model_generate_fn(prompt)
                selected = self.parse_verification_response(response, candidates)
            else:
                # Mock response for testing
                selected = candidates[0]
                response = f"Selected: {selected}"
            
            # Step 4: Create result
            result = VerificationResult(
                selected_code=selected or pred_code,
                confidence=1.0,
                all_candidates=candidates,
                reasoning=response if model_generate_fn else None
            )
            results.append(result)
        
        return results


class ClinicalCodingPipeline:
    """
    Complete end-to-end pipeline: Generate -> Expand -> Verify
    """
    
    def __init__(self, hierarchy: ICDHierarchy):
        self.hierarchy = hierarchy
        self.verifier = CodeVerifier(hierarchy)
        self.expander = CandidateExpander(hierarchy)
    
    def run_pipeline(self,
                    note: ClinicalNote,
                    initial_predictions: List[str],
                    expansion_type: ExpansionType = ExpansionType.ALL,
                    verification_prompt: VerificationPromptType = VerificationPromptType.DESCRIPTION_ONLY,
                    model_fn=None) -> List[str]:
        """
        Run complete pipeline on a clinical note.
        
        Args:
            note: Clinical note to process
            initial_predictions: Initial code predictions from generation step
            expansion_type: Type of candidate expansion
            verification_prompt: Type of verification prompt
            model_fn: Optional model function for verification
            
        Returns:
            List of verified ICD-10-CM codes
        """
        # Expand and verify each prediction
        results = self.verifier.verify_codes(
            note=note,
            predicted_codes=initial_predictions,
            expansion_type=expansion_type,
            prompt_type=verification_prompt,
            model_generate_fn=model_fn
        )
        
        # Extract verified codes
        verified_codes = [r.selected_code for r in results]
        
        return verified_codes
    
    def analyze_expansion(self, codes: List[str]) -> Dict:
        """Analyze expansion statistics for codes."""
        stats = {
            'total_codes': len(codes),
            'per_code_expansion': {},
            'total_candidates': 0
        }
        
        all_candidates = set()
        for code in codes:
            code_stats = self.expander.get_expansion_statistics(code)
            stats['per_code_expansion'][code] = code_stats
            all_candidates.update(self.expander.expand(code, ExpansionType.ALL))
        
        stats['total_candidates'] = len(all_candidates)
        stats['expansion_ratio'] = len(all_candidates) / len(codes) if codes else 0
        
        return stats
