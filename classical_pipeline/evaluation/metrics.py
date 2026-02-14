"""
Evaluation Metrics for Clinical Coding
Implements exact match, fuzzy match, and hierarchical metrics from the paper.
"""

from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.icd_hierarchy import ICDHierarchy


@dataclass
class MetricResult:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1: float
    support: int  # Number of examples


@dataclass
class FuzzyMatchResult:
    """Container for fuzzy match metrics."""
    prefix_1_f1: float
    prefix_2_f1: float
    prefix_overlap_ratio: float


class ClinicalCodingMetrics:
    """
    Evaluation metrics for clinical coding tasks.
    """
    
    def __init__(self, hierarchy: ICDHierarchy = None):
        self.hierarchy = hierarchy
    
    @staticmethod
    def exact_match_metrics(predicted: List[List[str]], 
                          gold: List[List[str]]) -> MetricResult:
        """
        Calculate exact match precision, recall, and F1.
        Works at per-note level.
        
        Args:
            predicted: List of predicted code lists (one per note)
            gold: List of gold code lists (one per note)
            
        Returns:
            MetricResult with aggregated metrics
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        n_examples = len(predicted)
        
        for pred_codes, gold_codes in zip(predicted, gold):
            pred_set = set(pred_codes)
            gold_set = set(gold_codes)
            
            if len(pred_set) == 0 and len(gold_set) == 0:
                # Both empty - perfect match
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            elif len(pred_set) == 0 or len(gold_set) == 0:
                # One empty - no match
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            else:
                # Calculate metrics
                true_positives = len(pred_set & gold_set)
                precision = true_positives / len(pred_set)
                recall = true_positives / len(gold_set)
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        # Macro-average across examples
        return MetricResult(
            precision=total_precision / n_examples if n_examples > 0 else 0.0,
            recall=total_recall / n_examples if n_examples > 0 else 0.0,
            f1=total_f1 / n_examples if n_examples > 0 else 0.0,
            support=n_examples
        )
    
    def prefix_match_metrics(self, 
                           predicted: List[List[str]], 
                           gold: List[List[str]],
                           n_levels: int = 1) -> MetricResult:
        """
        Calculate prefix-n match F1.
        Accepts predictions that match n levels above the leaf in ICD hierarchy.
        
        Args:
            predicted: List of predicted code lists
            gold: List of gold code lists
            n_levels: Number of levels above leaf to accept as match
            
        Returns:
            MetricResult with prefix-n metrics
        """
        if not self.hierarchy:
            raise ValueError("Hierarchy required for prefix matching")
        
        total_f1 = 0.0
        n_examples = len(predicted)
        
        for pred_codes, gold_codes in zip(predicted, gold):
            matches = 0
            
            for pred_code in pred_codes:
                for gold_code in gold_codes:
                    if self._is_prefix_match(pred_code, gold_code, n_levels):
                        matches += 1
                        break  # Each prediction can only match once
            
            # Calculate F1 for this example
            if len(pred_codes) == 0 and len(gold_codes) == 0:
                f1 = 1.0
            elif len(pred_codes) == 0 or len(gold_codes) == 0:
                f1 = 0.0
            else:
                precision = matches / len(pred_codes)
                recall = matches / len(gold_codes)
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
            
            total_f1 += f1
        
        avg_f1 = total_f1 / n_examples if n_examples > 0 else 0.0
        
        return MetricResult(
            precision=0.0,  # Not calculated for prefix match
            recall=0.0,
            f1=avg_f1,
            support=n_examples
        )
    
    def _is_prefix_match(self, pred_code: str, gold_code: str, n_levels: int) -> bool:
        """
        Check if predicted code matches gold code at n levels above leaf.
        """
        if pred_code == gold_code:
            return True
        
        # Get paths to root for both codes
        pred_path = self.hierarchy._get_path_to_root(pred_code)
        gold_path = self.hierarchy._get_path_to_root(gold_code)
        
        # Check if they match up to (depth - n_levels)
        if len(pred_path) < n_levels or len(gold_path) < n_levels:
            return False
        
        # Compare paths excluding last n_levels
        pred_prefix = pred_path[:-n_levels] if n_levels > 0 else pred_path
        gold_prefix = gold_path[:-n_levels] if n_levels > 0 else gold_path
        
        return pred_prefix == gold_prefix
    
    def prefix_overlap_ratio(self, 
                            predicted: List[List[str]], 
                            gold: List[List[str]]) -> float:
        """
        Calculate Prefix Overlap Ratio (POR).
        Measures weighted hierarchical overlap based on shared ancestry depth.
        
        Args:
            predicted: List of predicted code lists
            gold: List of gold code lists
            
        Returns:
            Average POR score
        """
        if not self.hierarchy:
            raise ValueError("Hierarchy required for POR calculation")
        
        total_por = 0.0
        n_examples = len(predicted)
        
        for pred_codes, gold_codes in zip(predicted, gold):
            if len(pred_codes) == 0 and len(gold_codes) == 0:
                total_por += 1.0
                continue
            elif len(pred_codes) == 0 or len(gold_codes) == 0:
                continue
            
            # Calculate overlap for this example
            max_overlap = 0.0
            for pred_code in pred_codes:
                for gold_code in gold_codes:
                    overlap = self._calculate_path_overlap(pred_code, gold_code)
                    max_overlap = max(max_overlap, overlap)
            
            total_por += max_overlap
        
        return total_por / n_examples if n_examples > 0 else 0.0
    
    def _calculate_path_overlap(self, code1: str, code2: str) -> float:
        """
        Calculate hierarchical overlap between two codes.
        Returns ratio of shared path length to maximum path length.
        """
        path1 = self.hierarchy._get_path_to_root(code1)
        path2 = self.hierarchy._get_path_to_root(code2)
        
        # Count shared prefix
        shared = 0
        for c1, c2 in zip(path1, path2):
            if c1 == c2:
                shared += 1
            else:
                break
        
        max_depth = max(len(path1), len(path2))
        
        return shared / max_depth if max_depth > 0 else 0.0
    
    def verification_accuracy(self, 
                            predicted: List[str], 
                            gold: List[str]) -> float:
        """
        Calculate verification accuracy.
        Used for standalone verification task where correct code is always in candidates.
        
        Args:
            predicted: List of selected codes
            gold: List of gold codes
            
        Returns:
            Accuracy (fraction correct)
        """
        correct = sum(1 for p, g in zip(predicted, gold) if p == g)
        return correct / len(predicted) if predicted else 0.0
    
    def comprehensive_evaluation(self,
                                predicted: List[List[str]],
                                gold: List[List[str]]) -> Dict[str, float]:
        """
        Run all evaluation metrics and return comprehensive results.
        
        Args:
            predicted: List of predicted code lists
            gold: List of gold code lists
            
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        # Exact match
        em = self.exact_match_metrics(predicted, gold)
        results['exact_match_precision'] = em.precision
        results['exact_match_recall'] = em.recall
        results['exact_match_f1'] = em.f1
        
        if self.hierarchy:
            # Prefix matches
            p1 = self.prefix_match_metrics(predicted, gold, n_levels=1)
            p2 = self.prefix_match_metrics(predicted, gold, n_levels=2)
            
            results['prefix_1_f1'] = p1.f1
            results['prefix_2_f1'] = p2.f1
            
            # Prefix overlap ratio
            results['prefix_overlap_ratio'] = self.prefix_overlap_ratio(predicted, gold)
        
        return results


def print_metrics_table(results: Dict[str, float], model_name: str = "Model"):
    """Pretty print metrics in table format."""
    print(f"\n{model_name} Performance:")
    print("=" * 60)
    print(f"{'Metric':<30} {'Score':>10}")
    print("-" * 60)
    
    for metric, score in sorted(results.items()):
        print(f"{metric:<30} {score:>10.2f}")
    
    print("=" * 60)


# Testing
if __name__ == "__main__":
    from utils.icd_hierarchy import create_sample_hierarchy
    
    print("Clinical Coding Evaluation Metrics")
    print("=" * 70)
    
    # Create hierarchy
    hierarchy = create_sample_hierarchy()
    metrics = ClinicalCodingMetrics(hierarchy)
    
    # Test cases
    print("\nTest Case 1: Perfect Match")
    print("-" * 70)
    predicted = [["M25.561", "I11.0"]]
    gold = [["M25.561", "I11.0"]]
    
    results = metrics.comprehensive_evaluation(predicted, gold)
    print_metrics_table(results, "Perfect Match")
    
    print("\nTest Case 2: Hierarchical Near-Miss (Right vs Left Knee)")
    print("-" * 70)
    predicted = [["M25.561"]]  # Right knee
    gold = [["M25.562"]]  # Left knee
    
    results = metrics.comprehensive_evaluation(predicted, gold)
    print_metrics_table(results, "Hierarchical Near-Miss")
    
    print("\nTest Case 3: Multiple Predictions")
    print("-" * 70)
    predicted = [
        ["M25.561", "I11.0"],
        ["M25.562", "R26.2"],
    ]
    gold = [
        ["M25.562", "I11.0"],  # First code wrong
        ["M25.562", "M51.27"],  # Second code wrong
    ]
    
    results = metrics.comprehensive_evaluation(predicted, gold)
    print_metrics_table(results, "Multiple Examples")
    
    print("\nTest Case 4: Verification Accuracy")
    print("-" * 70)
    predicted = ["M25.562", "I11.0", "R26.2"]
    gold = ["M25.562", "I11.9", "R26.2"]
    
    accuracy = metrics.verification_accuracy(predicted, gold)
    print(f"Verification Accuracy: {accuracy:.1%}")
    print(f"Correct: {sum(1 for p, g in zip(predicted, gold) if p == g)}/{len(predicted)}")