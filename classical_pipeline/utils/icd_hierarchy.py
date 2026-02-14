"""
ICD-10-CM Hierarchy Management
Implements the tree structure and graph relationships for ICD-10-CM codes.
"""

from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict
import json


class ICDHierarchy:
    """
    Manages the hierarchical structure of ICD-10-CM codes.
    
    The tabular list is represented as a tree where:
    - V: set of all ICD-10-CM codes
    - E: parent-child relationships
    - Only leaf nodes are billable codes
    """
    
    def __init__(self):
        self.codes: Set[str] = set()
        self.parent_map: Dict[str, str] = {}  # child -> parent
        self.children_map: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.descriptions: Dict[str, str] = {}
        self.billable_codes: Set[str] = set()
        
        # Index list (cross-references)
        self.index_neighbors: Dict[str, Set[str]] = defaultdict(set)
        
    def add_code(self, code: str, description: str, parent: Optional[str] = None, 
                 is_billable: bool = False):
        """Add a code to the hierarchy."""
        self.codes.add(code)
        self.descriptions[code] = description
        
        if is_billable:
            self.billable_codes.add(code)
        
        if parent:
            self.parent_map[code] = parent
            self.children_map[parent].add(code)
    
    def add_index_reference(self, code1: str, code2: str):
        """Add a cross-reference between codes in the index list."""
        self.index_neighbors[code1].add(code2)
        self.index_neighbors[code2].add(code1)
    
    def get_parent(self, code: str) -> Optional[str]:
        """Get the parent of a code: P(c)"""
        return self.parent_map.get(code)
    
    def get_siblings(self, code: str) -> Set[str]:
        """
        Get siblings of a code: S(c) = {s ∈ V : P(s) = P(c) and s ≠ c}
        Codes that share the same parent.
        """
        parent = self.get_parent(code)
        if not parent:
            return set()
        
        siblings = self.children_map[parent].copy()
        siblings.discard(code)
        return siblings
    
    def get_cousins(self, code: str) -> Set[str]:
        """
        Get cousins of a code: C(c) = {g ∈ V : P(P(g)) = P(P(c)) and g ∉ S(c)}
        Codes that share the same grandparent but are not siblings.
        """
        parent = self.get_parent(code)
        if not parent:
            return set()
        
        grandparent = self.get_parent(parent)
        if not grandparent:
            return set()
        
        cousins = set()
        siblings = self.get_siblings(code)
        
        # Get all grandchildren of the grandparent
        for uncle_aunt in self.children_map[grandparent]:
            for cousin in self.children_map[uncle_aunt]:
                if cousin != code and cousin not in siblings:
                    cousins.add(cousin)
        
        return cousins
    
    def get_1hop_neighbors(self, code: str) -> Set[str]:
        """
        Get 1-hop neighbors from index list: N₁(c) = {n ∈ V : (c, n) ∈ E'}
        """
        return self.index_neighbors.get(code, set()).copy()
    
    def get_2hop_neighbors(self, code: str) -> Set[str]:
        """
        Get 2-hop neighbors from index list: 
        N₂(c) = {n ∈ V : ∃v ∈ V, (c,v) ∈ E' and (v,n) ∈ E'}
        """
        two_hop = set()
        one_hop = self.get_1hop_neighbors(code)
        
        for neighbor in one_hop:
            two_hop.update(self.index_neighbors.get(neighbor, set()))
        
        # Remove the original code and 1-hop neighbors
        two_hop.discard(code)
        two_hop -= one_hop
        
        return two_hop
    
    def expand_candidates(self, code: str) -> Set[str]:
        """
        Full expansion: Expand(c) = S(c) ∪ C(c) ∪ N₁(c) ∪ N₂(c)
        """
        expansion = set()
        expansion.update(self.get_siblings(code))
        expansion.update(self.get_cousins(code))
        expansion.update(self.get_1hop_neighbors(code))
        expansion.update(self.get_2hop_neighbors(code))
        expansion.discard(code)  # Remove the original code
        return expansion
    
    def get_code_description(self, code: str) -> str:
        """Get description for a code."""
        return self.descriptions.get(code, "")
    
    def is_billable(self, code: str) -> bool:
        """Check if a code is billable (leaf node)."""
        return code in self.billable_codes
    
    def get_prefix_match(self, code1: str, code2: str) -> int:
        """
        Calculate hierarchical similarity by counting common prefix steps.
        Returns the number of hierarchy levels that match.
        """
        # Get hierarchical paths for both codes
        path1 = self._get_path_to_root(code1)
        path2 = self._get_path_to_root(code2)
        
        # Count common prefix
        match_count = 0
        for c1, c2 in zip(path1, path2):
            if c1 == c2:
                match_count += 1
            else:
                break
        
        return match_count
    
    def _get_path_to_root(self, code: str) -> List[str]:
        """Get the path from code to root."""
        path = []
        current = code
        
        while current:
            path.append(current)
            current = self.get_parent(current)
        
        path.reverse()
        return path
    
    def save_to_file(self, filepath: str):
        """Save hierarchy to JSON file."""
        data = {
            'codes': list(self.codes),
            'parent_map': self.parent_map,
            'children_map': {k: list(v) for k, v in self.children_map.items()},
            'descriptions': self.descriptions,
            'billable_codes': list(self.billable_codes),
            'index_neighbors': {k: list(v) for k, v in self.index_neighbors.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ICDHierarchy':
        """Load hierarchy from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        hierarchy = cls()
        hierarchy.codes = set(data['codes'])
        hierarchy.parent_map = data['parent_map']
        hierarchy.children_map = defaultdict(set, 
            {k: set(v) for k, v in data['children_map'].items()})
        hierarchy.descriptions = data['descriptions']
        hierarchy.billable_codes = set(data['billable_codes'])
        hierarchy.index_neighbors = defaultdict(set,
            {k: set(v) for k, v in data['index_neighbors'].items()})
        
        return hierarchy


def create_sample_hierarchy() -> ICDHierarchy:
    """
    Create a sample ICD-10-CM hierarchy for testing.
    Based on examples from the paper (Hypertensive diseases, Musculoskeletal, etc.)
    """
    hierarchy = ICDHierarchy()
    
    # Chapter I: Certain infectious and parasitic diseases (A00-B99)
    hierarchy.add_code("I", "Diseases of the circulatory system")
    
    # Hypertensive diseases (I10-I16)
    hierarchy.add_code("I10-I16", "Hypertensive diseases", parent="I")
    hierarchy.add_code("I11", "Hypertensive heart disease", parent="I10-I16")
    hierarchy.add_code("I11.0", "Hypertensive heart disease with heart failure", 
                      parent="I11", is_billable=True)
    hierarchy.add_code("I11.9", "Hypertensive heart disease without heart failure", 
                      parent="I11", is_billable=True)
    
    # Chapter M: Musculoskeletal system
    hierarchy.add_code("M", "Diseases of the musculoskeletal system and connective tissue")
    
    # Arthropathies (M00-M25)
    hierarchy.add_code("M00-M25", "Arthropathies", parent="M")
    hierarchy.add_code("M25", "Other joint disorder, not elsewhere classified", parent="M00-M25")
    hierarchy.add_code("M25.5", "Pain in joint", parent="M25")
    hierarchy.add_code("M25.56", "Pain in knee", parent="M25.5")
    hierarchy.add_code("M25.561", "Pain in right knee", parent="M25.56", is_billable=True)
    hierarchy.add_code("M25.562", "Pain in left knee", parent="M25.56", is_billable=True)
    
    # Dorsopathies (M50-M54)
    hierarchy.add_code("M50-M54", "Dorsopathies", parent="M")
    hierarchy.add_code("M51", "Thoracic, thoracolumbar, and lumbosacral intervertebral disc disorders", 
                      parent="M50-M54")
    hierarchy.add_code("M51.2", "Other thoracic, thoracolumbar and lumbosacral intervertebral disc displacement",
                      parent="M51")
    hierarchy.add_code("M51.27", "Other intervertebral disc displacement, lumbosacral region",
                      parent="M51.2", is_billable=True)
    
    # Chapter R: Symptoms, signs and abnormal findings
    hierarchy.add_code("R", "Symptoms, signs and abnormal clinical and laboratory findings")
    
    # Symptoms and signs involving the nervous and musculoskeletal systems (R25-R29)
    hierarchy.add_code("R25-R29", "Symptoms and signs involving the nervous and musculoskeletal systems",
                      parent="R")
    hierarchy.add_code("R26", "Abnormalities of gait and mobility", parent="R25-R29")
    hierarchy.add_code("R26.2", "Difficulty in walking, not elsewhere classified",
                      parent="R26", is_billable=True)
    
    # Abnormal findings on examination of blood (R70-R79)
    hierarchy.add_code("R70-R79", "Abnormal findings on examination of blood", parent="R")
    hierarchy.add_code("R78", "Findings of drugs and other substances, not normally found in blood",
                      parent="R70-R79")
    hierarchy.add_code("R78.7", "Abnormal lead level in blood", parent="R78")
    hierarchy.add_code("R78.71", "Abnormal lead level in blood", parent="R78.7", is_billable=True)
    hierarchy.add_code("R78.9", "Finding of unspecified substance, not normally found in blood",
                      parent="R78", is_billable=True)
    
    hierarchy.add_code("R79", "Other abnormal findings of blood chemistry", parent="R70-R79")
    hierarchy.add_code("R79.9", "Abnormal finding of blood chemistry, unspecified",
                      parent="R79", is_billable=True)
    
    # Add some index cross-references (semantic similarities)
    hierarchy.add_index_reference("R78.71", "R78.9")
    hierarchy.add_index_reference("R78.9", "R79.9")
    hierarchy.add_index_reference("M25.561", "M25.562")
    
    return hierarchy


if __name__ == "__main__":
    # Test the hierarchy
    hierarchy = create_sample_hierarchy()
    
    print("Testing ICD-10-CM Hierarchy")
    print("=" * 50)
    
    # Test expansion for M25.561 (Pain in right knee)
    code = "M25.561"
    print(f"\nExpanding code: {code} - {hierarchy.get_code_description(code)}")
    print(f"Siblings: {hierarchy.get_siblings(code)}")
    print(f"Cousins: {hierarchy.get_cousins(code)}")
    print(f"1-hop neighbors: {hierarchy.get_1hop_neighbors(code)}")
    print(f"2-hop neighbors: {hierarchy.get_2hop_neighbors(code)}")
    print(f"Full expansion: {hierarchy.expand_candidates(code)}")
    
    # Save to file
    hierarchy.save_to_file("/home/claude/clinical_coding_pipeline/data/sample_icd_hierarchy.json")
    print("\n✓ Hierarchy saved to sample_icd_hierarchy.json")