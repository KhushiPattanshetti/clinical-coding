import simple_icd_10_cm as icd
import requests

INDEX_API = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


class ICDExpander:
    def __init__(self):
        pass   # No init() needed in newer library versions

   

    def siblings(self, code):
        if not icd.is_valid_item(code):
            return []   # skip invalid codes safely

        parent = icd.get_parent(code)
        if parent:
            return icd.get_children(parent)
        return []


    def cousins(self, code):
        parent = icd.get_parent(code)
        if not parent:
            return []
        grandparent = icd.get_parent(parent)
        if not grandparent:
            return []
        cousins = []
        for p in icd.get_children(grandparent):
            if p != parent:
                cousins.extend(icd.get_children(p))
        return cousins

    def index_neighbors(self, term, max_results=10):
        results = []
        try:
            r = requests.get(
                INDEX_API,
                params={"terms": term, "sf": "code", "maxList": max_results},
                timeout=10,
            )
            data = r.json()
            if len(data) > 3:
                for code in data[3]:
                    results.append(code)
        except Exception:
            pass
        return results

    def expand(self, code, desc):
        if not icd.is_valid_item(code):
            return []

        sib = self.siblings(code)
        ...

