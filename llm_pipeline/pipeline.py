import re
from prompts import GENERATOR_PROMPT, VERIFIER_PROMPT
from expansion import ICDExpander


class ClinicalCodingPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.expander = ICDExpander()   # no icd.init()

    def parse_codes(self, text):
        """
        Extracts ICD codes and descriptions from model output.
        Format expected: CODE: DESCRIPTION
        """
        pattern = r"([A-Z][0-9A-Z.]+):\s*(.+)"
        return re.findall(pattern, text)

    def run(self, note: str):
        # Step 1: Generator
        gen_prompt = GENERATOR_PROMPT.format(note=note)
        gen_output = self.llm.generate(gen_prompt)

        extracted = self.parse_codes(gen_output)

        if not extracted:
            return ["No ICD codes extracted"]

        final_codes = []

        # Step 2 + 3: Expansion + Verification
        for code, desc in extracted:
            expanded_codes = self.expander.expand(code, desc) or []

            candidates = [f"{code}: {desc}"]

            for c in expanded_codes[:10]:
                    candidates.append(c)


            formatted = "\n".join(
                [f"{i}. {c}" for i, c in enumerate(candidates)]
            )

            ver_prompt = VERIFIER_PROMPT.format(
                note=note, candidates=formatted
            )

            choice = self.llm.generate(ver_prompt)

            try:
                idx = int(choice.strip())
                final_codes.append(candidates[idx])
            except:
                final_codes.append(f"{code}: {desc}")

        return final_codes
