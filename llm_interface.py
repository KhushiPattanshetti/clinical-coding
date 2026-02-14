from groq import Groq
from config import GROQ_API_KEY, MAX_TOKENS, TEMPERATURE

class LLMInterface:
    def __init__(self, model=None):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model or "llama-3.3-70b-versatile"

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
