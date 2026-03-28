import google.generativeai as genai
from app.core.config import settings

class LLMRepository:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def generate_summary(self, metadata_list: list):
        prompt = self._build_prompt(metadata_list)
        response = self.model.generate_content(prompt)
        return response.text

    def _build_prompt(self, metadata_list: list):
        context = ""
        for i, item in enumerate(metadata_list):
            context += f"Result {i+1}:\n"
            context += f"ID: {item.get('id')}\n"
            context += f"Prompt: {item.get('prompt')}\n"
            context += f"Model: {item.get('model')}\n"
            context += f"LoRAs: {item.get('loras')}\n"
            context += f"CFG Scale: {item.get('cfgscale')}\n"
            context += f"Steps: {item.get('steps')}\n"
            context += f"Sampler: {item.get('sampler')}\n"
            context += "-" * 20 + "\n"

        prompt = f"""
Below are the top 5 image generation results from our gallery search. 
Analyze these results and provide:
1. A summary of the visual themes and concepts present in these images based on their prompts.
2. An analysis of the common technical parameters (Model, CFG, Steps, Sampler) and their similarities.
3. Why these images are considered similar in our search space.

Search Results:
{context}

Please respond in a structured format (Markdown).
"""
        return prompt
