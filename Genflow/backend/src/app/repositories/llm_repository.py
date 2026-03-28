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

    def generate_metadata_from_intent(self, metadata_list: list, user_intent: str):
        prompt = self._build_generation_prompt(metadata_list, user_intent)
        # Force JSON response if possible, or use a strict instruction
        response = self.model.generate_content(prompt)
        # Attempt to extract JSON from response
        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content

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

    def _build_generation_prompt(self, metadata_list: list, user_intent: str):
        context = ""
        for i, item in enumerate(metadata_list):
            context += f"Reference Result {i+1}:\n"
            context += f"Prompt: {item.get('prompt')}\n"
            context += f"Style: {item.get('style', 'N/A')}\n"
            context += f"LoRAs: {item.get('loras', 'N/A')}\n"
            context += f"CFG Scale: {item.get('cfgscale')}\n"
            context += f"Steps: {item.get('steps')}\n"
            context += f"Sampler: {item.get('sampler')}\n"
            context += "-" * 20 + "\n"

        prompt = f"""
You are an expert AI image generation assistant. 
Based on the style and technical parameters of the {len(metadata_list)} reference images provided below, generate a new set of metadata for a user's specific intent.

User's Intent: "{user_intent}"

Reference Styles and Parameters:
{context}

Your task:
1. Extract common style keywords, lighting, and artist names from the reference prompts and styles.
2. Determine the optimal technical parameters (CFG Scale, Steps, Sampler) based on what worked best for these references.
3. Combine these style elements with the User's Intent to create a new, high-quality detailed prompt.
4. Output the result strictly as a JSON object with the following keys:
   - "prompt": The new detailed image generation prompt.
   - "negative_prompt": A suitable negative prompt based on common practices.
   - "cfgscale": Recommended CFG scale (float/int).
   - "steps": Recommended number of steps (int).
   - "sampler": Recommended sampler name (string).
   - "style": Summary of the stylistic keywords used.
   - "loras": Recommended LoRAs to use (string or list).

Return ONLY the JSON object.
"""
        return prompt
