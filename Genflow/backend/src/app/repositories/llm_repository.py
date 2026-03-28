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
        # Pass the full original JSON items for maximum context
        import json
        context = ""
        for i, item in enumerate(metadata_list):
            context += f"Reference {i+1} (Full JSON):\n"
            context += json.dumps(item, indent=2, ensure_ascii=False) + "\n"
            context += "-" * 20 + "\n"

        prompt = f"""
You are a world-class AI Image Synthesis Prompt Engineer and Stable Diffusion expert.
Your goal is to generate a comprehensive metadata object for a new image based on a user's intent and high-quality references.

### USER'S GENERATION INTENT:
"{user_intent}"

### REFERENCE DATA (FULL CONTEXT):
{context}

### YOUR MISSION:
1.  **Analyze Styles**: Identify the common artistic styles, lighting, color palettes, and composition patterns in the references.
2.  **Extract Model & Parameters**: Note the common Stable Diffusion models (checkpoints), samplers, CFG scales, and LoRAs used.
3.  **T2I Engineering Best Practices**:
    -   **Prompt Structuring**: Use a weighted, structured prompt. Start with the subject, followed by artistic style, lighting, artist names, and finally quality tags (e.g., masterpiece, best quality, 8k, ultra-detailed).
    -   **Weighting**: Use `(keyword:1.2)` syntax for emphasis where appropriate.
    -   **Negative Prompting**: Construct a robust negative prompt that covers common artifacts (e.g., blurry, out of frame, deformed, low quality, text, watermark).
    -   **LoRA Integration**: Integrate necessary LoRAs if they appear in the references or are relevant to the style, using `<lora:Name:Weight>` format.
    -   **Parameter Selection**: Choose a realistic `CFG scale` (4-8 range usually), `Steps` (20-40), and a matching `Sampler` from the references.

### OUTPUT REQUIREMENTS:
Generate a single JSON object that matches the following schema (aligning with the project's main metadata structure):
- "prompt": The full detailed T2I prompt (including weights and LoRAs).
- "negative_prompt": The comprehensive negative prompt.
- "cfgscale": The recommended CFG scale (string or float).
- "steps": The recommended steps (string or int).
- "sampler": The sampler name (all-caps string, e.g., "DPM++ 2M KARRAS").
- "seed": A random or recommended seed (string, e.g., "1234567890").
- "model": The recommended Stable Diffusion checkpoint name (string).
- "clipskip": Recommended clip skip (string or int, usually "1" or "2").
- "style": A comma-separated string of the aesthetic keywords used.
- "lora": A comma-separated list of friendly LoRA names used.
- "full_metadata_string": (Optional) A summary of the settings in standard SD format.

Return ONLY the raw JSON object. NO Markdown formatting tags.
"""
        return prompt
