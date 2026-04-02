from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types


@dataclass
class GenAIModel:
    client: genai.Client
    model_name: str
    system_instruction: str
    response_mime_type: Optional[str] = None
    temperature: Optional[float] = None

    def generate_content(self, content: str):
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type=self.response_mime_type,
            temperature=self.temperature,
        )
        return self.client.models.generate_content(
            model=self.model_name,
            contents=content,
            config=config,
        )
