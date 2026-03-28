import os
import google.generativeai as genai
from dotenv import load_dotenv

ENV_PATH = "/Users/mgccvmacair/Myproject/Academic/Genflow/Genflow/backend/src/.env"
load_dotenv(ENV_PATH)
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
