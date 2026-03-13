from dotenv import load_dotenv
import os

load_dotenv()

print("Project:", os.getenv("LANGCHAIN_PROJECT"))
print("API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))