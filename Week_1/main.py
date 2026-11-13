import os
from dotenv import load_dotenv
from google import genai

# Load variables from the .env file
load_dotenv()

# Read the Gemini API key securely from the environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check that .env exists "
        "and contains GEMINI_API_KEY=your_key."
    )

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Send a prompt to the model (Updated model name)
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="What is Artificial Intelligence? Explain in simple words."
)

# Display the generated answer
print("\nLLM Response:")
print(response.text)