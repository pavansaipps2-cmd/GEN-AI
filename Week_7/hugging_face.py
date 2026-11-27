
import os
import math
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in .env")

if not MODEL_ID:
    raise ValueError("HF_MODEL_ID not found in .env")

# Connect to Hugging Face
client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
)

# Input prompt
prompt = "The sun rises in the"

print("Input sentence:", prompt)
print("\nConnecting to Hugging Face Inference API...")

try:
    result = client.text_generation(
        prompt,
        max_new_tokens=1,
        details=True,
        top_n_tokens=5,
        do_sample=False
    )

    print("\nGenerated token:", repr(result.generated_text))

    print("\nTop 5 next-token predictions:")
    print("--------------------------------")

    for i, token in enumerate(result.details.top_tokens, start=1):
        probability = math.exp(token.logprob)

        print(
            f"{i}. Token: {repr(token.text)} "
            f"| Probability: {probability:.4%}"
        )

    print("\nExperiment completed!")

except Exception as e:
    print("\nHugging Face API request failed!")
    print("Error:", type(e).__name__)
    print("Details:", str(e))

    print(
        "\nCheck whether the selected model has a compatible "
        "Inference Provider and supports detailed token output."
    )