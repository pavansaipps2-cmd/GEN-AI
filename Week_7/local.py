
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Model configuration
model_name = "openai-community/gpt2"

print("Loading GPT-2 model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN
)

model.eval()

# Input prompt
prompt = "Artificial Intelligence is ruling the"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Get model output
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits for the next token
logits = outputs.logits[:, -1, :]

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

# Find top 5 candidates
top_k = torch.topk(probabilities, k=5)

print("\nInput sentence:", prompt)
print("\nTop 5 next-token predictions:")
print("--------------------------------")

for i in range(5):
    token_id = top_k.indices[0][i].item()
    probability = top_k.values[0][i].item()

    token = tokenizer.decode([token_id])

    print(
        f"{i + 1}. Token: {repr(token)} "
        f"| Probability: {probability:.4%}"
    )

print("\nExperiment completed!")