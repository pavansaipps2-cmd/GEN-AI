
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
model_name = "openai-community/gpt2"

print("Loading GPT-2 model...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

# Input prompt
prompt = "AI is taking over the"

# Convert text into token IDs
inputs = tokenizer(prompt, return_tensors="pt")

# Generate model predictions
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits for the next token
logits = outputs.logits[:, -1, :]

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

# Select top 5 candidates
top_k = torch.topk(probabilities, k=5)

print("\nInput sentence:", prompt)
print("\nDisplayed next-token predictions:")
print()

for i in range(5):
    token_id = top_k.indices[0][i].item()
    probability = top_k.values[0][i].item()

    token = tokenizer.decode([token_id])

    print(
        f"{i + 1}. Token: {repr(token)} "
        f"| Probability: {probability:.4%}"
    )