Autoregressive Next-Token Prediction with GPT-2
This experiment demonstrates next-token prediction using an autoregressive GPT-2 model.

The scripts load the openai-community/gpt2 causal language model and predict the next tokens for the prompt:

Artificial Intelligence is ruling the
For the prompt, each script obtains the model logits for the final token, converts them into probabilities, and prints the five most likely next tokens and their probabilities.

Files
main.py - Main script. Downloads and loads GPT-2 from Hugging Face, then prints the top five next-token predictions. It uses the public model-loading interface without an explicit Hugging Face token.
local.py - Loads GPT-2 while reading an optional Hugging Face token from an environment variable named HF_TOKEN. It otherwise performs the same top-five next-token prediction as main.py.
requirements.txt - Python packages required by the scripts.
Requirements
Install the dependencies with:

python -m pip install -r requirements.txt
The required packages are:

transformers
torch
python-dotenv
huggingface_hub
Run the Main Script
From the Week-7 folder:

python main.py
The first run requires internet access so the GPT-2 tokenizer and model can be downloaded from Hugging Face. After they are cached, subsequent runs can use the local cache.

Run with a Hugging Face Token
If the model requires authenticated access or you want to pass an explicit Hugging Face credential, create a .env file in the Week-7 folder:

HF_TOKEN=your_hugging_face_token
Then run:

python local.py
Keep the .env file private and do not commit the token to source control.

Output
Both scripts print:

The input prompt.
The top five candidate next tokens.
The probability of each candidate.
Example model output format:

Input sentence: Artificial Intelligence is ruling the

Top 5 next-token predictions:
--------------------------------
1. Token: ' world' | Probability: 12.3456%
The exact tokens and probabilities vary with the GPT-2 model and runtime environment.