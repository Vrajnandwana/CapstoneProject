import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Azure OpenAI setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYEMENT_NAME")

# Path to your saved model
MODEL_DIR = "./biobart-mri"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Example test cases
test_texts = [
    """Mild bone marrow edema of the patella noted. Subchondral cystic change of the lateral
tibial spine noted with surrounding focal bone marrow edema. Mucoid degeneration of the anterior
cruciate ligament noted, otherwise grossly intact. Posterior cruciate ligament is grossly intact...""",
    """No prior exams. Normal marrow stores are seen in the visualized osseous elements...
Disc desiccation is noted at all levels visualized in the cervical and upper thoracic spine...""",
    # add more cases...
]

# Tokenize
inputs = tokenizer(
    test_texts,
    return_tensors="pt",
    max_length=1024,
    truncation=True,
    padding=True
)

# Generate summaries (impressions)
summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=160,
    min_length=60,
    num_beams=4 ,
    early_stopping=True
)

generated_summaries = [
    tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids
]

# 🔹 Validate with Azure OpenAI (ChatGPT as Radiologist)
def validate_with_chatgpt(findings, impression):
    prompt = f"""
You are a senior radiologist. 
I will give you MRI findings and an AI-generated impression. 
Evaluate whether the impression is medically valid, precise, and consistent with the findings.

Findings:
{findings}

AI Impression:
{impression}

Please respond in JSON with:
- "validity" (0-100 accuracy score),
- "issues" (if any),
- "feedback" (short radiologist-style feedback).
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# Run validation for each test case
for i, (findings, summary) in enumerate(zip(test_texts, generated_summaries)):
    print(f"\n=== Test Case {i+1} ===")
    print("📝 Findings:\n", findings[:400], "...")
    print("🤖 AI Impression:\n", summary)

    validation = validate_with_chatgpt(findings, summary)
    print("✅ Radiologist Validation (ChatGPT):\n", validation)
