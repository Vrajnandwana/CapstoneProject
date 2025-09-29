import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables (Azure credentials)
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

# Path to your saved BioBART model
MODEL_DIR = "./biobart-mri"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# ---------- Step 1: Read MRI findings from text file ----------
with open("mri_findings.txt", "r", encoding="utf-8") as f:
    mri_findings = f.read().strip()

print("\n📄 MRI Findings:\n", mri_findings)

# ---------- Step 2: Generate impression ----------
inputs = tokenizer(mri_findings, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=160,
    min_length=40,
    num_beams=4,
    early_stopping=True
)
generated_impression = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n🤖 Generated Impression:\n", generated_impression)

# ---------- Step 3: Load optional ground-truth ----------
reference_impression = None
if os.path.exists("radiologist_impression.txt"):
    with open("radiologist_impression.txt", "r", encoding="utf-8") as f:
        reference_impression = f.read().strip()

# ---------- Step 4: Validation ----------
if reference_impression:
    print("\n📄 Ground Truth (Radiologist Impression):\n", reference_impression)

    # ROUGE evaluation
    rouge = load("rouge")
    scores = rouge.compute(predictions=[generated_impression], references=[reference_impression])
    print("\n📊 ROUGE Scores:")
    for k, v in scores.items():
         if hasattr(v, "mid"):  # some versions return an object
            print(f"{k}: {v.mid.fmeasure:.4f}")
         else:
            print(f"{k}: {float(v):.4f}")   

    # ChatGPT validation with ground truth
    validation_prompt = f"""
    You are a radiologist. 
    MRI Findings: {mri_findings}
    Generated Impression: {generated_impression}
    Radiologist Impression: {reference_impression}

    Compare the generated impression with the radiologist impression.
    Tell if it is clinically correct, highlight differences, and give a score out of 100.
    """

else:
    # ChatGPT validation without ground truth
    validation_prompt = f"""
    You are a radiologist.
    MRI Findings: {mri_findings}
    Generated Impression: {generated_impression}

    Check if the generated impression is clinically valid based on the findings.
    Give reasoning and a confidence score out of 100.
    """

response = client.chat.completions.create(
    model=deployment_name,
    messages=[{"role": "system", "content": "You are an expert radiologist validating AI generated  reports of mri scan only ."},
              {"role": "user", "content": validation_prompt}],
    max_tokens=max
)

print("\n✅ ChatGPT Validation:\n", response.choices[0].message.content)
