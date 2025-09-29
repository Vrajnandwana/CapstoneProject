from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import torch

MODEL_DIR = "./biobart-mri"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# ---- Multi-line input helper ----
def multiline_input(prompt):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line.strip())
    return " ".join(lines)

# ---- Collect MRI findings ----
user_text = multiline_input("Enter MRI findings (press ENTER twice when done):")

inputs = tokenizer(user_text, return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=160,
    min_length=40,
    num_beams=4,
    early_stopping=True
)

generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n🔹 Generated Impression:\n", generated_summary)

# ---- Ground truth impression ----
reference_summary = multiline_input("\nEnter ground-truth impression (press ENTER twice when done):")

if reference_summary.strip():
    rouge = load("rouge")  # load once here
    scores = rouge.compute(predictions=[generated_summary], references=[reference_summary])

    print("\n📊 ROUGE Scores:")
    for k, v in scores.items():
        if hasattr(v, "mid"):
            print(f"{k}: {v.mid.fmeasure:.4f}")
        else:  # some scores are plain floats
            print(f"{k}: {v:.4f}")
else:
    print("\n⚠️ No ground-truth provided, skipping ROUGE evaluation.")
