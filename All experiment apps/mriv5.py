# # biobart_app.py
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from dotenv import load_dotenv
# import os
# from openai import AzureOpenAI
# from unstructured.partition.pdf import partition_pdf  # ✅ Unstructured PDF extraction
# import tempfile

# # ---- Load environment variables ----
# load_dotenv()
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_DEPLOYEMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYEMENT_NAME")

# # ---- Model config ----
# MODEL_DIR = "./biobart-mri"
# DEVICE = "cpu"

# @st.cache_resource(show_spinner=True)
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         MODEL_DIR,
#         device_map=None,
#         dtype=torch.float32
#     )
#     model.to(DEVICE)
#     return tokenizer, model

# tokenizer, model = load_model()

# # ---- Helper: generate summary ----
# def generate_summary(text, min_len=40, max_len=160, beams=4):
#     inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     summary_ids = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=max_len,
#         min_length=min_len,
#         num_beams=beams,
#         early_stopping=True
#     )
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# # ---- Azure GPT Helper ----
# def enhance_with_gpt(raw_summary: str):
#     client = AzureOpenAI(
#         api_key=AZURE_OPENAI_API_KEY,
#         api_version=AZURE_OPENAI_API_VERSION,
#         azure_endpoint=AZURE_OPENAI_ENDPOINT
#     )
#     prompt = f"""
#     You are a medical assistant. Here is an MRI impression draft:

#     {raw_summary}

#     Validate it for clinical correctness and complete any missing or important details
#     so that it becomes a clear, professional radiology impression.
#     """
#     response = client.chat.completions.create(
#         model=AZURE_OPENAI_DEPLOYEMENT_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=300
#     )
#     return response.choices[0].message.content

# # ---- Helper: extract findings from PDF using Unstructured ----
# def extract_findings_from_pdf(pdf_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(pdf_file.read())
#         tmp_path = tmp_file.name

#     elements = partition_pdf(filename=tmp_path)
#     full_text = "\n".join([el.text for el in elements if el.text])
    
#     # Extract text until 'Impression'
#     impression_keywords = ["Impression", "IMPRESSION"]
#     split_index = len(full_text)
#     for kw in impression_keywords:
#         idx = full_text.find(kw)
#         if idx != -1:
#             split_index = min(split_index, idx)
#     findings_text = full_text[:split_index].strip()
#     return findings_text

# # ---- Streamlit UI ----
# st.title("🧠 BioBART MRI Impression Assistant")

# input_option = st.radio("Select input type:", ["Text Input", "Unstructured PDF Extraction"])

# text_input = ""

# if input_option == "Text Input":
#     text_input = st.text_area("Paste MRI findings here:", height=300)
# elif input_option == "Unstructured PDF Extraction":
#     pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
#     if pdf_file:
#         extracted_text = extract_findings_from_pdf(pdf_file)
#         st.info("Text extracted up to 'Impression' heading. You can edit or add more findings before generating.")
#         text_input = st.text_area("Edit/Add Findings:", value=extracted_text, height=300)

# if st.button("Generate & Enhance Impression"):
#     if text_input.strip():
#         # Step 1: BioBART raw impression
#         raw_summary = generate_summary(text_input)
#         st.subheader("🔹 Raw Impression (BioBART)")
#         st.text_area("Raw Impression", value=raw_summary, height=200)

#         # Step 2: Azure GPT Validation & Completion
#         if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
#             enhanced_summary = enhance_with_gpt(raw_summary)
#             st.subheader("🤖 Enhanced Impression (with GPT)")
#             st.text_area("Enhanced Impression", value=enhanced_summary, height=200)
#         else:
#             st.warning("⚠️ Azure OpenAI not configured. Skipping enhancement.")
#     else:
#         st.warning("Please provide MRI findings before generating.")


        # biobart_app.py
        import streamlit as st
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        from dotenv import load_dotenv
        import os
        from openai import OpenAI
        from unstructured.partition.pdf import partition_pdf
        import tempfile
        from PIL import Image
        import io

        # ---- Load environment variables ----
        load_dotenv()
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_OPENAI_DEPLOYEMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYEMENT_NAME")

        # ---- OpenAI GPT-4V API Key ----
        OPENAI_API_KEY = "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"

        # ---- Model config ----
        MODEL_DIR = "./biobart-mri"
        DEVICE = "cpu"

        @st.cache_resource(show_spinner=True)
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_DIR,
                device_map=None,
                dtype=torch.float32
            )
            model.to(DEVICE)
            return tokenizer, model

        tokenizer, model = load_model()

        # ---- Helper: generate summary using BioBART ----
        def generate_summary(text, min_len=40, max_len=160, beams=4):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len,
                min_length=min_len,
                num_beams=beams,
                early_stopping=True
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # ---- Azure GPT Helper ----
        def enhance_with_gpt(raw_summary: str):
            if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT):
                return raw_summary  # skip if Azure not configured

            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            prompt = f"""
            You are a medical assistant. Here is an MRI impression draft:

            {raw_summary}

            Validate it for clinical correctness and complete any missing or important details
            so that it becomes a clear, professional radiology impression.
            """
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYEMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content

        # ---- Helper: extract findings from PDF using Unstructured ----
        def extract_findings_from_pdf(pdf_file):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name

            elements = partition_pdf(filename=tmp_path)
            full_text = "\n".join([el.text for el in elements if el.text])

            # Extract text until 'Impression'
            impression_keywords = ["Impression", "IMPRESSION"]
            split_index = len(full_text)
            for kw in impression_keywords:
                idx = full_text.find(kw)
                if idx != -1:
                    split_index = min(split_index, idx)
            findings_text = full_text[:split_index].strip()
            return findings_text

        # ---- Helper: prepare any image type for GPT-4V ----
        def prepare_image_bytes(image_file):
            image = Image.open(image_file).convert("RGB")  # ensure RGB
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")  # convert to PNG
            return buffered.getvalue()

        # ---- Helper: analyze MRI image using GPT-4V ----
        def analyze_mri_image(image_file):
            img_bytes = prepare_image_bytes(image_file)
            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a medical assistant AI. Analyze this MRI image and summarize the findings as a professional radiology impression."
                        },
                        {
                            "type": "input_image",
                            "image_bytes": img_bytes,
                            "image_name": "mri_image.png"
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=500  # correct parameter
            )

            return response.choices[0].message.content

        # ---- Streamlit UI ----
        st.title("🧠 BioBART MRI Impression Assistant")

        input_option = st.radio("Select input type:", ["Text Input", "Unstructured PDF Extraction", "MRI Image Upload"])
        text_input = ""

        # --- Text input ---
        if input_option == "Text Input":
            text_input = st.text_area("Paste MRI findings here:", height=300)

        # --- PDF extraction ---
        elif input_option == "Unstructured PDF Extraction":
            pdf_file = st.file_uploader("Upload MRI PDF", type=["pdf"])
            if pdf_file:
                extracted_text = extract_findings_from_pdf(pdf_file)
                st.info("Text extracted up to 'Impression' heading. You can edit or add more findings before generating.")
                text_input = st.text_area("Edit/Add Findings:", value=extracted_text, height=300)

        # --- MRI image upload ---
        elif input_option == "MRI Image Upload":
            img_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg", "tif", "bmp"])
            if img_file:
                st.image(img_file, caption="Uploaded MRI Image", use_column_width=True)
                st.info("Analyzing MRI image. This may take some time...")
                findings_text = analyze_mri_image(img_file)
                st.subheader("🔹 Extracted Findings from Image")
                st.text_area("Findings", value=findings_text, height=300)
                text_input = findings_text  # feed into BioBART & GPT enhancement

        # ---- Generate & Enhance Impression ----
        if st.button("Generate & Enhance Impression"):
            if text_input.strip():
                # Step 1: BioBART raw impression
                raw_summary = generate_summary(text_input)
                st.subheader("🔹 Raw Impression (BioBART)")
                st.text_area("Raw Impression", value=raw_summary, height=200)

                # Step 2: Azure GPT Validation & Completion
                enhanced_summary = enhance_with_gpt(raw_summary)
                st.subheader("🤖 Enhanced Impression (with GPT)")
                st.text_area("Enhanced Impression", value=enhanced_summary, height=200)
            else:
                st.warning("Please provide MRI findings before generating.")


