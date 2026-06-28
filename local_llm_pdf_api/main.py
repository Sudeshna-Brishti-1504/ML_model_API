import os
import uuid
from fastapi import FastAPI, File, Form, UploadFile
import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Local LLM PDF Question Answering API")

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
tokenizer = None
model = None


@app.get("/")
def home():
    return {
        "message": "Local LLM PDF API is running",
        "endpoints": ["/load-model", "/ask-pdf"],
    }


@app.post("/load-model")
def load_model():
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.eval()

    return {"message": "Model loaded successfully", "model_name": MODEL_NAME}


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


@app.post("/ask-pdf")
async def ask_pdf(file: UploadFile = File(...), question: str = Form(...)):
    global tokenizer, model

    if model is None or tokenizer is None:
        return {"error": "Model not loaded. Please call /load-model first."}

    # All lines below are now properly indented inside the function!
    file_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, file_id + "_" + file.filename)

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    pdf_text = extract_text_from_pdf(pdf_path)
    pdf_text = pdf_text[:3000]

    prompt = f"""
You are a helpful AI assistant.

Answer the question only using the PDF content given below.

PDF CONTENT:
{pdf_text}

QUESTION:
{question}

ANSWER:
"""

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full_response.split("ANSWER:")[-1].strip()

    return {
        "question": question,
        "answer": answer,
        "pdf_file": file.filename,
        "characters_used_from_pdf": len(pdf_text),
    }