import torch
import fitz  # PyMuPDF
import faiss
import numpy as np
import tiktoken
import requests
import gradio as gr
import pandas as pd
import json
import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# --- ESGAnalyzer CLASS ---
class ESGAnalyzer:
    def __init__(self, esg_model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(esg_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(esg_model_name).to(self.device)
        self.model.eval()

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.labels = [
            "Critical_Incident_Risk_Management", "Air_Quality", "GHG_Emissions",
            "Management_Of_Legal_And_Regulatory_Framework", "Director_Removal",
            "Product_Design_And_Lifecycle_Management", "Energy_Management",
            "Competitive_Behavior", "Systemic_Risk_Management", "Supply_Chain_Management",
            "Employee_Health_And_Safety", "Business_Model_Resilience", "Customer_Privacy",
            "Business_Ethics", "Water_And_Wastewater_Management", "Customer_Welfare"
        ]
        self.embeddings = []
        self.text_chunks = []
        self.index = None

    def extract_pdf_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def score_esg(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        return dict(sorted({label: float(score) for label, score in zip(self.labels, scores)}.items(), key=lambda x: x[1], reverse=True))

    def split_text(self, text: str, max_tokens: int = 300) -> List[str]:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        words = text.split()
        chunks, chunk, tokens = [], [], 0
        for word in words:
            word_tokens = len(enc.encode(word))
            if tokens + word_tokens > max_tokens:
                chunks.append(" ".join(chunk))
                chunk, tokens = [], 0
            chunk.append(word)
            tokens += word_tokens
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    def embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode(text)

    def build_faiss_index(self, chunks: List[str]):
        self.text_chunks = chunks
        self.embeddings = [self.embed_text(chunk) for chunk in chunks]
        emb_matrix = normalize(np.array(self.embeddings)).astype('float32')
        self.index = faiss.IndexFlatIP(emb_matrix.shape[1])
        self.index.add(emb_matrix)

    def query_context(self, query: str, top_k: int = 3) -> str:
        query_vec = normalize(self.embed_text(query).reshape(1, -1)).astype('float32')
        _, indices = self.index.search(query_vec, top_k)
        context = "\n\n".join([self.text_chunks[i] for i in indices[0]])
        return context

    def ask_question(self, question: str, ollama_model: str = "deepseek-coder") -> str:
        context = self.query_context(question)
        prompt = f"""You are an expert ESG analyst. Based on the following ESG report context, answer the question thoroughly and professionally:\n\n{context}\n\nQ: {question}\nA:"""
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json()["message"]["content"].strip()
            else:
                return f"❌ Ollama error: {response.text}"
        except Exception as e:
            return f"❌ Ollama connection error: {str(e)}"

    def analyze_pdf(self, pdf_path: str, company_name: str = "Unknown") -> Dict:
        text = self.extract_pdf_text(pdf_path)
        esg_scores = self.score_esg(text)
        chunks = self.split_text(text)
        self.build_faiss_index(chunks)
        return {
            "company": company_name,
            "esg_scores": esg_scores,
            "ask": lambda q: self.ask_question(q)
        }

# --- GRADIO UI FUNCTIONS ---
analyzer = ESGAnalyzer(esg_model_name="ESGBERT/EnvironmentalBERT-environmental")

def analyze(file, company):
    result = analyzer.analyze_pdf(file.name, company)
    scores = result["esg_scores"]

    json_path = f"{company}_esg_scores.json"
    csv_path = f"{company}_esg_scores.csv"

    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2)

    pd.DataFrame(scores.items(), columns=["Category", "Score"]).to_csv(csv_path, index=False)
    return scores, json_path, csv_path

def ask(file, company, question):
    result = analyzer.analyze_pdf(file.name, company)
    return result["ask"](question)

# --- GRADIO UI ---
esg_interface = gr.Interface(
    fn=analyze,
    inputs=[
        gr.File(label="Upload ESG PDF"),
        gr.Textbox(label="Company Name")
    ],
    outputs=[
        gr.JSON(label="ESG Scores"),
        gr.File(label="Download JSON"),
        gr.File(label="Download CSV")
    ],
    title="📊 ESG Score Analyzer With Deepseek Ollama with Gradio",
    description="Upload a company's ESG report to get scores and export results. Runs offline with DeepSeek via Ollama."
)

qa_interface = gr.Interface(
    fn=ask,
    inputs=[
        gr.File(label="Upload ESG PDF"),
        gr.Textbox(label="Company Name"),
        gr.Textbox(label="Ask ESG-related Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🤖 ESG Q&A with DeepSeek",
    description="Ask ESG questions using DeepSeek LLM locally via Ollama."
)

gr.TabbedInterface([esg_interface, qa_interface], ["ESG Scoring", "Ask Questions"]).launch()
