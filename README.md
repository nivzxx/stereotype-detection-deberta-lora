# Stereotype Detection using DeBERTa (LoRA Fine-Tuning)

## 📌 Problem Statement
Detect stereotypical or biased statements in text using a transformer-based NLP model.

## 🚀 Approach
- Used a pre-trained transformer model for text classification.
- Applied parameter-efficient fine-tuning using LoRA.
- Built an end-to-end NLP pipeline: preprocessing, training, and inference.

## 🧠 Model Used
- DeBERTa (microsoft/deberta-v3-base)
- LoRA Fine-Tuning
- Binary Classification (Stereotype vs Not Stereotype)

## 🛠 Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Google Colab

## 🔍 Example Outputs
- "Women are bad drivers" → Stereotype  
- "Everyone deserves equal rights" → Not Stereotype  

## 📎 How to Run
1. Open the notebook in Google Colab
2. Run all cells
3. Modify input text to test predictions
