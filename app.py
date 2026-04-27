import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# ---- Load Model ----
MODEL_PATH = "your-huggingface-username/your-model-name"  # change this

config = PeftConfig.from_pretrained(MODEL_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path,
    num_labels=2
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

labels = {0: "✅ Non-Stereotyped", 1: "⚠️ Stereotyped"}

# ---- Inference Function ----
def detect_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()

    return {
        "Prediction": labels[pred],
        "Confidence": f"{probs[pred].item() * 100:.2f}%",
        "Non-Stereotyped": f"{probs[0].item() * 100:.2f}%",
        "Stereotyped": f"{probs[1].item() * 100:.2f}%"
    }

# ---- Gradio UI ----
examples = [
    ["Women are bad at driving."],
    ["She was promoted because of her hard work."],
    ["All Asians are good at math."],
    ["He volunteers at the local shelter every weekend."]
]

with gr.Blocks(theme=gr.themes.Soft(), title="Bias & Stereotype Detector") as demo:
    gr.Markdown("""
    # 🔍 Bias & Stereotype Detection
    **DeBERTa-V3 fine-tuned with LoRA** to detect stereotyped language in text.
    """)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter a sentence",
                placeholder="Type a sentence here...",
                lines=3
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output = gr.JSON(label="Results")

    gr.Examples(examples=examples, inputs=text_input)
    submit_btn.click(fn=detect_bias, inputs=text_input, outputs=output)

demo.launch()
