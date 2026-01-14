import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def get_prediction(model, tokenizer, text):
    prompt = text.split("<|assistant|>")[0] + "<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False,
            use_cache=False  # <--- Add this line to fix the error
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = full_text.split("assistant")[-1].strip().lower()
    return prediction

def run_evaluation():
    print("📋 Loading Test Data...")
    test_df = pd.read_csv("data/test.csv")
    
    # We test on a subset (e.g., 50-100) to save time in Colab, or use len(test_df) for full
    test_samples = test_df.head(100) 
    
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Evaluate Base Model
    print("🤖 Evaluating Base Model (Untrained)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16,  # Changed from torch_dtype to dtype
        device_map="auto",
        trust_remote_code=True
    )
    base_model.config.use_cache = False # <--- Add this line as well
    
    base_preds = []
    for txt in tqdm(test_samples['text']):
        base_preds.append(get_prediction(base_model, tokenizer, txt))
    
    # 2. Evaluate Fine-Tuned Model
    print("🚀 Evaluating Fine-Tuned Model (LoRA)...")
    # Load the LoRA adapter on top of the base model
    ft_model = PeftModel.from_pretrained(base_model, "./outputs/final_adapter")
    
    ft_preds = []
    for txt in tqdm(test_samples['text']):
        ft_preds.append(get_prediction(ft_model, tokenizer, txt))
    
    # 3. Calculate Metrics
    # Extract actual labels from the assistant block in the CSV
    y_true = [txt.split("<|assistant|>")[-1].replace("<|end|>", "").strip().lower() for txt in test_samples['text']]
    
    base_acc = accuracy_score(y_true, base_preds)
    ft_acc = accuracy_score(y_true, ft_preds)
    
    improvement = ((ft_acc - base_acc) / base_acc) * 100 if base_acc > 0 else 0

    print("\n" + "="*30)
    print(f"RESULTS SUMMARY")
    print("="*30)
    print(f"Base Model Accuracy: {base_acc*100:.2f}%")
    print(f"Fine-Tuned Accuracy: {ft_acc*100:.2f}%")
    print(f"Net Improvement:     {improvement:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()