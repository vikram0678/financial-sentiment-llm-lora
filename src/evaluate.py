import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def clean_output(text):
    """Helper to clean model output of extra tokens and whitespace."""
    text = text.lower().strip()
    # Remove common extra tokens that might be generated
    for word in ["positive", "neutral", "negative"]:
        if word in text:
            return word
    return "unknown"

def get_prediction(model, tokenizer, text):
    # Prompt stops exactly where the assistant should begin
    prompt = text.split("<|assistant|>")[0] + "<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, # We only need 1 word
            do_sample=False,
            use_cache=False 
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Get everything after 'assistant'
    raw_prediction = full_text.split("assistant")[-1]
    return clean_output(raw_prediction)

def run_evaluation():
    print("📋 Loading Test Data...")
    test_df = pd.read_csv("data/test.csv")
    test_samples = test_df.head(100) 
    
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Evaluate Base Model
    print("🤖 Evaluating Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    base_model.config.use_cache = False
    
    base_preds = [get_prediction(base_model, tokenizer, txt) for txt in tqdm(test_samples['text'])]
    
    # 2. Evaluate Fine-Tuned Model
    print("🚀 Evaluating Fine-Tuned Model...")
    ft_model = PeftModel.from_pretrained(base_model, "./outputs/final_adapter")
    ft_preds = [get_prediction(ft_model, tokenizer, txt) for txt in tqdm(test_samples['text'])]
    
    # 3. Extract Ground Truth accurately
    y_true = []
    for txt in test_samples['text']:
        # Extract word between assistant and end tokens
        label = txt.split("<|assistant|>")[-1].split("<|end|>")[0].strip().lower()
        y_true.append(label)
    
    base_acc = accuracy_score(y_true, base_preds)
    ft_acc = accuracy_score(y_true, ft_preds)
    
    # Calculate improvement (with safety for zero base_acc)
    improvement = ((ft_acc - base_acc) / (base_acc if base_acc > 0 else 1)) * 100

    print("\n" + "="*30)
    print(f"RESULTS SUMMARY")
    print("="*30)
    print(f"Base Model Accuracy: {base_acc*100:.2f}%")
    print(f"Fine-Tuned Accuracy: {ft_acc*100:.2f}%")
    print(f"Net Improvement:     {improvement:.2f}%")
    print("="*30)
    
    # Show a sample comparison
    print("\nSample Comparisons (Ground Truth | Base | Fine-Tuned):")
    for i in range(3):
        print(f"Example {i+1}: {y_true[i]} | {base_preds[i]} | {ft_preds[i]}")

if __name__ == "__main__":
    run_evaluation()