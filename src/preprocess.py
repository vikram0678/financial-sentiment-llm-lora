import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def run_preprocessing():
    print("🚀 Starting Financial Data Preprocessing...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)

    # 1. Load Dataset with specific config
    try:
        print("📥 Loading pre-converted Parquet dataset (Config: 78516)...")
        # Specifying the config '78516' as requested by the error message
        dataset = load_dataset(
            "gtfintechlab/financial_phrasebank_sentences_allagree", 
            "78516", 
            split="train"
        )
        df = dataset.to_pandas()
        print(f"✅ Successfully loaded {len(df)} rows.")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 2. Map labels (0: neg, 1: neu, 2: pos)
    # Mirror datasets sometimes use strings or integers; this handles both.
    label_map = {0: "negative", 1: "neutral", 2: "positive", "0": "negative", "1": "neutral", "2": "positive"}
    df['label_text'] = df['label'].map(label_map)

    # 3. Apply Phi-3 Chat Template
    def format_phi3(row):
        system_prompt = "You are a professional financial analyst. Classify the sentiment of the following news sentence."
        user_text = f"Sentence: {row['sentence']}\nSentiment:"
        assistant_reply = str(row['label_text'])
        
        # Proper Phi-3 Template format
        return (f"<|system|>\n{system_prompt}<|end|>\n"
                f"<|user|>\n{user_text}<|end|>\n"
                f"<|assistant|>\n{assistant_reply}<|end|>")

    df['text'] = df.apply(format_phi3, axis=1)

    # 4. Split Dataset (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # 5. Save Splits
    train_df[['text']].to_csv('data/train.csv', index=False)
    val_df[['text']].to_csv('data/val.csv', index=False)
    test_df[['text']].to_csv('data/test.csv', index=False)

    print(f"✅ Preprocessing Complete!")
    print(f"📊 Training:   {len(train_df)} rows")
    print(f"📊 Validation: {len(val_df)} rows")
    print(f"📊 Test:       {len(test_df)} rows")

if __name__ == "__main__":
    run_preprocessing()