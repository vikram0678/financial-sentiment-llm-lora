import os
import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig # Updated import
from dotenv import load_dotenv

# Load environment variables (WANDB_API_KEY) from .env file
load_dotenv()

def train():
    # 1. Load Configuration
    if not os.path.exists("config/config.yaml"):
        print("❌ Error: config/config.yaml not found!")
        return
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Weights & Biases
    wandb.init(
        project="financial-sentiment-llm",
        name="phi3-sentiment-finetune",
        config=config
    )

    # 3. BitsAndBytes Config (Optimized for T4/Local GPUs)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Model and Tokenizer
    print(f"🚀 Loading model: {config['model_id']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16  
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. Load Processed Dataset
    dataset = load_dataset("csv", data_files={
        "train": "data/train.csv",
        "validation": "data/val.csv"
    })

    # 7. Modern SFTConfig (Combines TrainingArgs + SFT Params)
    sft_config = SFTConfig(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['num_epochs'],
        logging_steps=config['logging_steps'],
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config['save_steps'],
        report_to="wandb",
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        # SFT Specific arguments moved here:
        dataset_text_field="text",
        max_length=config['max_seq_length'], 
        packing=False
    )

    # 8. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer, # Recommended over 'tokenizer'
        args=sft_config,
    )

    # 9. Execute Training
    print("🔥 Starting fine-tuning...")
    trainer.train()

    # 10. Save Results
    final_path = os.path.join(config['output_dir'], "final_adapter")
    trainer.model.save_pretrained(final_path)
    print(f"✅ Training Complete! Weights saved to {final_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train()