import os, yaml, torch, wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project="financial-sentiment-llm", name="phi3-t4-force-fp16", config=config)

    # 1. Force everything to float16
    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, # Force compute to fp16
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], 
        quantization_config=bnb_config, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=compute_dtype # Force weights to fp16
    )
    
    # 3. Prepare for kbit training
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=config['lora_dropout'],
        task_type="CAUSAL_LM"
    )

    # 5. Load Data
    dataset = load_dataset("csv", data_files={"train": "data/train.csv", "val": "data/val.csv"})

    # 6. SFTConfig - Force fp16 and disable bf16
    sft_config = SFTConfig(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['num_epochs'],
        logging_steps=config['logging_steps'],
        save_strategy="steps", 
        save_steps=config['save_steps'],
        report_to="wandb", 
        fp16=True,             
        bf16=False,            
        dataset_text_field="text",
        max_length=config['max_seq_length'],
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model, 
        train_dataset=dataset["train"], 
        eval_dataset=dataset["val"],
        peft_config=peft_config, 
        processing_class=tokenizer, 
        args=sft_config,
    )

    # 8. CRITICAL: Manual cast to fix the "NotImplementedError"
    # This forces the LoRA layers into float32/float16 to avoid BF16 scaling errors
    for name, module in trainer.model.named_modules():
        if "lora_" in name:
            module.to(torch.float32) # GradScaler handles fp32/fp16 perfectly

    print("🚀 Starting training. Precision cast to FP32/FP16 for T4 compatibility.")
    trainer.train()
    
    final_path = os.path.join(config['output_dir'], "final_adapter")
    trainer.model.save_pretrained(final_path)
    print(f"✅ Success! weights saved to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    train()