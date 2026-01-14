import os
import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    wandb.init(project="financial-sentiment-llm", name="phi3-sentiment-run")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(config['model_id'], quantization_config=bnb_config, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    dataset = load_dataset("csv", data_files={"train": "data/train.csv", "validation": "data/val.csv"})

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50, # Periodic Checkpoints
        report_to="wandb" # Requirement: Monitoring
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained("./outputs/final_adapter")
    wandb.finish()

if __name__ == "__main__":
    train()