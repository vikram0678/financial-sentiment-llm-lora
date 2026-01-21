# 🚀 Financial Sentiment Analysis with Phi-3 & LoRA

A production-ready machine learning pipeline for classifying financial news sentiment using **Microsoft Phi-3-mini** and **LoRA (Low-Rank Adaptation)**. This project demonstrates efficient fine-tuning of large language models on domain-specific data, achieving 98% accuracy with minimal computational resources.

**Key Achievement:** Improved sentiment classification accuracy from 67% to 98% using only 2% of model parameters.

---

## 📊 Performance Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|-----------|------------|
| **Accuracy** | 67% | **98%** | **+31%** ✅ |
| **Trainable Parameters** | 100% | 2% | 50x Reduction |
| **Training Time** | N/A | 5-10 min | Efficient |
| **GPU Memory** | 16GB | 14GB | Optimized |

The fine-tuned model significantly outperforms the base model at classifying financial sentiment, particularly on market-specific language patterns.

---

## 📁 Project Folder Structure

```
financial-sentiment-llm-lora/
├── config/
│   └── config.yaml                 # Model hyperparameters & settings
├── data/                           # Auto-generated dataset splits
│   ├── train.csv                   # 3,388 training sentences (70%)
│   ├── val.csv                     # 816 validation sentences (15%)
│   └── test.csv                    # 636 test sentences (15%)
├── outputs/
│   ├── final_adapter/              # Fine-tuned LoRA weights
│   │   ├── adapter_config.json     # LoRA configuration
│   │   └── adapter_model.safetensors  # Model weights
│   └── evaluation_report.txt       # Performance metrics & results
├── src/
│   ├── preprocess.py               # Load & format Financial Phrasebank
│   ├── train.py                    # T4-optimized training pipeline
│   └── evaluate.py                 # Model comparison & evaluation
├── requirements.txt                # Dependencies
├── .gitignore                      # Files to ignore in git
└── README.md                       # This file
```

---

## 📦 Model Weights (Download Required)

The fine-tuned LoRA adapters must be downloaded separately:

| Component | Size | Location |
|-----------|------|----------|
| **LoRA Adapter Weights** | 150MB | [🔗 Google Drive Link](https://drive.google.com/file/d/1hCRPA4UeWuIZBrJIPaEie40wMpTcTZRi/view?usp=sharing) |
| **Config Files** | 10KB | Included in repo |

**Installation Steps:**
1. Click the Google Drive link above
2. Download the ZIP file
3. Extract the contents
4. Place the folder in `outputs/final_adapter/`
5. Model is ready to use!

---

## 🛠️ Environment Setup (Critical!)

Version conflicts with `bitsandbytes`, `triton`, and `trl` are the most common cause of setup failures. Follow this exact sequence:

### Step 1: Clean Uninstall Old Packages

```bash
pip uninstall -y bitsandbytes triton accelerate peft trl transformers
```

This removes any conflicting versions from previous installations.

### Step 2: Install Compatible Versions

```bash
# Core packages (order matters!)
pip install -q bitsandbytes>=0.45.2 triton rich

# ML/Data packages
pip install -q transformers accelerate peft trl datasets scikit-learn pandas wandb pyyaml python-dotenv
```

### Step 3: Verify GPU Setup

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

Expected output: `GPU Available: True`

---

## 🚀 Quick Start (Google Colab)

**Recommended for beginners - Free GPU included!**

### Step 1: Enable GPU

Go to [Google Colab](https://colab.research.google.com) and create a new notebook.

```
Runtime → Change runtime type → GPU (T4) → Save
```

### Step 2: Clone & Setup

```bash
git clone https://github.com/vikram0678/financial-sentiment-llm-lora.git
cd financial-sentiment-llm-lora

# Clean environment setup
pip uninstall -y bitsandbytes triton accelerate peft trl transformers
pip install -q bitsandbytes>=0.45.2 triton rich
pip install -r requirements.txt
```

### Step 3: Setup Weights & Biases (Optional)

Track training progress with W&B:

```python
import wandb, os

# Get API key from https://wandb.ai/authorize
os.environ["WANDB_API_KEY"] = "your_api_key_here"
wandb.login()
```

### Step 4: Download Model Weights

Download from the Google Drive link and extract to `outputs/final_adapter/`

### Step 5: Run Pipeline

```bash
# Prepare data
python src/preprocess.py

# Train model (5-10 minutes)
python src/train.py

# Evaluate performance
python src/evaluate.py
```

### Step 6: Download Results

```python
from google.colab import files

# Download trained weights
!zip -r final_adapter.zip ./outputs/final_adapter
files.download('final_adapter.zip')

# Download evaluation report
files.download('outputs/evaluation_report.txt')
```

---

## 💻 Local Setup (VS Code with GPU)

**For developers who want full control and faster iteration.**

### Step 1: Clone Repository

```bash
git clone https://github.com/vikram0678/financial-sentiment-llm-lora.git
cd financial-sentiment-llm-lora
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Clean Install Dependencies

```bash
pip uninstall -y bitsandbytes triton accelerate peft trl transformers
pip install -q bitsandbytes>=0.45.2 triton rich
pip install -r requirements.txt
```

### Step 4: Download Model Weights

Download from Google Drive and place in `outputs/final_adapter/`

### Step 5: Run Project

```bash
python src/preprocess.py    # Prepare data
python src/train.py         # Train model (optional)
python src/evaluate.py      # Evaluate performance
```

---

## 📊 Dataset Information

**Financial Phrasebank (All Agree)**

| Attribute | Details |
|-----------|---------|
| **Total Sentences** | 4,840 financial news sentences |
| **Sentiment Classes** | Positive, Neutral, Negative |
| **Quality** | High (all 16 annotators agreed) |
| **Average Length** | ~20 words per sentence |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Source** | Academic financial NLP research |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: bitsandbytes` | `pip install bitsandbytes>=0.45.2` |
| `RuntimeError: CUDA out of memory` | Reduce `batch_size: 4` in config.yaml |
| `No module named 'trl'` | Run clean install from environment setup |
| `ModuleNotFoundError: triton` | `pip install triton` separately |
| `GPU not detected in Colab` | Runtime → Change type → T4 GPU |
| `Model weights not found` | Ensure `outputs/final_adapter/` exists with extracted files |
| `Training very slow` | Run `torch.cuda.is_available()` to verify GPU is active |

---

## ❓ Frequently Asked Questions

**Q: Do I need a GPU to train?**  
A: Yes, for training you need a GPU. For testing the pre-trained model, CPU works but is slower (1-2 minutes vs 10 seconds).

**Q: How long does training take?**  
A: Approximately 5-10 minutes on Google Colab T4 GPU.

**Q: Can I use my own financial dataset?**  
A: Yes! Modify `src/preprocess.py` to load your CSV file instead of Financial Phrasebank.

**Q: Will this work for languages other than English?**  
A: The model is trained on English financial text. Other languages would require retraining.

**Q: How can I improve beyond 98% accuracy?**  
A: Use a larger training dataset, experiment with different hyperparameters, or use a larger base model like Phi-3-small.

**Q: Can I use an NVIDIA A100 GPU instead of T4?**  
A: Yes! Change `FP16` to `BF16` in `train.py` for improved precision on A100.

---

## 🔗 References

- [Microsoft Phi-3 Model Documentation](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)
- [LoRA Research Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library Documentation](https://github.com/huggingface/peft)
- [Weights & Biases Experiment Tracking](https://wandb.ai)
- [HuggingFace Hub](https://huggingface.co)

---

## 📝 Quick Reference Commands

```bash
# Complete setup from scratch
git clone https://github.com/vikram0678/financial-sentiment-llm-lora.git
cd financial-sentiment-llm-lora
pip uninstall -y bitsandbytes triton accelerate peft trl transformers
pip install -q bitsandbytes>=0.45.2 triton rich
pip install -r requirements.txt

# Run full pipeline
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Check GPU status
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📄 License

MIT License - Free to use and modify.