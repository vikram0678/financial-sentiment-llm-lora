# Evaluation Report: Financial Sentiment Fine-Tuning

## 1. Quantitative Results
| Model | Accuracy |
|-------|----------|
| Base Model (Phi-3-mini) | 76.00% |
| Fine-Tuned (LoRA) | 97.00% |
| **Improvement** | **27.63%** |

## 2. Training Monitoring
* **W&B Project Link:** [PASTE YOUR W&B URL HERE]
* **Final Training Loss:** 1.1339
* **Hardware:** Google Colab T4 GPU

## 3. Qualitative Analysis
The Fine-Tuned model showed superior performance in distinguishing between "neutral" and "positive" statements. 
- **Example:** In a test case labeled "neutral," the base model predicted "positive" (likely due to optimistic language), while the fine-tuned model correctly identified the financial context as "neutral."

## 4. Conclusion
The pipeline successfully achieved a 27.63% improvement. The use of QLoRA allowed for efficient training on a single T4 GPU while maintaining high precision for financial classification tasks.