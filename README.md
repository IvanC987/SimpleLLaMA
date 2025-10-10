# SimpleLLaMA

**An open, educational framework for understanding and reproducing the complete training and alignment pipeline of modern Large Language Models (LLMs).**

---

## Development Status

This project/readme as a whole is currently under **active development**.  
Certain parts, like documentation and RLHF modules, are still being finalized, with expected completion around **[Late November, 2025]**.  
While the pretraining and SFT stages is mostly complete and functional, certain components (e.g., RLHF implementation, extended benchmarks) remains WIP.

Feel free to explore, test, and contribute, but please note that the repository may change significantly before its final release.

---

## Overview

**SimpleLLaMA** is a comprehensive project designed to demystify the lifecycle of LLM development, starting from raw data to a functioning aligned model.  
It provides a transparent implementation of the three main stages of language model creation:

1. **Pretraining** — Unsupervised training of a 1.3B-parameter transformer model on a 50B-token curated corpus.  
2. **Supervised Fine-Tuning (SFT)** — Instruction-tuning on human-written datasets to enable task-following and conversational behavior.  
3. **Reinforcement Learning from Human Feedback (RLHF)** — Alignment via **Direct Preference Optimization (DPO)** to refine model responses based on human preference data.

In addition, the project includes modules for **data preparation, tokenization, evaluation, and deployment**, enabling users to experiment with every major step of the modern LLM pipeline.

---

## Key Features

- **Full LLM Training Lifecycle:** Covers pretraining → SFT → DPO alignment in one unified framework.  
- **Scalable Transformer Architecture:** Implements a 1.3B parameter model inspired by LLaMA, trained efficiently on 50B tokens.  
- **Alignment Techniques:** Integrates full Fine-Tuning (possibly LoRA later on) and DPO for behavioral training and preference optimization.  
- **Evaluation Framework:** Benchmarked on common understanding benchmarks including **MMLU**, **HellaSwag**, **ARC**, **PIQA**.  
- **Deployment Ready:** Includes inference utilities for text generation and context management.  
- **Documentation Site:** Fully documented with architecture breakdowns, training logs, configurations, and detailed walkthroughes of the entire repository.

---

## Getting Started

To play with the model: 

Clone the repository:
```bash
git clone https://github.com/IvanC987/SimpleLLaMA
cd SimpleLLaMA
pip install -r requirements.txt
pip install -e .
```

(More to be added later here once completed)


If you wish to run custom pretraining, fine-tuning, or reinforcement learning, please refer to the `Custom Training` section in the `SimpleLLaMA Documentations` page

---

## Documentation & Technical Report

For an in-depth look into the architecture, experiments, and training methodology, visit the full documentation:

**📘 Documentation:** [https://ivanc987.github.io/SimpleLLaMA/](https://ivanc987.github.io/SimpleLLaMA/)  
**📄 Technical Report:** [Technical_Report.md](./TECHNICAL_REPORT.md)

---

## Benchmarks

| Dataset         | Metric | Score  |
|-----------------|---------|--------------|
| MMLU            | Accuracy | XX.X% |
| ARC (Challenge) | Accuracy | XX.X% |
| ARC (Easy)      | Accuracy | XX.X% |
| HellaSwag       | Accuracy | XX.X% |
| PIQA            | Accuracy | XX.X% |

*(See the `Misc/Benchmarking` section in documentations for more details)*

---

## License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.
