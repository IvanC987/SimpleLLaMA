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

This project is licensed under the **MIT License**.  
Feel free to use, extend, or adapt it for research or application purposes.

---

## Author

**Ivan Cao**  
Senior CS Student | University of Mississippi  
Open to collaboration and research questions.  
GitHub: [https://github.com/IvanC987/](https://github.com/IvanC987/)

---

## Acknowledgements

This project was inspired by LLaMA, DeepSeek, and various other open source Large Language Models

Papers: 
- [LLaMA: Open and Efficient Foundation Language Models (Meta)](https://arxiv.org/abs/2302.13971)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

Videos:
- [Umar Jamil's 'Coding LLaMA 2 from scratch in PyTorch' Video](https://www.youtube.com/watch?v=oM4VmoabDAI)
- [Dr. Karpathy's 'Let's reproduce GPT-2 (124M)' Video](https://www.youtube.com/watch?v=l8pRSuU81PU)

Datasets:
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), derived from FineWeb and released under the Open Data Commons Attribution (ODC-By) license.  
  See: [“The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale”](https://arxiv.org/abs/2406.17557).

Portions of the model architecture are adapted from:
- [hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama) by Umar Jamil (MIT License)


Much of the implementation also borrows design clarity from these excellent open-source efforts.

---

