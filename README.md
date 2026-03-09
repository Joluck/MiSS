<div align="center">
<img src="assets/logo.png" height="120" alt="MiSS Logo" />

# MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure(ICLR'26)

[![arXiv](https://img.shields.io/badge/arXiv-2409.15371-b31b1b.svg)](https://arxiv.org/abs/2409.15371)
[![License](https://img.shields.io/github/license/huggingface/trl.svg?color=blue)](https://github.com/huggingface/trl/blob/main/LICENSE)

</div>

A lightweight Parameter‑Efficient Fine‑Tuning (PEFT) technique that introduces **Matrix Shard Sharing (MiSS)** to balance adaptability and efficiency in large language models.

---

## 📌 Table of Contents

1. [🚀 News](#-news)
2. [🔧 Installation](#-installation)
3. [⚡ Quick Start](#-quick-start)
4. [📊 Benchmarks & Results](#-benchmarks--results)
5. [📚 Citation](#citation)

---

> **Note:** MiSS is supported by [Hugging Face PEFT](https://github.com/huggingface/peft.git) and is actively being improved.

## 🚀 **News**

**🎯 ICLR 2026 paper accepted on 2026‑01‑26!**

<details>
<summary>Previous updates</summary>

- **2025‑06‑13:** Accepted at ES‑Fomo III workshop @ ICML 2025
- **2025‑05‑16:** Released MiSS paper version
- **2024‑12‑31:** Released DiSHA paper version
- **2024‑11‑05:** Integrated into Hugging Face PEFT repo
- **2024‑09‑19:** ArXiv release (Bone)
- **2024‑08‑07:** First proposed the Bone method

</details>

---

## 🔧 Installation

MiSS will eventually be available via `pip install peft`. For now:

```bash
# clone and install PEFT (editable mode)
git clone https://github.com/huggingface/peft.git
cd peft
pip install -e .

# grab this repository
git clone https://github.com/JL-er/MiSS.git
cd MiSS
sh scripts/run_miss.sh
```

---

## ⚡ Quick Start

```python
from transformers import AutoModelForCausalLM
from peft import MissConfig, TaskType, get_peft_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", device_map=device
)

peft_config = MissConfig(r=16, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable: 3,686,400 / 3,089,625,088 (0.12%)

# training follows using Trainer or custom loop
model.save_pretrained("qwen2.5-3b-miss")
```

<p align="center">
<img src="./assets/from.png" alt="from" />
<img src="./assets/space.png" alt="space" />
</p>

---

## 📊 Benchmarks & Results

<p align="center">
<img src="./assets/peft-compare.png" alt="PEFT comparison" />
</p>

MiSS outperforms common LoRA variants while reducing memory and compute. See the paper for detailed numbers.

<details>
<summary>🔍 Block Affine Transformation (Bat)</summary>

Our experiments revealed that Bone's shard updates are collinear, limiting expressiveness. Bat uses pre-trained weights as nonlinear projectors to break this collinearity without extra parameters:

1. **Tensor factorization** of $\mathbf{W}_0$ and $\mathbf{D}$.
2. **Affine transformation** via tensor contraction.
3. **Reconstruction** of the full update matrix.

Different reshaping strategies (Bat‑Row, Bat‑Col) offer flexible dimensional control. The full derivation is in the paper.

</details>

---

## 📚 Citation

```bibtex
@misc{kang2025missrevisitingtradeofflora,
  title={MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure},
  author={Jiale Kang and Qingyu Yin},
  year={2025},
  eprint={2409.15371},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2409.15371},
}
```

---

Thanks for checking out MiSS! Contributions and issues are welcome.
