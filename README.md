<div align="center" >
    <img src="assets/logo.png" height=120 alt="" style="margin-bottom:px"/> 

**MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure**

[![arXiv](https://img.shields.io/badge/arXiv-2409.15371-b31b1b.svg)](https://arxiv.org/abs/2409.15371)
<a href="https://github.com/huggingface/trl/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/trl.svg?color=blue"></a>

[🤗PEFT](https://github.com/huggingface/peft/tree/main/src/peft/tuners/bone) |
[📑Paper](https://arxiv.org/abs/2409.15371)(ICLR'25 Poster) |
[📘Documentation](https://huggingface.co/docs/peft/main/package_reference/bone) |
[🛠️Installation](https://huggingface.co/docs/peft/install#source) |
[❓Issues](https://github.com/JL-er/MiSS/issues/new/choose)

</div>

> \[!IMPORTANT\]
>
> **MiSS** is supported by [Huggingface/peft](https://github.com/huggingface/peft.git)
> 
>Paper Version(Bone->DiSHA->MiSS)
>
> We are still improving **MiSS**


MiSS (Matrix Shard Sharing) is a novel Parameter-Efficient Fine-Tuning (PEFT) method designed to address the trade-off between adaptability and efficiency in Large Language Models. The core approach of MiSS involves a simple shard-sharing mechanism. It achieves low-rank adaptation by decomposing a weight matrix into multiple fragments and then utilizing a shared, trainable "common fragment." The final low-rank update matrix is constructed by replicating these shared, partitioned shards.


## 🚀News
- **\[2026.01.26\]** Our paper was accepted by ICLR2026
- **\[2025.06.13\]** Our paper was accepted by ES-Fomo III workshop @ICML2025! 
- **\[2025.05.16\]** We released a new version of our paper!(MiSS) 
- **\[2024.12.31\]** We released a new version of our paper!(DiSHA) 
- **\[2024.11.05\]** Merged into the Hugging Face PEFT repo! 
- **\[2024.09.19\]** Our paper was available on ArXiv!(Bone) 
- **\[2024.08.07\]** First proposed the Bone method! 

## 🔧Installation
### HF Model
MiSS is currently being merged into the official PEFT repository. In the future, you will only need to run `pip install peft`
```
git clone https://github.com/huggingface/peft.git
cd peft
pip install -e .
```
```
git clone https://github.com/JL-er/MiSS.git
```
```
cd MiSS
sh scripts/run_miss.sh
```

### Advanced Usage
```
from transformers import AutoModelForCausalLM
from peft import MissConfig, TaskType, get_peft_model

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
peft_config = MissConfig(
    r=16,
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# prints: trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.1193

# now perform training on your dataset, e.g. using transformers Trainer, then save the model
model.save_pretrained("qwen2.5-3b-miss")
```
<p>
  <img src="./assets/from.png"/>
  <img src="./assets/space.png"/>
</p>



### PEFT Arena
<p>
  <img src="./assets/peft-compare.png"/>
</p>

<details>
<summary>🔍 <b>Bat: Block Affine Transformation</b> </summary>
We conducted extensive experiments on both NLU and NLG tasks to validate the effectiveness of Bone. It outperforms many LoRA variants and surpasses LoRA in terms of memory consumption and computational efficiency.
However, we found that Bone results in the updates between different shards within the same matrix being collinear. Specifically, all shards in the weights use the same trainable matrix for updates, causing the updates of all shards to be collinear, restricting the model’s expressive power. To address the issue of linear correlation among shard updates, our initial idea was to use a trainable coefficient matrix to control the updates of different shards. However, this approach would increase additional parameters.

Inspired by methods like PiSSA that leverage pre-trained weight matrix information, we propose \textbf{Block Affine Transformation (Bat)} to break update collinearity without adding parameters. The key insight is to leverage pre-trained weights $\mathbf{W}_0$ as nonlinear projectors:  

1. \textbf{Tensor Factorization}:  

   \textbullet\  Reshape $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ into 4D tensor $\mathcal{W}_0 \in \mathbb{R}^{\frac{k}{r} \times \frac{d}{r} \times r \times r}$  
   
   \textbullet\  Reshape $\mathbf{D} \in \mathbb{R}^{r \times d}$ into $\mathcal{D} \in \mathbb{R}^{\frac{d}{r} \times r \times r}$  

2. \textbf{Affine Transformation}:  
   Compute shard-specific updates via tensor contraction:  
   \[
   \Delta \mathcal{W} = \mathcal{W}_0 \times \mathcal{D} + \mathcal{D} \quad \in \mathbb{R}^{\frac{k}{r} \times \frac{d}{r} \times r \times r}  
   \]  
   where $\times_3$ denotes contraction along the third dimension.  

3. \textbf{Reconstruction}:  
   Reshape $\Delta \mathcal{W}$ to obtain full update matrix:  
   \[
   \Delta \mathbf{W} = \operatorname{Reshape}(\Delta \mathcal{W}) \in \mathbb{R}^{d \times k}  
   \]  
Bat allows for flexible configuration of different dimensional transformation strategies based on the settings of $\mathbf{D}$. For example: 

Bat-Row:  
  Reshape $\mathbf{W}_0$ into $\mathcal{W} \in \mathbb{R}^{\frac{d}{r} \times \frac{k}{r} \times r \times r}$ and $\mathbf{D} \in \mathbb{R}^{r \times k}$ into $\mathcal{D} \in \mathbb{R}^{\frac{k}{r} \times r \times r}$
  
Bat-Col:  
    Reshape $\mathbf{W}_0$ into $\mathcal{W} \in \mathbb{R}^{\frac{k}{r} \times \frac{d}{r} \times r \times r}$ and $\mathbf{D} \in \mathbb{R}^{r \times d}$ into $\mathcal{D} \in \mathbb{R}^{\frac{d}{r} \times r \times r}$

The term $\mathcal{W}_0 \times \mathcal{D}$ introduces shard-dependent perturbations proportional to $\mathbf{W}_0$'s singular vectors, breaking the collinearity enforced by Bone's shared $\mathbf{D}$.


</details>


# Citation
If you find this repo useful, please consider citing our works:
# Citation
```bib
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
