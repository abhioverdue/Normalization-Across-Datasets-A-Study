# Normalization in Deep Learning: When and Why Does It Actually Matter?

A controlled empirical study comparing BatchNorm, LayerNorm, GroupNorm, and no normalization across MNIST and CIFAR-10.

## Structure

```
├── MNIST_Notebook.ipynb       # MNIST experiments (5 epochs, BS ∈ {2,4,8,32,64}, 3 seeds)
├── CIFAR10_Notebook.ipynb     # CIFAR-10 experiments (10 epochs, BS ∈ {8,64}, 2 seeds)
├── mnist_full_results.csv     # Raw results — MNIST
├── cifar_full_results.csv     # Raw results — CIFAR-10
└── normalization_paper.docx   # Full research paper
```

## Setup

```bash
pip install torch torchvision matplotlib pandas numpy
```

## Reproducing Results

Run each notebook top-to-bottom. Datasets download automatically via `torchvision`. CIFAR-10 uses checkpoint saving (`cifar_progress.csv`) so interrupted runs resume safely.

## Key Findings

| Dataset  | Does norm help? | Most sensitive to batch size |
|----------|-----------------|------------------------------|
| MNIST    | No              | No Norm (but gap is noise)   |
| CIFAR-10 | Yes             | No Norm (−2.70 pp at BS=8)   |

- **MNIST**: all methods score 98.6–98.9% — normalization is irrelevant.
- **CIFAR-10**: no normalization collapses at small batch sizes; GroupNorm is most robust.
- **BatchNorm** degrades at BS < 16 due to noisy batch statistics — prefer GroupNorm or LayerNorm when batch size is constrained.

## Citation

If you use this work, please cite the paper included in this repository.
