# SpaTranslator
![alt text](Model_Overview.png)

- 
  **SpaTranslator** is a graph-based generative model for cross-modality prediction in spatial omics data. It enables the in silico generation of missing omics modalities—such as spatial ATAC-seq, RNA-seq, and histone modification profiles—across paired or unpaired tissue sections, by leveraging spatial neighborhood information and adversarial learning.
---

## 🧠 Highlights

- 🔄 **Cross-modality generation** between spatial transcriptomics and epigenomics/proteomics
- 📈 **Superior performance** on 12 benchmark datasets over state-of-the-art methods (multiDGD, JAMIE, scButterfly, scPair)
- 🧬 Supports multiple data types: RNA, ATAC, histone marks (H3K27me3, H3K4me3), protein
- 📍 **Graph Neural Network** integration for spatial structure modeling
- 🧪 Flexible in intra- and inter-dataset prediction tasks

---

## 📂 Datasets

SpaTranslator is benchmarked on:
- **Spatial-ATAC-RNA-seq** (mouse brain)
- **MISAR-seq** (mouse brain at stages E11, E13, E15, E18)
- **10x Visium RNA-Protein** (human tonsil&lymph node)

> Detailed documentation is written in the jupyter notebook [Tutorial1 for A2R task](./tutorial/A2R/MISAR-seq_A2R_step1_batch_effect_correction.ipynb) ; [Tutorial2 for R2A task](./tutorial/R2A/MISAR-seq_R2A_step1_batch_effect_correction.ipynb)
---

## 🚀 Installation Guide

**Install with pip**  
```bash
conda create -n SpaTranslator python==3.9
pip install spatranslator
```

**Install PyTorch Geometric and Related Dependencies**
It's crucial to install versions of these packages that match your PyTorch and CUDA versions: 
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-source

The following command installs pyg_lib, torch_scatter, torch_sparse, torch_cluster, and torch_spline_conv.
Note: The URL https://data.pyg.org/whl/torch-2.6.0+cu124.html specifies wheels for PyTorch 2.6.0 and CUDA 12.4.
Adjust ${TORCH} and ${CUDA} in the URL below to match your installed PyTorch version and CUDA version. You can find the correct URL on the PyTorch Geometric installation page.

```bash
# Example for PyTorch 2.6.0 and CUDA 12.4
# YOU MUST ADJUST THE URL BASED ON YOUR PYTORCH AND CUDA VERSION
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
---

## 📄 License

SpaTranslator is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software for academic and non-commercial purposes with proper attribution.

---

## 📬 Contact

If you have questions, feature requests, or would like to contribute:

- 💡 **Issues**: Please submit via the [GitHub Issues Page]()
- ✉️ **Email**: `donghongyu@westlake.edu.cn` /  `maosheng@westlake.edu.cn`
