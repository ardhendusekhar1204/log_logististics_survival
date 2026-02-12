**This work is an implementation of the paper "Spatially-Aware Mixture of Experts with Log-Logistic Survival Modeling for Whole-Slide Images"(https://arxiv.org/abs/2511.06266). Accurate survival prediction from histopathology whole-slide images (WSIs) remains challenging due to gigapixel resolutions, spatial heterogeneity, and complex survival distributions. We introduce a comprehensive computational pathology framework that addresses these limitations through four synergistic innovations: (1) Quantile-Gated Patch Selection to dynamically identify prognostically relevant regions; (2) Graph-Guided Clustering that groups patches by spatial-morphological similarity to capture phenotypic diversity; (3) Hierarchical Context Attention to model both local tissue interactions and global slide-level context; and (4) an Expert-Driven Mixture of Log-Logistics module that flexibly models complex survival distributions. On large-scale TCGA cohorts, our method achieves state-of-the-art performance, with time-dependent concordance indices of 0.644±0.059 on LUAD, 0.751±0.037 on KIRC, and 0.752±0.011 on BRCA—significantly outperforming both histology-only and multimodal benchmarks. The framework provides improved calibration and interpretability, advancing the potential of WSIs for personalized cancer prognosis.**

## 🔄 Pipeline Steps for creating the Virtual Environment

1. Create and activate environment
   ```
   conda create -n sgcmll python=3.9 

   conda activate sgcmll
   ```

2. Install PyTorch (CUDA 11.3 build)
   ```
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. Install RAPIDS cuDF/cuML
   ```
   pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==23.10.0 cuml-cu11==23.10.0
   ```

4. Install other Python dependencies
   ```
   pip install tqdm lifelines munch tensorboardX einops h5py seaborn
   ```
5. While running the train script, some errors may appear after creating the virtual environment. If errors appear then follow the steps in the text file 'changes_in_virtual_env_types_python_file.txt' right after the creation of the         virtual environment.
   
## 📂 Data Preparation

1. TCGA data:
Download diagnostic WSIs and corresponding clinical metadata from TCGA(https://portal.gdc.cancer.gov/).

2. Patch extraction:
Use the CLAM WSI processing tool(https://github.com/mahmoodlab/CLAM) to crop WSIs into 256×256 patches at 40× magnification.

3. Feature extraction:
Extract patch-level features with a ViT(https://github.com/lunit-io/benchmark-ssl-pathology#pre-trained-weights) model pretrained on large-scale WSI collections using self-supervised learning.

4. Annoation files and folder structure: 
Prepare you own 'wsi_annos_vit-s-dino-p16.txt' file.

## 📂 Dataset Structure

```
data/
├── kirc/
│   ├── 5fold_wsi-rnaseq/
│   │   ├── fold1/
│   │   │   ├── train.txt
│   │   │   └── val.txt
│   │   ├── fold2/
│   │   ├── fold3/
│   │   ├── fold4/
│   │   └── fold5/
│   ├── clinical.csv
│   └── wsi_annos_vit-s-dino-p16.txt
└── luad/
    ├── 5fold_wsi-rnaseq/
    │   ├── fold1/
    │   │   ├── train.txt
    │   │   └── val.txt
    │   ├── fold2/
    │   ├── fold3/
    │   ├── fold4/
    │   └── fold5/
    ├── clinical.csv
    └── wsi_annos_vit-s-dino-p16.txt
```
    
## 🧪  Train the model
python train.py --config configs/luad_sgcmll.yaml

## 🌡️ Final Plot
python plot.py --config configs/luad_sgcmll.py
<p align="center">
  <img src="plots/luad.png" alt="Centered Image" width="500"/>
</p>

## 📚 Acknowledgements
Thanks Zekang Yang, Hong Liu, and Xiangdong Wang for the public access of the codebase https://github.com/yang-ze-kang/SCMIL
