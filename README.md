# FLARE-IDS: Explainable Language-Model-Based Network Intrusion Detection with Dual-Layer SHAP and Attention Analysis

This repository contains the code, trained models, and supporting files for the FLARE-IDS framework. FLARE-IDS fine-tunes DistilBERT on serialized network-flow records from the UNSW-NB15 dataset for multi-class intrusion detection, and adds a dual-layer explainability pipeline combining SHAP token attribution with transformer attention visualization.

## Repository Structure

```
.
├── 00_preprocessing_baselines.ipynb   # Data loading, preprocessing, feature serialization, CNN and LSTM baselines
├── 01_transformer_training.ipynb      # DistilBERT and RoBERTa fine-tuning, class imbalance ablation study
├── 02_shap_analysis.ipynb             # SHAP PartitionExplainer setup and per-class token attribution
├── 03_attention_analysis.ipynb        # Attention entropy, feature-level attention, SHAP-attention consistency
├── 04_revision_extras.ipynb           # MCC, PR-AUC, bootstrap CIs, token-length distribution, CIC majority baseline
├── label_encoder.pkl                  # Scikit-learn LabelEncoder mapping class names to integer labels
├── feat_cols.json                     # JSON file containing the 42 feature names used for serialization
└── README.md
```

### Trained Model

The fine-tuned DistilBERT checkpoint is hosted on Hugging Face:
**[muzamil-saleem/xai-llm-ids](https://huggingface.co/muzamil-saleem/xai-llm-ids)**

Load it directly with HuggingFace Transformers:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("muzamil-saleem/xai-llm-ids")
tokenizer = DistilBertTokenizerFast.from_pretrained("muzamil-saleem/xai-llm-ids")
```

## Datasets

This project uses two publicly available datasets (not included in this repository):

- **UNSW-NB15** (primary, training and evaluation): https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CIC-IDS2017** (cross-dataset zero-shot transfer evaluation): https://www.unb.ca/cic/datasets/ids-2017.html

Download both datasets and place them in your Google Drive under the paths referenced in Notebook 00.

## Notebook Descriptions

### 00_preprocessing_baselines.ipynb
Loads the raw UNSW-NB15 CSV files and performs the full preprocessing pipeline: missing-value imputation (median for numeric, mode for categorical), Min-Max normalization of 39 numeric features, and retention of 3 categorical features (proto, state, service) as named string tokens. Serializes each record into a natural-language sentence using a fixed template. Trains and evaluates CNN and LSTM baseline models. Saves preprocessed data, predictions, the label encoder, and feature column names to Drive.

### 01_transformer_training.ipynb
Fine-tunes DistilBERT-base-uncased and RoBERTa-base on the serialized text records using HuggingFace Transformers. Implements inverse-frequency class weighting to handle the severe class imbalance in UNSW-NB15. Runs the four-variant class imbalance ablation study (no weights, inverse-frequency, SMOTE only, SMOTE + weights). Records training time, per-class metrics, and overall performance. Saves model checkpoints and all prediction arrays.

### 02_shap_analysis.ipynb
Computes SHAP token attributions using `shap.Explainer` with `shap.maskers.Text` (PartitionExplainer). Evaluates 1,844 stratified test samples against a background of 200 training records. Generates per-class SHAP summary plots, individual force plots, and waterfall plots. Computes SHAP fidelity (correlation between SHAP magnitude and model confidence).

### 03_attention_analysis.ipynb
Extracts attention weights from all 6 layers and 12 heads of the fine-tuned DistilBERT. Computes per-head attention entropy to identify specialized heads. Aggregates attention at the feature level (mapping sub-word tokens back to their source features). Performs SHAP-attention rank correlation analysis (Spearman) and generates feature-importance comparison visualizations.

### 04_revision_extras.ipynb
Computes additional evaluation metrics requested during peer review: Matthews Correlation Coefficient (MCC) for all four models, per-class Average Precision (area under the PR curve) for DistilBERT, bootstrap 95% confidence intervals on macro F1 (1,000 iterations), token-length distribution analysis, and a majority-class baseline for the CIC-IDS2017 zero-shot evaluation.

## Key Results

| Model | Macro F1 | AUC-ROC | MCC | Parameters |
|-------|----------|---------|-----|------------|
| CNN | 0.28 | 0.897 | 0.467 | ~850K |
| LSTM | 0.23 | 0.873 | 0.406 | ~230K |
| RoBERTa | 0.50 | 0.960 | 0.641 | 125M |
| **DistilBERT** | **0.50** | **0.962** | **0.642** | **66M** |

Bootstrap 95% CIs on macro F1: DistilBERT [0.492, 0.514], RoBERTa [0.483, 0.505], CNN [0.272, 0.278], LSTM [0.227, 0.231].

All pairwise McNemar's tests are significant at the Bonferroni-corrected threshold (p < 0.0167).

## Requirements

All notebooks are designed to run on Google Colab with a T4 GPU (free tier). The main dependencies are:

- Python 3.10+
- PyTorch 2.x
- transformers 4.38+
- scikit-learn 1.3+
- shap 0.45+
- pandas, numpy, matplotlib, seaborn
- bertviz (for attention visualization)
- joblib

Install any missing packages in Colab with:
```
!pip install transformers shap bertviz
```

## How to Reproduce

1. Download the UNSW-NB15 and CIC-IDS2017 datasets and upload them to your Google Drive.
2. Run the notebooks in order (00 through 04). Each notebook mounts Google Drive and reads/writes to `MyDrive/XAI_LLM_IDS/`.
3. Notebook 00 must run first as it generates the preprocessed data and baseline predictions that all subsequent notebooks depend on.
4. Notebooks 02 and 03 (SHAP and attention analysis) require the trained DistilBERT checkpoint from Notebook 01. You can alternatively load the pre-trained checkpoint from the Hugging Face link above.
5. Notebook 04 requires the prediction arrays saved by Notebooks 00 and 01.

## Hardware

- Training: Google Colab T4 GPU (16 GB VRAM). DistilBERT fine-tuning takes approximately 4.5 hours for 10 epochs.
- SHAP analysis: approximately 2 additional hours on the same hardware.
- All other notebooks run in under 30 minutes.

## Citation

If you use this code or model in your research, please cite:

```
@article{flare-ids,
  title={FLARE-IDS: Explainable Language-Model-Based Network Intrusion Detection with Dual-Layer SHAP and Attention Analysis},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## License

This project is released for academic and research use. See the LICENSE file for details.
