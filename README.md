# HyBioSum: Hybrid Biomedical Long-Document Summarization

This repository provides the official implementation of **HyBioSum**, a hybrid summarization framework that combines a **supervised extractive summarizer (Longformer-based)** with a **zero-shot abstractive large language model (LLM, e.g., Mistral)**. The pipeline is designed for **biomedical long-document summarization** and has been evaluated on **PubMed** and **CORD-19** datasets.

---

## 🚀 Project Workflow

The HyBioSum pipeline has **five main stages**:

1. **Data Conversion** → Convert abstractive task into extractive form (oracle sentence labels).  
2. **Preprocessing** → Sentence segmentation, tokenization, oracle label generation.  
3. **Training** → Fine-tune the extractive summarizer (Longformer).  
4. **Evaluation** → Evaluate extractive model & prepare dataset for abstractive stage.  
5. **Abstractive Summarization** → Run zero-shot LLMs on extractive outputs.  
6. **Inference** → Run the full pipeline on raw documents.  

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```
## 📘 Usage

### 1. Data Conversion

Convert abstractive datasets (e.g., PubMed, CORD-19) into extractive format.

```bash
python data_conversion.py \
  --dataset_dir /path/to/pubmed_raw \
  --dataset_name pubmed \
  --output_dir /path/to/pubmed_dataset \
  --num_processes 2 \
  --selection_size 8 \
  --head
```
**Inputs:**
- Dataset with splits: `train`, `validation`, `test`
- Each split must contain the fields: `article`, `abstract`

**Outputs:**
- Dataset with the following columns:
  - `id` → unique document identifier  
  - `sentences` → segmented sentences from the article  
  - `ref_summary` → gold abstractive summary (reference)  
  - `labels` → oracle binary labels for extractive training



### 2. Preprocessing

Prepare dataset for Longformer-based extractive summarization.
```bash
python preprocess.py \
  --dataset_dir /path/to/pubmed_dataset \
  --output_dir /path/to/preprocess_dataset \
  --model_name allenai/longformer-base-4096 \
  --num_processes 2
```

**Outputs:**
- Dataset with columns:
  - `id` → document ID
  - `sentences` →  segmented sentences
  - `ref_summary` → gold abstract
  - `labels` → oracle binary labels
 



### 3. Train Extractive Summarizer

Fine-tune Longformer-based extractive summarizer.
```bash
python train.py \
  --dataset_dir /path/to/preprocess_dataset \
  --model_name allenai/longformer-base-4096 \
  --epochs 25 \
  --batch_size 4 \
  --lr 3e-4 \
  --history_path ./outputs/history_pubmed.csv \
  --checkpoints_path ./outputs/checkpoints_pubmed \
  --checkpoints_freq 1 \
  --checkpoint_best \
  --mixed_precision
```
### 4. Evaluate Extractive Summarizer
```bash
python evaluate.py \
  --dataset_dir /path/to/preprocess_dataset \
  --checkpoint ./outputs/checkpoints_pubmed/cp_allenai_longformer-base-4096_ep001.tar \
  --model_name allenai/longformer-base-4096 \
  --batch_size 2 \
  --selection_size 20 \
  --input_size 4096 \
  --output_dir ./outputs/evaluation
```

**Outputs:**
- extractive_dataset/ with article, abstract, extractive_summaries
- CSV and metrics JSON files




### 5. Abstractive Summarization

Run zero-shot LLMs (Mistral, Phi, LLaMA, etc.) on extractive summaries.
```bash
python abstractive.py \
  --dataset_path ./outputs/evaluation/extractive_dataset \
  --output_path ./abstractive_summaries \
  --model_name unsloth/mistral-7b-instruct-v0.2-bnb-4bit
```
### 6. Inference (End-to-End)

Run the full pipeline (extractive + abstractive) on a raw text file.
```bash
python inference.py \
  --input_file ./sample_article.txt \
  --checkpoint ./outputs/checkpoints_pubmed/cp_allenai_longformer-base-4096_ep002.tar \
  --extractive_model_name allenai/longformer-base-4096 \
  --abstractive_model_name unsloth/mistral-7b-instruct-v0.2-bnb-4bit \
  --num_sentences 20 \
  --output_file ./outputs/sample_inference
```

**Outputs:**
- JSON + CSV with article extractive_summary abstractive_summary


