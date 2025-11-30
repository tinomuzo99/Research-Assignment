# Research Assignment — Utilizing Transformer Models for Analysing Unstructured Feedback in Recommender Systems 
**Author:** Tinomutendayi Muzondidya  
**Institution:** Stellenbosch University  
**Year:** 2025  

This repository contains all the Jupyter notebooks developed for my thesis research project on **representation learning for recommendation systems**.  
The project compares different embedding strategies (GloVe vs. MiniLM) and architectures (BERT-tiny classifier vs. DeepCoNN) on Amazon review datasets.

Due to data size restrictions, the datasets and pretrained MiniLM model files used for the experiments are not stored in this repository.  
However, **all data and models required to reproduce the experiments are stored in my public Google Drive**:

 **Google Drive Data Folder:**  
https://drive.google.com/drive/folders/1QN5Pt0JWpInnvhn8KkmCudWkYWrnEMvN

---

#  Repository Structure

The repo contains the following notebooks:

| Notebook | Description |
|---------|-------------|
| **glove-embeddings-bert-tiny-architecture.ipynb** | Uses GloVe embeddings + BERT-tiny architecture for sentiment/review modeling. |
| **glove-embeddings-deepconn-architecture.ipynb** | Uses GloVe embeddings + DeepCoNN architecture for user-item interaction modeling. |
| **minilm-embeddings-bert-tiny-architecture.ipynb** | Uses MiniLM sentence embeddings + BERT-tiny architecture. |
| **minilm-embeddings-deepconn-architecture.ipynb** | Uses MiniLM sentence embeddings + DeepCoNN architecture. |

Each notebook represents a different combination of **embedding type** and **architecture**, allowing for direct comparison between models.

---

#  Data & Model Files (Google Drive)

The Google Drive folder contains:

### **1. MiniLM Model Files (Local Offline Version)**  
Located in:  
`minilm-l6-v2-local/all-MiniLM-L6-v2/`

Includes:
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`
- `pytorch_model.bin`
- `model.safetensors`

These files allow the project to load **all-MiniLM-L6-v2** using `AutoModel.from_pretrained()` without internet access.

---

### **2. GloVe Word Embeddings**
Located in:
`glove6b100dtxt/glove.6B.100d.txt`

Used in the GloVe-based models to create static embeddings for the review text.

---

### **3. Amazon Review Datasets**
Located in:

- `all-beauty/All_Beauty.jsonl`
- `digital-music-5/Digital_Music_5.json`

Each dataset contains:
- `reviewText`
- `overall` (rating)
- `asin` (item ID)
- `reviewerID` (user ID)

These datasets are used to train recommendation and sentiment models.

---

### **4. Preprocessing Resources**
Located in:

- `digital-music-5/stopwords.txt`
- `digital-music-5/punctuations.txt`

Used to clean and preprocess review text before feeding it into the models.

---

# 🧠 Project Overview

This thesis investigates how different text embedding strategies affect the performance of downstream recommendation tasks.

Two embedding families are evaluated:

1. **GloVe (static embeddings)**  
2. **MiniLM (contextual embeddings)**

Two architectures are compared:

1. **BERT-tiny text classification architecture**  
2. **DeepCoNN user–item interaction architecture**

This yields **four experiments**, each represented by one notebook.

---

#  How to Reproduce the Experiments

### **1. Clone this repository**
```bash
git clone https://github.com/tinomuzo99/Research-Assignment.git
cd Research-Assignment
