# 📰 News Bias Detection System

## 📌 Overview
The **News Bias Detection System** is an NLP-based machine learning project designed to automatically identify and analyze bias in news articles and online media content. With the growing reliance on digital news platforms and social media, biased reporting can significantly influence public opinion and societal discourse. This project aims to promote **media literacy** and **critical consumption of news** by providing an automated way to detect biased content.

The system leverages **TF-IDF feature extraction** combined with **machine learning classifiers** to detect different levels of bias such as *Highly Biased*, *Slightly Biased*, and *Neutral* reporting.

---

## 🎯 Objectives
- Automatically detect bias in news text
- Classify articles into multiple bias categories
- Analyze subtle linguistic and stylistic indicators of bias
- Provide reproducible and extensible ML pipelines

---

## 🧠 Dataset
The project uses a large annotated news bias dataset with the following relevant fields:

| Column | Description |
|------|-------------|
| `text` | News/article content |
| `label` | Bias label (*Highly Biased*, *Slightly Biased*, *Neutral*) |
| `dimension` | High-level bias category |
| `aspect` | Type of bias (e.g., framing, body-shaming) |
| `biased_words` | Annotated bias-triggering words |
| `sentiment` | Sentiment polarity |
| `toxic` | Toxicity indicator |
| `identity_mention` | Mentioned demographic identity |

📌 **Primary prediction target:** `label`

---

## 🔍 Models Implemented

### 1️⃣ Baseline Model — Word-level TF-IDF + Logistic Regression
- Word unigrams
- Logistic Regression (multiclass)
- Balanced class weights

**Performance (Test Set):**
- Accuracy: ~80%
- Strong performance on *Highly Biased* and *Neutral*
- Moderate confusion between *Neutral* and *Slightly Biased*

---

### 2️⃣ Improved Model — Word + Character TF-IDF + Logistic Regression
- **Word n-grams (1–2)**
- **Character n-grams (3–5)** to capture stylistic bias (punctuation, framing, emphasis)
- Logistic Regression (`solver="saga"`)

📈 **Improvement Observed:**
- Higher Macro-F1 score
- Better recall for *Slightly Biased* articles
- More robust detection of subtle bias patterns

---

## 📊 Evaluation Metrics
- Accuracy
- Precision / Recall / F1-score
- Macro F1 (important due to class imbalance)
- Confusion Matrix (normalized)

## 📚 Dataset & Citation

This project uses the **BEADs (Bias Evaluation Across Domains)** dataset for training and evaluation.

If you use this dataset, please cite the following paper:

@article{raza2024beads,
  title={BEADs: Bias Evaluation Across Domains},
  author={Raza, Shaina and Rahman, Mizanur and Zhang, Michael R},
  journal={arXiv preprint arXiv:2406.04220},
  year={2024}
}

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install pandas pyarrow scikit-learn joblib numpy matplotlib seaborn

🔬 Key Findings

Bias detection is significantly harder for Slightly Biased content due to subtle framing.

Character-level features improve detection of stylistic bias cues.

Hierarchical bias classification may further improve performance.




🔮 Future Work
Hierarchical classification (Biased vs Neutral → Bias intensity)
Transformer-based models (BERT / RoBERTa / DeBERTa)
Bias explanation using annotated biased_words
Probability calibration for confidence-aware predictions
Deployment as an API or web application

📚 Technologies Used
Python
scikit-learn
Pandas / NumPy
TF-IDF
Logistic Regression
Jupyter Notebook
Matplotlib / Seaborn

⭐ Acknowledgements

This project was developed to explore the intersection of Natural Language Processing, Machine Learning, and Media Bias Analysis, with the goal of promoting responsible and critical news consumption.