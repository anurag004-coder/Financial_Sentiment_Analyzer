# Context-Aware Financial Sentiment Analyzer

An end-to-end Machine Learning web application designed to parse and classify specialized financial text for hidden market sentiment. Standard NLP models often misinterpret financial terminology (e.g., viewing phrases like "slashing costs" or "lowering expenses" as negative when they represent positive outcomes for investors). This application utilizes a specialized transformer pipeline to contextualize business jargon and evaluate whether market statements are inherently **Positive**, **Negative**, or **Neutral**.

Live Link: "https://financialsentimentanalyzer04.streamlit.app/"

---

## 🚀 Key Features

### 1. Single Sentence Analysis
* **Real-Time Inference:** Users can input any custom financial or corporate sentence into a text field.
* **Context Interpretation:** Leverages a domain-optimized model to evaluate complex financial variables, reporting both the classified sentiment and a precise model confidence score.

![Single Sentence Analysis UI]

### 2. Batch Processing & Built-In Evaluation
* **Automated Dataset Ingestion:** Automatically reads and parses a repository-hosted dataset (`financial_phrasebank.csv`) using specific text encoding handles (`ISO-8859-1`) to avoid structural parsing bugs.
* **Dynamic Range Ingestion:** Features an interactive slider allowing users to adjust batch sizes dynamically from 1 to 50 rows.
* **Visual Data Presentation:** Runs batch evaluations seamlessly on a virtualized CPU tier, outputting an interactive table complete with color-coded sentiment indicators for scannable analysis.

![Batch Processing UI]

---

## 🛠️ Tech Stack & Architecture

* **Frontend Framework:** Streamlit (Web Application Layout & Widgets)
* **Core Language & Tools:** Python, Pandas (Data Manipulation), System OS
* **Deep Learning Framework:** PyTorch (CPU-optimized compilation)
* **NLP Pipeline:** Hugging Face Transformers (`pipeline`, `AutoModelForSequenceClassification`)
* **Underlying Model:** `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`

---

## 📦 Project Directory Structure

```text
├── app.py                     # Main Streamlit web application source code
├── financial_phrasebank.csv   # Target domain dataset with historical statements
└── requirements.txt           # Dependency mapping for environment replication
