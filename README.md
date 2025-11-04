# Stock Market Sentiment Analysis: Predicting VTI Price Movements from News Headlines
*Machine Learning | NLP | Financial Forecasting | XGBoost on GPU*

---

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?logo=xgboost&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch&logoColor=white)](#)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-all--MiniLM--L6--v2-green)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

This project **predicts daily price movements of the Vanguard Total Stock Market ETF (VTI)** using **financial news headlines** as the sole input. Leveraging **natural language processing (NLP)** and **gradient-boosted decision trees**, the model classifies whether VTI will **close higher or lower** than its opening price based on the sentiment and context of news headlines published that day.

> **Key Objective**: Build a binary classifier (`Up` vs `Down`) using only textual news data — no technical indicators, volume, or price history.

---

## Tech Stack & Tools

| Category             | Technology |
|----------------------|----------|
| **Language**         | Python 3.11 |
| **ML Framework**     | XGBoost (`XGBClassifier`) with GPU acceleration |
| **NLP**              | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Deep Learning**    | PyTorch (backend for embeddings) |
| **Data Processing**  | Pandas, NumPy |
| **Model Evaluation** | scikit-learn (`train_test_split`, `confusion_matrix`) |
| **Visualization**    | Matplotlib, Seaborn |

---

## Model Performance

The trained **XGBoost model achieved strong predictive performance** on unseen test data:

| Metric                 | Value |
|------------------------|-------|
| **Test Accuracy**      | **78.4%** |
| **True Positives (Up)**| 19,453 |
| **True Negatives (Down)** | 19,135 |
| **False Positives**    | 15,475 |
| **False Negatives**    | 15,157 |

### Confusion Matrix
![GPU Stock Prediction Performance](gpu_confusion_matrix2.png)

> *The model correctly classifies **38,588 out of 69,220** test samples — demonstrating robust generalization.*

---

## Project Architecture
