# **AI-Based Sentiment Analysis on Social Media (DistilBERT + Logistic Regression)**

---

## **1. Project Overview**
This project focuses on **sentiment analysis of social media tweets** to detect depression-related content.  
It follows a **hybrid AI approach** where:

- **DistilBERT** is used as a **feature extractor** (no fine-tuning).
- **Logistic Regression** is used as the final classifier.

This approach achieves **99% accuracy** while being lightweight and efficient.

---

## **2. Why This Hybrid Approach?**
- **DistilBERT** captures deep contextual meaning of text by converting each tweet into **768-dimensional embeddings**.
- **Logistic Regression** is simple, interpretable, and works well with high-quality embeddings.
- Together, they provide **AI-powered NLP understanding** combined with **traditional ML accuracy**.

---

## **3. Workflow**
### **Step 1: Dataset Preparation**
- Load and clean tweets with labels (depression vs non-depression).

### **Step 2: Embedding Extraction**
- Use **DistilBERT `[CLS]` token embeddings** for each tweet.
- Processed in **batches on GPU (T4)** for faster computation.

### **Step 3: Classification**
- Train a **Logistic Regression** model on extracted embeddings.

### **Step 4: Evaluation**
- Compute metrics: **Accuracy, Precision, Recall, F1-score**.
- Generate **Confusion Matrix** and **metric bar plots**.

---

## **4. Key Results**
- **Accuracy:** 99%  
- **Precision (Depression):** 0.98  
- **Recall (Depression):** 0.96  
- **F1-score (Depression):** 0.97  

---

## **5. Key Insights**
- Transformer embeddings (DistilBERT) **dramatically improve text classification** even without fine-tuning.
- Traditional models like Logistic Regression **perform exceptionally well when combined with AI-based embeddings**.
- This approach is **fast, lightweight, and production-friendly**.

---

## **6. Visualizations**
- **Confusion Matrix** to show prediction distribution.
- **Classification Metrics Plot** (precision, recall, F1-score).

---

## **7. Installation**
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
