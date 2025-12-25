# Cancer Cell Line Growth Property Prediction using Machine Learning

## 📌 Overview
This project applies machine learning techniques to analyze biological metadata from cancer cell lines and predict their growth behavior (Adherent vs Suspension). The objective is to demonstrate how supervised learning can be applied to heterogeneous biological datasets and how model interpretability can help extract meaningful biological insights.

---

## 🧬 Dataset
**Source:** `Cell_Lines_Details.xlsx`

The dataset contains structured metadata of cancer cell lines, including biological and experimental attributes such as tissue origin, disease classification, and cellular growth characteristics. This metadata is commonly used for cell line characterization in cancer research.

---

## 🧠 Methodology
- Cleaning and preprocessing of real biological metadata
- Encoding heterogeneous categorical features into numerical representations
- Training a supervised classification model using **XGBoost**
- Model evaluation using:
  - Accuracy
  - Precision, Recall, and F1-score
  - Confusion Matrix
- Feature importance analysis to interpret influential biological attributes

---

## 📊 Results
The trained model achieved high classification performance, demonstrating that cancer cell line metadata is highly predictive of growth behavior. Feature importance analysis highlighted key biological attributes influencing adherent versus suspension growth patterns.

---

## 📈 Visualizations
The following visual outputs are generated and saved in the `results/` folder:
- Confusion matrix
- Feature importance bar chart
- Class distribution plot

---

## 🛠️ Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**

---

## 📁 Project Structure
cell-line-growth-prediction/
│
├── data/
│ └── Cell_Lines_Details.xlsx
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│
├── results/
│ ├── metrics.txt
│ ├── confusion_matrix.png
│ ├── feature_importance.png
│ └── class_distribution.png
│
├── requirements.txt
└── README.md
## How to Run

### 1 Install dependencies
```bash
pip install -r requirements.txt
```
### 2 Preprocess the dataset
```bash
cd src
python preprocess.py
```
### 3 Train the model and generate results
```bash
python train.py
```
