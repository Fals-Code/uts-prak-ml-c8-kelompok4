# UTS Praktikum Machine Learning — Kelompok 4 C8

Project UTS Praktikum Machine Learning 2026 untuk **Soal D**.

Dataset: **IBM HR Analytics Employee Attrition & Performance**  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Target klasifikasi: `Attrition` (`Yes` / `No`).

## Alur

1. Preprocessing data
2. Split data 80% training dan 20% testing
3. Min-Max scaling
4. Feature selection:
   - Mutual Information
   - L1-based Feature Selection
5. Decision Tree
6. Evaluasi dengan Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score

## Struktur

```text
.
├── data/
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── modeling.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```

## Menjalankan Project

```bash
pip install -r requirements.txt
```

Download dataset dari Kaggle dan simpan sebagai:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Untuk menjalankan preprocessing:

```bash
python -m src.preprocessing
```

## Branch

- `falah/preprocessing` — dataset dan preprocessing
- `adlha/feature-selection` — split, Min-Max, dan feature selection
- `azzam/model-evaluation` — Decision Tree dan evaluasi
