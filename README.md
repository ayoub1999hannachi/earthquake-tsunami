

---

```markdown
# 🌋 Earthquake–Tsunami Prediction Project

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/ayoub1999hannachi/earthquake-tsunami?style=social)](https://github.com/ayoub1999hannachi/earthquake-tsunami/stargazers)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build](https://github.com/ayoub1999hannachi/earthquake-tsunami/actions/workflows/ci.yml/badge.svg)](https://github.com/ayoub1999hannachi/earthquake-tsunami/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](#)

---

## 🧭 Project Overview

Predict the likelihood of a **tsunami following an earthquake** using seismic and geographical features.  
This project integrates **data preprocessing**, **EDA**, and **machine learning modeling** with **MLOps best practices** including **CI/CD**, **unit testing**, and **reproducibility**.

---

## 🎯 Objectives

- Preprocess and clean global earthquake datasets  
- Conduct **EDA** with statistical summaries and visualizations  
- Build and evaluate a **predictive model** for tsunami occurrence  
- Maintain **high code quality** via **unit tests** and **CI/CD**  
- Provide a **reproducible workflow** from raw data to actionable insights

---

## 📊 Dataset

Data sourced from the [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/) with the following features:

| Feature         | Description                             |
|-----------------|-----------------------------------------|
| `latitude`, `longitude` | Geographic coordinates          |
| `depth`         | Depth of the earthquake (km)            |
| `mag`           | Magnitude of the earthquake             |
| `sig`           | Significance value of the event        |
| `mmi`, `cdi`    | Intensity and felt reports              |
| `tsunami`       | Target variable: 1 if tsunami occurred, 0 otherwise |

**Raw data location:**  
```

data/raw/earthquake_data_tsunami.csv

````

---

## ⚙️ Project Workflow

```mermaid
graph LR
A[Raw Data] --> B[Data Preprocessing]
B --> C[Exploratory Data Analysis]
C --> D[Model Training & Evaluation]
D --> E[Reports & Insights]
E --> F[Continuous Integration & Testing]
````

---

### 1️⃣ Data Preprocessing (`src/data_preprocessing.py`)

* Loads and cleans dataset
* Handles missing values
* Splits features and target

### 2️⃣ Exploratory Data Analysis (`src/run_eda.py`)

* Generates descriptive statistics & correlation heatmaps
* Creates boxplots, histograms, and pairplots
* Saves visualizations to `reports/`

### 3️⃣ Model Training (`src/train_model.py`)

* Trains a **Random Forest Classifier**
* Evaluates metrics: accuracy, precision, recall, F1-score
* Saves trained models to `models/`

### 4️⃣ Testing (`tests/`)

* Validates preprocessing and model pipeline using **Pytest**

### 5️⃣ Continuous Integration (`.github/workflows/ci.yml`)

* Runs unit tests automatically on each commit
* Ensures code quality before merging

---

## 📁 Folder Structure

```
earthquake-tsunami-project/
│
├── data/
│   └── raw/                # Original dataset
│
├── models/                 # Trained models (.pkl)
├── notebooks/              # Analysis & report notebooks
├── reports/                # EDA results and plots
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── run_eda.py
│
├── tests/
│   ├── test_data.py
│   └── test_model.py
│
├── Makefile                # Workflow commands
├── requirements.txt
├── README.md
└── .github/workflows/ci.yml
```

---

## 🧰 Installation & Usage

```bash
git clone https://github.com/ayoub1999hannachi/earthquake-tsunami.git
cd earthquake-tsunami-project
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## 🧪 Makefile Commands

| Command      | Description                               |
| ------------ | ----------------------------------------- |
| `make data`  | Load or generate the dataset              |
| `make eda`   | Run full exploratory data analysis        |
| `make train` | Train and evaluate predictive model       |
| `make test`  | Run all unit tests                        |
| `make clean` | Remove cached files and generated reports |

---

## 📈 Results Summary

| Metric                  | Value                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Accuracy**            | 0.85                                                                                                       |
| **Precision (tsunami)** | 0.00                                                                                                       |
| **Recall (tsunami)**    | 0.00                                                                                                       |
| **F1-score**            | 0.00                                                                                                       |
| **Observation**         | Model performs well on non-tsunami events; class imbalance needs attention (e.g., SMOTE, class weighting). |

---

## 🚀 Continuous Integration

**GitHub Actions** workflow:

* Installs dependencies
* Runs all **unit tests** automatically
* Validates code reliability before merging

Workflow path:

```
.github/workflows/ci.yml
```

---

## 📚 Future Improvements

* Address class imbalance (**SMOTE, undersampling, anomaly detection**)
* Experiment with **gradient boosting models** (XGBoost, LightGBM)
* Add **hyperparameter tuning** and **feature importance analysis**
* Extend to **regional time-series modeling**

---

## 👨‍💻 Author

**Ayoub Hannachi**
📍 Tunisia
💼 [GitHub Profile](https://github.com/ayoub1999hannachi)
📧 Contact: *available upon request*

---

## 🏁 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

```


Do you want me to do that too?
```
