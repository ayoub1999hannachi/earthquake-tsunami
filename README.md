*

---

```markdown
# 🌋 Earthquake–Tsunami Prediction Project

## 🧭 Overview

This project focuses on predicting whether an **earthquake** event is likely to cause a **tsunami** based on seismic and geographical features.  
It combines **data preprocessing**, **exploratory data analysis (EDA)**, and **machine learning modeling** into a reproducible workflow with professional MLOps practices using **Git**, **Pytest**, and **GitHub Actions** for CI/CD.

---

## 🎯 Objectives

- Clean and preprocess raw earthquake data.  
- Perform **exploratory data analysis (EDA)** with visual insights and statistical summaries.  
- Develop and evaluate a **predictive model** for tsunami occurrence.  
- Ensure **code quality** through unit tests and continuous integration.  
- Provide a **reproducible workflow** for data-to-model experimentation.

---

## 📊 Dataset Source

The dataset used in this project comes from the [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/), containing global earthquake events with attributes such as:

| Feature | Description |
|----------|--------------|
| `latitude`, `longitude` | Geographic coordinates |
| `depth` | Depth of the earthquake (km) |
| `mag` | Magnitude of the event |
| `sig` | Significance value |
| `mmi`, `cdi` | Intensity and felt reports |
| `tsunami` | Target variable: 1 if tsunami occurred, 0 otherwise |

Raw data is stored in:  
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

1. **Data Preprocessing (`src/data_preprocessing.py`)**

   * Loads and cleans dataset
   * Handles missing values
   * Splits features and target

2. **Exploratory Data Analysis (`src/run_eda.py`)**

   * Generates statistical summaries and correlation heatmaps
   * Creates boxplots, histograms, and pairplots
   * Saves plots to `reports/`

3. **Model Training (`src/train_model.py`)**

   * Trains a Random Forest Classifier
   * Evaluates accuracy, precision, recall, and F1-score
   * Saves trained model in `models/`

4. **Testing (`tests/`)**

   * Validates data preprocessing and model pipeline with **Pytest**

5. **Continuous Integration (`.github/workflows/ci.yml`)**

   * Automatically runs unit tests on every commit

---

## 📁 Folder Structure

```
earthquake-tsunami-project/
│
├── data/
│   ├── raw/                 # Original dataset
│  
│
├── models/                  # Trained models (.pkl)
├── notebooks/               # Analysis & report notebooks
├── reports/                 # EDA results and plots
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── run_eda.py
│ 
│
├── tests/
│   ├── test_data.py
│   └── test_model.py
│
├── Makefile                 # Simplified workflow commands
├── requirements.txt
├── README.md
└── .github/workflows/ci.yml
```

---

## 🧰 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ayoub1999hannachi/earthquake-tsunami.git
cd earthquake-tsunami-project
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # on Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Available Commands (via Makefile)

| Command      | Description                             |
| ------------ | --------------------------------------- |
| `make data`  | Generate or load the dataset            |
| `make eda`   | Run enhanced exploratory analysis       |
| `make train` | Train and evaluate the predictive model |
| `make test`  | Run all unit tests                      |
| `make clean` | Remove cached files and reports         |

---

## 📈 Results Summary

| Metric                  | Value                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Accuracy**            | 0.85                                                                                                                           |
| **Precision (tsunami)** | 0.00                                                                                                                           |
| **Recall (tsunami)**    | 0.00                                                                                                                           |
| **F1-score**            | 0.00                                                                                                                           |
| **Observation**         | The model performs well on non-tsunami cases, but further tuning (e.g., SMOTE, class weighting) is needed for class imbalance. |

---

## 🚀 Continuous Integration

A **GitHub Actions** workflow automatically:

* Installs dependencies
* Runs tests (`pytest`)
* Ensures code reliability before merging to main branches

Workflow file:

```
.github/workflows/ci.yml
```

---

## 📚 Future Improvements

* Address class imbalance using oversampling or anomaly detection.
* Experiment with gradient boosting models (XGBoost, LightGBM).
* Add hyperparameter optimization and feature importance analysis.
* Extend the dataset to regional time-series modeling.

---

## 👨‍💻 Author

**Ayoub Hannachi**
📍 Tunisia
💼 [GitHub Profile](https://github.com/ayoub1999hannachi)
📧 Contact: *available upon request*

---

## 🏁 License

This project is released under the **MIT License** — free to use, modify, and distribute with attribution.

---

```

---


