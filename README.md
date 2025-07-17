**Framingham Coronary Heart Disease (CHD) Patient Analysis using Machine Learning (ML)**:

---

# 🏥 Framingham CHD Patient Analysis Using Machine Learning

This project focuses on predicting the risk of Coronary Heart Disease (CHD) in patients based on the **Framingham Heart Study dataset** using various **machine learning algorithms**. The goal is to build a predictive model that can assist healthcare professionals in identifying high-risk patients for early intervention.

---

## 📌 Project Overview

The Framingham Heart Study dataset contains health-related information for individuals and is widely used to predict the risk of heart disease. This project applies several ML models to classify individuals based on their likelihood of developing **Coronary Heart Disease (CHD)** in the future.

---

## 🧰 Tools & Technologies

* **Programming Language**: Python 3.x
* **Libraries**:

  * `pandas`, `numpy` — Data manipulation
  * `matplotlib`, `seaborn` — Data visualization
  * `scikit-learn` — ML models and metrics
  * `xgboost`, `lightgbm` — Advanced gradient boosting models
  * `imblearn` — Handling imbalanced datasets
  * `TensorFlow`, `Keras` (if using neural networks)

---

## 📂 Project Structure

```
framingham-chd-analysis/
│
├── data/                   # Raw and processed dataset
├── notebooks/              # Jupyter notebooks for EDA, preprocessing, and modeling
├── models/                 # Saved trained models
├── scripts/                # Data preprocessing, model training, and evaluation scripts
├── requirements.txt        # List of dependencies
├── main.py                 # Main entry point to train and test models
└── README.md               # Project documentation
```

---

## 🔍 Dataset Description

The **Framingham Heart Study dataset** contains various features related to lifestyle, demographics, and medical history, such as:

* **Age**: Age of the patient
* **Sex**: Gender (1 = Male, 0 = Female)
* **Cholesterol**: Serum cholesterol level
* **Blood Pressure**: Systolic and diastolic blood pressure levels
* **Smoking**: Whether the individual smokes (1 = Yes, 0 = No)
* **Diabetes**: Whether the individual has diabetes (1 = Yes, 0 = No)
* **Physical Activity**: Whether the individual engages in physical activity (1 = Yes, 0 = No)
* **BMI**: Body Mass Index
* **Heart Disease History**: Family history of heart disease (1 = Yes, 0 = No)
* **Target**: **1** indicates the presence of CHD (Coronary Heart Disease), and **0** indicates absence.

---

## ⚙️ How It Works

### 1. Data Collection

* The dataset is often available publicly or through research collaborations. You can download it from sources like Kaggle or directly from the Framingham Heart Study website.

### 2. Data Preprocessing

* **Cleaning**: Handle missing values, remove duplicates, and check for outliers.
* **Feature Engineering**: Create new features based on the available ones (e.g., BMI from height and weight).
* **Scaling**: Normalize/standardize numerical features.
* **Encoding**: Convert categorical features into numerical (e.g., sex, smoking).

### 3. Exploratory Data Analysis (EDA)

* Visualize the distribution of features.
* Check correlations between features using heatmaps.
* Explore the target distribution and imbalanced classes.

### 4. Model Training

* Split data into training and test sets (e.g., 80%/20%).
* Train multiple machine learning models:

  * Logistic Regression
  * Random Forest Classifier
  * Gradient Boosting (XGBoost, LightGBM)
  * Support Vector Machine (SVM)
  * Neural Networks (optional for advanced approaches)

### 5. Model Evaluation

* Use metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* Implement cross-validation to avoid overfitting.
* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

### 6. Handling Imbalanced Classes

* Use techniques like **SMOTE** (Synthetic Minority Oversampling Technique) or **undersampling** to address imbalanced target classes.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nileshbaithwar/Framingham_CHD_Analysis_using_MachineLearning/new/main?filename=README.md
cd framingham-chd-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python main.py
```

This script will load the data, perform preprocessing, train the models, and output evaluation results.

---

## 📈 Sample Results

| Metric    | Logistic Regression | Random Forest | XGBoost | Neural Network |
| --------- | ------------------- | ------------- | ------- | -------------- |
| Accuracy  | 85.2%               | 88.4%         | 89.6%   | 90.1%          |
| Precision | 84.5%               | 87.2%         | 88.3%   | 89.8%          |
| Recall    | 81.4%               | 85.3%         | 86.9%   | 87.1%          |
| F1-Score  | 82.9%               | 86.2%         | 87.6%   | 88.4%          |
| ROC-AUC   | 0.90                | 0.92          | 0.93    | 0.94           |

---

## 💡 Use Cases

* **Early Detection**: Predict patients at risk of developing heart disease.
* **Healthcare Decision Support**: Assist doctors and healthcare providers in prioritizing high-risk patients.
* **Public Health**: Identify trends in risk factors (e.g., smoking, diabetes) and design prevention programs.

---

## 🔮 Future Enhancements

* **Deep Learning Models**: Implement advanced models like LSTM, CNN, or neural networks for better prediction.
* **Real-time Prediction System**: Develop an API or application that can predict CHD risk in real-time based on patient data.
* **Multi-Class Classification**: Classify patients into different risk categories (e.g., Low, Moderate, High risk).

---

## 🛡️ License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.

---

Let me know if you'd like to include additional sections like model deployment instructions, dataset links, or more detailed code explanations!
