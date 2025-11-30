# 🚗 OLA Driver Churn Prediction | Machine Learning Dashboard

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**OLA Driver Churn Prediction** is an end-to-end Machine Learning project designed to predict driver attrition. By analyzing driver demographics, tenure, performance, and income data, this application identifies at-risk drivers and provides actionable insights to improve retention strategies.

The project utilizes **Ensemble Learning** techniques (Random Forest, Bagging, XGBoost, Gradient Boosting) to handle class imbalance and maximize prediction accuracy. The results are presented in a professional, interactive **Streamlit Dashboard**.

---
## 🎬 Demo
- **Streamlit Profile** - https://share.streamlit.io/user/ratnesh-181998
- **Project Demo** - https://ola-driver-churn-prediction-machine-learning-mmntzrjjgxbadbxd4.streamlit.app/
---

## 🚀 Key Features
- **Interactive Dashboard**: Built with Streamlit, featuring a modern UI with dark mode and gradients.
- **Comprehensive EDA**: Visualizations for churn distribution, demographic analysis, and correlation heatmaps.
- **Advanced Preprocessing**: KNN Imputation, Feature Engineering (Rating/Income trends), and One-Hot Encoding.
- **Ensemble Modeling**: Implementation and comparison of Random Forest, Bagging, XGBoost, and Gradient Boosting.
- **Model Evaluation**: ROC-AUC curves, Precision-Recall curves, Confusion Matrices, and Feature Importance plots.
- **Business Insights**: Actionable recommendations based on data-driven findings.

---

## �️ Streamlit UI Walkthrough
The application is organized into intuitive tabs for a seamless user experience:

### 1. 📊 Data Overview
- **Key Metrics**: Displays total records, unique drivers, feature count, and time period.
- **Raw Data**: View the first 10 rows of the dataset.
- **Data Types**: Summary of column types and non-null counts.

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Distributions**: Visualizations for Gender, Education, Age, Income, and City.
- **Missing Values**: Heatmap and bar chart to identify data gaps.
- **Correlation**: Heatmap showing relationships between numerical features.

### 3. 📋 Case Study
- **Problem Statement**: Detailed explanation of the business challenge.
- **Churn Analysis**: Charts showing churn rates by Gender, Education, Rating, and Grade.
- **Concepts Tested**: Overview of Ensemble Learning and handling imbalanced data.

### 4. 🛠️ Preprocessing
- **Pipeline Steps**: Visual guide to data cleaning, encoding, and scaling.
- **Missing Data Map**: Heatmap visualizing the `LastWorkingDate` (churn indicator).

### 5. ⚙️ Features
- **Feature Engineering**: Analysis of derived features like `Quarterly_Rating_Increase` and `Income_Increase`.
- **Categorical Analysis**: Interactive bar charts comparing feature categories against churn.
- **Importance**: List of top features driving the model's predictions.

### 6. 🤖 Models
- **Model Training**: Code snippets and configuration for Random Forest, Bagging, XGBoost, and Gradient Boosting.
- **Learning Curves**: Visualization of training vs. validation performance.
- **Precision-Recall**: Curve demonstrating the trade-off for the best model.

### 7. 📊 Evaluation
- **Performance Metrics**: Comparative table of Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **ROC Curves**: Comparison of ROC curves for all models.
- **Confusion Matrices**: Heatmaps showing true positives, false positives, etc.

### 8. 💡 Insights
- **Key Findings**: Bullet points summarizing critical discoveries (e.g., 2018-2019 cohort risk).
- **Churn Trends**: Visual analysis of churn over time/cohorts.
- **Recommendations**: Strategic business actions to reduce attrition.

### 9. ❓ Questionnaire
- **Interactive Q&A**: Guided questions to explore the analysis findings.
- **Quiz**: Test your knowledge on the key drivers of churn.

### 10. 📚 Complete Analysis
- **Full Walkthrough**: Comprehensive, code-rich explanation of the entire project pipeline, from raw data to final model.

---

## �🛠️ Tech Stack
- **Language**: Python
- **Frontend**: Streamlit
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Category Encoders
- **Tools**: VS Code, Git

---

## 📂 Project Structure
```
OLA-Driver-Churn-Prediction/
├── app.py                   # Main Streamlit application
├── ola_analysis.py          # Core analysis and model training code
├── ola_driver_scaler.csv    # Dataset
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── .gitignore               # Git ignore file
└── logs/                    # Application logs
```

---

## 📊 Model Performance
We compared multiple models to find the best predictor for driver churn.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** | **89.1%** | **0.929** | **0.912** | **0.920** | **0.945** |
| Bagging (DT) | 88.0% | 0.939 | 0.876 | 0.906 | 0.935 |
| XGBoost | 87.0% | 0.884 | 0.923 | 0.900 | 0.930 |
| Random Forest | 86.8% | 0.928 | 0.866 | 0.890 | 0.920 |

**Winner:** Gradient Boosting Classifier provided the best balance of Precision and Recall with the highest ROC-AUC score.

---

## 💡 Key Business Insights
1.  **Cohort Risk**: Drivers who joined in **2018-2019** have significantly higher churn rates compared to newer joiners.
2.  **Rating Impact**: A quarterly rating of **1** is a critical warning sign, with churn rates nearing 70%.
3.  **Income Stagnation**: Drivers who did not receive an income increase during their tenure are highly likely to leave.
4.  **Education**: Drivers with lower education levels (10+, 12+) require more support and upskilling opportunities.

---

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ratnesh-181998/OLA-Driver-Churn-Prediction-Machine-Learning.git
    cd OLA-Driver-Churn-Prediction-Machine-Learning
    ```

2.  **Create a Virtual Environment (Optional)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🤝 Contact
**RATNESH SINGH**

- 📧 **Email**: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 **LinkedIn**: [https://www.linkedin.com/in/ratneshkumar1998/](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 **GitHub**: [https://github.com/Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 **Phone**: +91-947XXXXX46

### Project Links
- 🌐 Live Demo: [Streamlit](https://ola-driver-churn-prediction-machine-learning-mmntzrjjgxbadbxd4.streamlit.app/)
- 📖 Documentation: [GitHub Wiki](https://github.com/Ratnesh-181998/OLA-Driver-Churn-Prediction-Machine-Learning/wiki)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/OLA-Driver-Churn-Prediction-Machine-Learning/issues)

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
