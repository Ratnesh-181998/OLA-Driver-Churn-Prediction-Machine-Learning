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

<img width="2879" height="1623" alt="image" src="https://github.com/user-attachments/assets/7bcc9491-0a83-4437-b8bf-1ec58d2e4adf" />
<img width="2869" height="1580" alt="image" src="https://github.com/user-attachments/assets/5c04be20-63d0-43ff-a7cd-7185ccf76b66" />

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Distributions**: Visualizations for Gender, Education, Age, Income, and City.
- **Missing Values**: Heatmap and bar chart to identify data gaps.
- **Correlation**: Heatmap showing relationships between numerical features.
<img width="2869" height="1623" alt="image" src="https://github.com/user-attachments/assets/28db0db6-15ff-42c3-859c-741e62f4b80d" />
<img width="2369" height="1315" alt="image" src="https://github.com/user-attachments/assets/e3cf259a-30aa-42f5-955d-2674f55dd1f1" />
<img width="2408" height="1472" alt="image" src="https://github.com/user-attachments/assets/759e88eb-aab4-4e25-8261-d706de207601" />
<img width="2432" height="1086" alt="image" src="https://github.com/user-attachments/assets/cee8a27f-ba69-4a25-8708-e4c628bef981" />
<img width="2421" height="1440" alt="image" src="https://github.com/user-attachments/assets/e0664fb4-2fe1-4538-bd6f-2f5abc86c841" />
<img width="2454" height="1149" alt="image" src="https://github.com/user-attachments/assets/aa9b9634-9b68-44dd-8ac7-40c01444dcc1" />

### 3. 📋 Case Study
- **Problem Statement**: Detailed explanation of the business challenge.
- **Churn Analysis**: Charts showing churn rates by Gender, Education, Rating, and Grade.
- **Concepts Tested**: Overview of Ensemble Learning and handling imbalanced data.
<img width="2425" height="1444" alt="image" src="https://github.com/user-attachments/assets/bb220d09-025c-44b1-94af-55e739d8f90a" />
<img width="2392" height="1317" alt="image" src="https://github.com/user-attachments/assets/a9ef14fa-e897-4e0a-8434-4abf948b3f29" />
<img width="2391" height="1161" alt="image" src="https://github.com/user-attachments/assets/9516b6b2-bf4d-4a09-8fb8-574f935d1090" />
<img width="2391" height="1151" alt="image" src="https://github.com/user-attachments/assets/4089b815-1cbe-4109-958f-ce7aeb894a00" />
<img width="2410" height="1395" alt="image" src="https://github.com/user-attachments/assets/2dbfc551-df17-4df4-8f64-8f7c5f0ea2bd" />

### 4. 🛠️ Preprocessing
- **Pipeline Steps**: Visual guide to data cleaning, encoding, and scaling.
- **Missing Data Map**: Heatmap visualizing the `LastWorkingDate` (churn indicator).
<img width="2828" height="1505" alt="image" src="https://github.com/user-attachments/assets/3df70b96-b8fe-4016-bce9-dee9273e0353" />
<img width="2465" height="1408" alt="image" src="https://github.com/user-attachments/assets/3322f92d-96d2-40c9-a717-1ef9dc940bae" />

### 5. ⚙️ Features
- **Feature Engineering**: Analysis of derived features like `Quarterly_Rating_Increase` and `Income_Increase`.
- **Categorical Analysis**: Interactive bar charts comparing feature categories against churn.
- **Importance**: List of top features driving the model's predictions.
<img width="2335" height="1369" alt="image" src="https://github.com/user-attachments/assets/e0f3a44c-63c4-4f8c-a090-f52fcda3e997" />
<img width="2252" height="1410" alt="image" src="https://github.com/user-attachments/assets/815fc0e3-c2e8-4747-b0b8-a22a2139eba4" />
<img width="2354" height="1161" alt="image" src="https://github.com/user-attachments/assets/eaf5ecb9-7d03-4fc9-8629-c741d5786f07" />
<img width="2331" height="1369" alt="image" src="https://github.com/user-attachments/assets/00a1d08e-60e7-4955-ae7d-265cd6191960" />
<img width="2307" height="1386" alt="image" src="https://github.com/user-attachments/assets/e9f6540e-ec00-4c11-801b-563907990683" />

### 6. 🤖 Models
- **Model Training**: Code snippets and configuration for Random Forest, Bagging, XGBoost, and Gradient Boosting.
- **Learning Curves**: Visualization of training vs. validation performance.
- **Precision-Recall**: Curve demonstrating the trade-off for the best model.
<img width="2834" height="1521" alt="image" src="https://github.com/user-attachments/assets/92d453e3-2ac6-40f2-bdfc-3d2b178c23fd" />
<img width="2309" height="1381" alt="image" src="https://github.com/user-attachments/assets/8fe7c15f-808a-4c47-8cd5-7c3211e9317c" />
<img width="2345" height="1414" alt="image" src="https://github.com/user-attachments/assets/81f1faf8-0d56-4797-b7cd-822e17b7393c" />
<img width="2350" height="1377" alt="image" src="https://github.com/user-attachments/assets/57219a7c-10ba-4523-b3b4-3288f08eed7e" />
<img width="2865" height="1465" alt="image" src="https://github.com/user-attachments/assets/517f2467-77a8-46a2-a560-a51dcdc651df" />

### 7. 📊 Evaluation
- **Performance Metrics**: Comparative table of Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **ROC Curves**: Comparison of ROC curves for all models.
- **Confusion Matrices**: Heatmaps showing true positives, false positives, etc.
<img width="2878" height="1515" alt="image" src="https://github.com/user-attachments/assets/0a4bb00b-1d88-4174-b7dd-41f693742dc6" />
<img width="2356" height="1069" alt="image" src="https://github.com/user-attachments/assets/40b1f17e-d434-4b57-8ef4-10e7a9667bc3" />
<img width="2336" height="1287" alt="image" src="https://github.com/user-attachments/assets/8df3473a-bf60-47eb-ba86-8e6e7023a3ee" />
<img width="2328" height="1359" alt="image" src="https://github.com/user-attachments/assets/26ecc9a6-2be5-495f-a7da-926096b8bd5a" />
<img width="2353" height="1423" alt="image" src="https://github.com/user-attachments/assets/7cd41a9d-aadb-4a8c-ad9e-b5af6c31eb73" />
<img width="2307" height="1395" alt="image" src="https://github.com/user-attachments/assets/15d848bc-372b-4afe-839c-aabed5927802" />

### 8. 💡 Insights
- **Key Findings**: Bullet points summarizing critical discoveries (e.g., 2018-2019 cohort risk).
- **Churn Trends**: Visual analysis of churn over time/cohorts.
- **Recommendations**: Strategic business actions to reduce attrition.
<img width="2327" height="1054" alt="image" src="https://github.com/user-attachments/assets/bfb71944-d980-4f8d-b058-75fbd59af24e" />
<img width="2320" height="1413" alt="image" src="https://github.com/user-attachments/assets/2532ebbb-9795-43db-8c6a-f9e987b0ecd4" />
<img width="2251" height="1344" alt="image" src="https://github.com/user-attachments/assets/cd5e508c-e829-45da-ba93-a4b60d890f2c" />

### 9. ❓ Questionnaire
- **Interactive Q&A**: Guided questions to explore the analysis findings.
- **Quiz**: Test your knowledge on the key drivers of churn.
<img width="2297" height="1290" alt="image" src="https://github.com/user-attachments/assets/eaab89f5-7690-48e2-bad7-9b290ed7cca3" />
<img width="2340" height="1328" alt="image" src="https://github.com/user-attachments/assets/f521aeeb-37f4-42ee-ad7b-026b84a58e70" />
<img width="2406" height="1405" alt="image" src="https://github.com/user-attachments/assets/bb2c52ec-98c5-406c-949a-a2409658803f" />
<img width="2866" height="1284" alt="image" src="https://github.com/user-attachments/assets/ce9f7471-c1dd-4437-9549-1a9827eb11f9" />

### 10. 📚 Complete Analysis
- **Full Walkthrough**: Comprehensive, code-rich explanation of the entire project pipeline, from raw data to final model.
<img width="2341" height="1377" alt="image" src="https://github.com/user-attachments/assets/bf3157a8-f248-4b2d-b7ed-c5e888a77fd1" />
<img width="2316" height="1417" alt="image" src="https://github.com/user-attachments/assets/0abd070d-b4d3-46ee-8ee6-f7257a3f418c" />
<img width="2313" height="1433" alt="image" src="https://github.com/user-attachments/assets/4669cec3-8c7d-4eda-a5f5-0dfb4a52445b" />
<img width="2269" height="1429" alt="image" src="https://github.com/user-attachments/assets/689b7db1-d020-4ac2-add6-13f11b3b2010" />
<img width="2219" height="1395" alt="image" src="https://github.com/user-attachments/assets/17d5f07f-f36a-4fcd-8481-3425f5a4b6a5" />
<img width="2204" height="1287" alt="image" src="https://github.com/user-attachments/assets/2500f584-2e00-46bb-b407-7a809fb3c799" />
<img width="2021" height="1438" alt="image" src="https://github.com/user-attachments/assets/59aa9c75-570d-4369-92c4-8a433ed8173d" />
<img width="2304" height="1367" alt="image" src="https://github.com/user-attachments/assets/fdf932c6-9f55-4d5b-aaf2-865da5936648" />

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

---


<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=24,20,12,6&height=3" width="100%">


## 📜 **License**

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

**Licensed under the MIT License** - Feel free to fork and build upon this innovation! 🚀

---

# 📞 **CONTACT & NETWORKING** 📞


### 💼 Professional Networks

[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ratneshkumar1998/)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ratnesh-181998)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/RatneshS16497)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-FF6B6B?style=for-the-badge&logo=google-chrome&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![Email](https://img.shields.io/badge/✉️_Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rattudacsit2021gate@gmail.com)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@rattudacsit2021gate)
[![Stack Overflow](https://img.shields.io/badge/Stack_Overflow-F58025?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/32068937/ratnesh-kumar)

### 🚀 AI/ML & Data Science
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/RattuDa98)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/rattuda)

### 💻 Competitive Programming (Including all coding plateform's 5000+ Problems/Questions solved )
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/Ratnesh_1998/)
[![HackerRank](https://img.shields.io/badge/HackerRank-00EA64?style=for-the-badge&logo=hackerrank&logoColor=black)](https://www.hackerrank.com/profile/rattudacsit20211)
[![CodeChef](https://img.shields.io/badge/CodeChef-5B4638?style=for-the-badge&logo=codechef&logoColor=white)](https://www.codechef.com/users/ratnesh_181998)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white)](https://codeforces.com/profile/Ratnesh_181998)
[![GeeksforGeeks](https://img.shields.io/badge/GeeksforGeeks-2F8D46?style=for-the-badge&logo=geeksforgeeks&logoColor=white)](https://www.geeksforgeeks.org/profile/ratnesh1998)
[![HackerEarth](https://img.shields.io/badge/HackerEarth-323754?style=for-the-badge&logo=hackerearth&logoColor=white)](https://www.hackerearth.com/@ratnesh138/)
[![InterviewBit](https://img.shields.io/badge/InterviewBit-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://www.interviewbit.com/profile/rattudacsit2021gate_d9a25bc44230/)


---

## 📊 **GitHub Stats & Metrics** 📊



![Profile Views](https://komarev.com/ghpvc/?username=Ratnesh-181998&color=blueviolet&style=for-the-badge&label=PROFILE+VIEWS)



<img 
  src="https://streak-stats.demolab.com?user=Ratnesh-181998&theme=radical&hide_border=true&background=0D1117&stroke=4ECDC4&ring=F38181&fire=FF6B6B&currStreakLabel=4ECDC4"
  alt="GitHub Streak Stats"
width="48%"/>




<img src="https://github-readme-activity-graph.vercel.app/graph?username=Ratnesh-181998&theme=react-dark&hide_border=true&bg_color=0D1117&color=4ECDC4&line=F38181&point=FF6B6B" width="48%" />

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&lines=Ratnesh+Kumar+Singh;Data+Scientist+%7C+AI%2FML+Engineer;4%2B+Years+Building+Production+AI+Systems" alt="Typing SVG" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=Built+with+passion+for+the+AI+Community+🚀;Innovating+the+Future+of+AI+%26+ML;MLOps+%7C+LLMOps+%7C+AIOps+%7C+GenAI+%7C+AgenticAI+Excellence" alt="Footer Typing SVG" />


<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%">


