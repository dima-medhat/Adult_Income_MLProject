# 💼 Adult Income Prediction

A Streamlit web application that predicts whether an individual's annual income is **greater than 50K** or **less than or equal to 50K** using supervised machine learning. The application allows users to upload a dataset, train multiple classification models, compare their performance, and make real-time predictions through an interactive interface.

---

## 🚀 Features

- Upload and explore the Adult Income dataset
- Interactive dataset preview and summary statistics
- Train machine learning models directly in the application
- Support for five classification algorithms
- Compare **Unbalanced** and **SMOTE** training
- Hyperparameter tuning using **RandomizedSearchCV**
- Model evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Precision-Recall AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance visualization (Tree-based models)
- Interactive income prediction with prediction probabilities

---

## 🤖 Machine Learning Models

- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Choose your GitHub repository.
5. Set the main file to `app.py`.
6. Deploy the application.

---

## 📌 Important Note

Models are trained during the active Streamlit session. Before training a model, upload the Adult Income dataset on the **Data Overview** page.

---

## 📂 Application Pages

- 🏠 Home
- 📄 Data Overview
- ⚙️ Model Training
- 📊 Performance Dashboard
- 🎯 Income Prediction

---

## 📷 Screenshots

Add screenshots of each application page here after deployment.

---

## 👤 Author

**Dima.M.AlShurafa**

Machine Learning & Data Analytics
