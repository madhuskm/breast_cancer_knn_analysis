![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

# 🩺 Breast Cancer Classifier using K-Nearest Neighbors (KNN)

This project demonstrates how to use the **K-Nearest Neighbors (KNN)** algorithm to classify breast cancer tumors as **malignant** or **benign** using the **Breast Cancer Wisconsin dataset** from `scikit-learn`.

The script includes **data loading**, **splitting**, **model training**, **hyperparameter tuning (k=1–100)**, and **visualization of accuracy** vs **k-values**.

---

## 🚀 Features

- 📦 Loads the Breast Cancer dataset from `sklearn.datasets`
- ✂️ Splits data into training and validation sets (80/20)
- 🧠 Trains a `KNeighborsClassifier` with `k` ranging from 1 to 100
- 📈 Evaluates and visualizes validation accuracy
- 🎯 Helps identify the optimal `k` for best model performance

---

## 📂 Project Structure

▶️ How to Run

Clone this repository:

git clone https://github.com/<your-username>/breast_cancer_knn_analysis.git
cd breast_cancer_knn_analysis


Install dependencies:
pip install -r requirements.txt


Run the script:
python breast_cancer_knn_analysis.py


View Output:
Console: Accuracy for k=1 to 100

Plot: Accuracy vs k visualization
📊 Sample Output
k = 1, Accuracy = 0.9123
k = 2, Accuracy = 0.9035
...
k = 7, Accuracy = 0.9561


The plot shows how accuracy varies with the number of neighbors, helping you pick the best k.

📘 Learnings
How to load and inspect datasets in scikit-learn
How to split data using train_test_split
How to train and evaluate a KNN model
How to visualize results with matplotlib

🧑‍💻 Author

Madhusudhan Sanjeev Kumar M.
AI Product Consultant • Data & Learning Technologist
📧 madhu@curieq.com

🌐 (https://github.com/madhuskm) | (https://www.linkedin.com/in/madhuskm/)
