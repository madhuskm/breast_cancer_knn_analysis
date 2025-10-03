"""
Breast Cancer Classifier using K-Nearest Neighbors (KNN)
-------------------------------------------------------
This script demonstrates how to:
1. Load and explore the Breast Cancer dataset from sklearn.
2. Split the data into training and validation sets.
3. Train a KNN classifier across multiple k-values.
4. Evaluate and visualize model accuracy.
"""

# 📦 Import required libraries
from sklearn.datasets import load_breast_cancer        # Dataset
from sklearn.model_selection import train_test_split    # Train/test split
from sklearn.neighbors import KNeighborsClassifier      # KNN model
import matplotlib.pyplot as plt                         # Visualization

# 🎯 Load dataset
breast_cancer_data = load_breast_cancer()

# 🔍 Explore dataset
print("First data point:\n", breast_cancer_data.data[0])
print("\nFeature names:\n", breast_cancer_data.feature_names)
print("\nTarget labels (0 = malignant, 1 = benign):\n", breast_cancer_data.target)
print("\nTarget names:\n", breast_cancer_data.target_names)

# ✂️ Split dataset into training and validation sets (80/20 split)
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=80  # Controls reproducibility
)

print(f"\nTraining samples: {len(training_data)}")
print(f"Validation samples: {len(validation_data)}")

# 📈 Evaluate model performance for k = 1 → 100
accuracies = []  # Store accuracy scores
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracy = classifier.score(validation_data, validation_labels)
    accuracies.append(accuracy)
    print(f"k = {k}, Accuracy = {accuracy:.4f}")

# 📊 Prepare k values for plotting
k_list = range(1, 101)

# 🧭 Visualize accuracy vs. k
plt.figure(figsize=(10, 6))
plt.plot(k_list, accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy (KNN)')
plt.grid(True)
plt.show()
