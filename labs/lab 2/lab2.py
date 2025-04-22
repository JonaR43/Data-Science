import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset (you'll need to provide the actual path to your data)
# For demonstration, I'll assume the path is 'heart.csv'
data = pd.read_csv('heart.csv')

# Question 2: Create a correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Heart Disease Data')
plt.show()

# Question 4: Check for null values in each feature
null_values = data.isnull().sum()
print("Null values in each feature:")
print(null_values)

# Question 5: Show histogram of each factor
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.columns):
    plt.subplot(4, 4, i+1)
    data[column].hist(bins=20)
    plt.title(column)
plt.tight_layout()
plt.show()

# Question 6: Split the data into X and y
features = ['age', 'sex', 'cp', 'thalach', 'slope', 'restecg']
X = data[features]
y = data['target']

# Question 7: Plot K Neighbors Classifier Scores for different K values
k_range = range(1, 51)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Value of K')
plt.ylabel('Cross-validation accuracy')
plt.title('K Neighbors Classifier Scores for Different K Values')
plt.grid(True)
plt.show()

# Find K with highest cross-validation score
best_k = k_range[k_scores.index(max(k_scores))]
print(f"The K value with the highest cross-validation score is: {best_k}")

# Question 8: Find the Average cross validation score for 11 neighbors
knn_11 = KNeighborsClassifier(n_neighbors=11)
scores_11 = cross_val_score(knn_11, X, y, cv=5, scoring='accuracy')
avg_score_11 = scores_11.mean()
print(f"Average cross-validation score for 11 neighbors: {avg_score_11:.4f}")

# Question 9: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Train the classifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

# Predict y using the classifier
y_pred = knn.predict(X_test)

# Question 10: Find accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Question 11: Predict for the first real-life example
example1 = np.array([50, 1, 3, 222, 0, 2]).reshape(1, -1)
prediction1 = knn.predict(example1)
print(f"Prediction for the first real-life example: {prediction1[0]}")
print(f"This person {'has' if prediction1[0] == 1 else 'does not have'} a chance of developing heart disease.")

# Question 12: Predict for the second real-life example
example2 = np.array([63, 0, 1, 100, 2, 0]).reshape(1, -1)
prediction2 = knn.predict(example2)
print(f"Prediction for the second real-life example: {prediction2[0]}")
print(f"This person {'has' if prediction2[0] == 1 else 'does not have'} a chance of developing heart disease.")