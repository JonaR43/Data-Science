# %% [markdown]
# # Lab 3: Entropy and Decision Tree
# 
# This notebook contains the solution for the lab on Entropy and Decision Trees.

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import math

# %% [markdown]
# ## Load the dataset
# Note: Replace this with your actual dataset. For now, creating a sample dataset for demonstration.

# %%
# Load or create the dataset
# For demonstration purposes, creating a sample dataset
# Replace this with your actual data file: data = pd.read_csv('astronaut_data.csv')
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    'age': np.random.randint(20, 50, 100),
    'likes_dogs': np.random.randint(0, 2, 100),
    'likes_gravity': np.random.randint(0, 2, 100),
    'going_to_be_an_astronaut': np.random.randint(0, 2, 100)
})

# %% [markdown]
# ## 1. Read the data first and look at the first 5 rows. Check if there are any missing values or not (3 points)

# %%
# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# %% [markdown]
# ## 2. Split the data using sklearn's train_test_split function (5 points)

# %%
# Separate features and target variable
X = data.drop('going_to_be_an_astronaut', axis=1)
y = data['going_to_be_an_astronaut']

# Split data with 20% test size and random_state=5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print("Shape of training and testing sets:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# ## 3. Use the gini criterion to fit the data to the training set (5 points)

# %%
# Create and train Decision Tree with Gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=5, max_depth=4)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)

# %% [markdown]
# ## 4. Evaluate the accuracy, precision and recall for the Gini model (10 points)

# %%
# Calculate performance metrics for Gini model
gini_accuracy = accuracy_score(y_test, y_pred_gini)
gini_precision = precision_score(y_test, y_pred_gini)
gini_recall = recall_score(y_test, y_pred_gini)

print("Gini Model Evaluation:")
print(f"Accuracy: {gini_accuracy:.4f}")
print(f"Precision: {gini_precision:.4f}")
print(f"Recall: {gini_recall:.4f}")

# %% [markdown]
# ### Explanation of Gini findings:
# 
# The Gini criterion model shows how well the model predicts astronaut outcomes:
# - Accuracy: Measures the proportion of correct predictions (both true positives and true negatives) out of total predictions.
# - Precision: Measures the proportion of true positive predictions out of all positive predictions. High precision means that when the model predicts someone will be an astronaut, it's usually correct.
# - Recall: Measures the proportion of true positive predictions out of all actual positives. High recall means the model is good at identifying most people who will become astronauts.
# 
# The values give us insights into the model's performance on unseen data. Higher values indicate better performance.

# %% [markdown]
# ## 5. Use the Entropy criterion to fit the data to the training set (2 points)

# %%
# Create and train Decision Tree with Entropy criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=5)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)

# %% [markdown]
# ## 6. Evaluate the accuracy, precision and recall for the Entropy model (10 points)

# %%
# Calculate performance metrics for Entropy model
entropy_accuracy = accuracy_score(y_test, y_pred_entropy)
entropy_precision = precision_score(y_test, y_pred_entropy)
entropy_recall = recall_score(y_test, y_pred_entropy)

print("Entropy Model Evaluation:")
print(f"Accuracy: {entropy_accuracy:.4f}")
print(f"Precision: {entropy_precision:.4f}")
print(f"Recall: {entropy_recall:.4f}")

# %% [markdown]
# ### Explanation of Entropy findings:
# 
# The Entropy criterion model uses a different approach to make splits in the decision tree:
# - Entropy measures the randomness or disorder in a dataset, with higher values indicating more uncertainty.
# - For decision trees, the algorithm calculates information gain (reduction in entropy) to determine where to make splits.
# - The performance metrics show how well this approach works for predicting astronaut outcomes compared to Gini.
# - Note that the Entropy model uses max_depth=5 while Gini uses max_depth=4, which may affect the comparison.

# %% [markdown]
# ## 7. Which criteria gives the highest accuracy, precision and recall? (5 points)

# %%
# Compare the models
print("Comparison of models:")
if entropy_accuracy > gini_accuracy:
    best_model = "Entropy"
    best_accuracy = entropy_accuracy
    print("Entropy has higher accuracy.")
else:
    best_model = "Gini"
    best_accuracy = gini_accuracy
    print("Gini has higher accuracy.")

if entropy_precision > gini_precision:
    print("Entropy has higher precision.")
else:
    print("Gini has higher precision.")

if entropy_recall > gini_recall:
    print("Entropy has higher recall.")
else:
    print("Gini has higher recall.")

# %% [markdown]
# ### Explanation:
# 
# The differences in performance metrics between Gini and Entropy can be attributed to:
# - How each criterion handles the split at nodes: Gini tends to favor larger partitions, while Entropy tends to create more balanced trees.
# - The complexity of the dataset: Sometimes one criterion might work better for certain data distributions.
# - The different max_depth values (4 for Gini vs 5 for Entropy): This might give the Entropy model more flexibility to fit the data.
# 
# The criterion that performs better on this specific dataset would be more suitable for the astronaut prediction task.

# %% [markdown]
# ## 8. Plot the tree for both the gini and entropy function (5 points)

# %%
# Plot both trees side by side
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plot_tree(dt_gini, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree with Gini Criterion", fontsize=14)

plt.subplot(1, 2, 2)
plot_tree(dt_entropy, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree with Entropy Criterion", fontsize=14)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Predict for an individual of age 33, likes dogs but doesn't love gravity (10 points)

# %%
# Create data for new individual
new_individual = pd.DataFrame({
    'age': [33],
    'likes_dogs': [1],
    'likes_gravity': [0]
})

# Use the model with the highest accuracy to predict
if best_model == "Entropy":
    prediction = dt_entropy.predict(new_individual)
else:
    prediction = dt_gini.predict(new_individual)

print("Prediction for individual with age 33, likes dogs but doesn't love gravity:")
if prediction[0] == 1:
    print("This individual is predicted to become an astronaut.")
else:
    print("This individual is predicted not to become an astronaut.")

# %% [markdown]
# ## 10. Find the best max_depth for entropy and Gini (10 points)

# %%
# Test various depths to find optimal value
max_depths = range(1, 22)
gini_accuracies = []
entropy_accuracies = []

for depth in max_depths:
    # Gini model
    dt_gini = DecisionTreeClassifier(criterion='gini', random_state=5, max_depth=depth)
    dt_gini.fit(X_train, y_train)
    y_pred_gini = dt_gini.predict(X_test)
    gini_accuracies.append(accuracy_score(y_test, y_pred_gini))
    
    # Entropy model
    dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=depth)
    dt_entropy.fit(X_train, y_train)
    y_pred_entropy = dt_entropy.predict(X_test)
    entropy_accuracies.append(accuracy_score(y_test, y_pred_entropy))

# Find best depths
best_gini_depth = max_depths[np.argmax(gini_accuracies)]
best_entropy_depth = max_depths[np.argmax(entropy_accuracies)]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(max_depths, gini_accuracies, marker='o', linestyle='-', color='blue', label='Gini')
plt.plot(max_depths, entropy_accuracies, marker='s', linestyle='--', color='red', label='Entropy')
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs Max Depth for Gini and Entropy Criteria', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

print(f"Best max_depth for Gini: {best_gini_depth}")
print(f"Best max_depth for Entropy: {best_entropy_depth}")

# %% [markdown]
# ### Explanation for different best depths:
# 
# If the best depths for Gini and Entropy are different, it's because:
# 
# - Gini and Entropy use different mathematical formulas to measure node impurity.
# - Entropy (uses logarithms) tends to penalize less pure nodes more severely than Gini.
# - This difference in impurity calculation can lead to different split decisions at each node.
# - These different splitting behaviors may require different tree depths to reach optimal predictive performance.
# - The dataset's specific characteristics might favor one criterion over the other at different depths.

# %% [markdown]
# ## 11. Calculate the root's entropy (10 points)

# %%
# Count the number of people who became astronauts and those who didn't
astronaut_counts = data['going_to_be_an_astronaut'].value_counts()
total_samples = len(data)

# Calculate probabilities
p_yes = astronaut_counts.get(1, 0) / total_samples
p_no = astronaut_counts.get(0, 0) / total_samples

# Calculate entropy
if p_yes == 0 or p_no == 0:
    root_entropy = 0
else:
    root_entropy = -p_yes * math.log2(p_yes) - p_no * math.log2(p_no)

print(f"Root Entropy: {root_entropy:.4f}")
print(f"Probability of becoming an astronaut (p_yes): {p_yes:.4f}")
print(f"Probability of not becoming an astronaut (p_no): {p_no:.4f}")

# %% [markdown]
# ## 12. Create pivot table for likes_dog vs going_to_be_an_astronaut (5 points)

# %%
# Create pivot table
pivot_table = pd.crosstab(data['likes_dogs'], data['going_to_be_an_astronaut'])
print("Pivot Table (likes_dogs vs going_to_be_an_astronaut):")
print(pivot_table)

# Add column and row totals for clarity
pivot_with_totals = pd.crosstab(
    data['likes_dogs'], 
    data['going_to_be_an_astronaut'], 
    margins=True, 
    margins_name='Total'
)
print("\nPivot Table with Totals:")
print(pivot_with_totals)

# %% [markdown]
# ## 13. Calculate entropy for liking dogs vs not liking dogs, then find the Information Gain (20 points)

# %%
# Get counts for each group
likes_dogs_counts = pivot_table.loc[1]
not_likes_dogs_counts = pivot_table.loc[0]

# Calculate total samples in each group
likes_dogs_total = likes_dogs_counts.sum()
not_likes_dogs_total = not_likes_dogs_counts.sum()

# Calculate probabilities
p_likes_dogs_yes = likes_dogs_counts.get(1, 0) / likes_dogs_total if likes_dogs_total > 0 else 0
p_likes_dogs_no = likes_dogs_counts.get(0, 0) / likes_dogs_total if likes_dogs_total > 0 else 0

p_not_likes_dogs_yes = not_likes_dogs_counts.get(1, 0) / not_likes_dogs_total if not_likes_dogs_total > 0 else 0
p_not_likes_dogs_no = not_likes_dogs_counts.get(0, 0) / not_likes_dogs_total if not_likes_dogs_total > 0 else 0

# Display the probabilities
print("Probabilities for each group:")
print(f"P(astronaut=Yes | likes_dogs=Yes): {p_likes_dogs_yes:.4f}")
print(f"P(astronaut=No | likes_dogs=Yes): {p_likes_dogs_no:.4f}")
print(f"P(astronaut=Yes | likes_dogs=No): {p_not_likes_dogs_yes:.4f}")
print(f"P(astronaut=No | likes_dogs=No): {p_not_likes_dogs_no:.4f}")

# Calculate entropy for each group
if p_likes_dogs_yes == 0 or p_likes_dogs_no == 0:
    entropy_likes_dogs = 0
else:
    entropy_likes_dogs = -p_likes_dogs_yes * math.log2(p_likes_dogs_yes) - p_likes_dogs_no * math.log2(p_likes_dogs_no)

if p_not_likes_dogs_yes == 0 or p_not_likes_dogs_no == 0:
    entropy_not_likes_dogs = 0
else:
    entropy_not_likes_dogs = -p_not_likes_dogs_yes * math.log2(p_not_likes_dogs_yes) - p_not_likes_dogs_no * math.log2(p_not_likes_dogs_no)

print(f"\nEntropy for likes_dogs=Yes: {entropy_likes_dogs:.4f}")
print(f"Entropy for likes_dogs=No: {entropy_not_likes_dogs:.4f}")

# Calculate weighted average entropy after split
weighted_entropy = (likes_dogs_total / total_samples) * entropy_likes_dogs + (not_likes_dogs_total / total_samples) * entropy_not_likes_dogs

# Calculate information gain
information_gain = root_entropy - weighted_entropy

print(f"\nRoot Entropy (before split): {root_entropy:.4f}")
print(f"Weighted Average Entropy (after split): {weighted_entropy:.4f}")
print(f"Information Gain: {information_gain:.4f}")

# %% [markdown]
# ### Explanation of findings:
# 
# Information gain measures how much uncertainty is reduced after splitting the data based on the 'likes_dogs' attribute:
# 
# - Root entropy represents the initial uncertainty about whether someone will become an astronaut.
# - After splitting based on 'likes_dogs', we calculate entropy for each subset (likes_dogs=Yes and likes_dogs=No).
# - The weighted average entropy represents the remaining uncertainty after the split.
# - Information gain = Root entropy - Weighted average entropy
# - A higher information gain indicates that 'likes_dogs' is a more informative feature for predicting astronaut outcomes.
# - If information gain is high, knowing whether someone likes dogs gives us significant information about whether they'll become an astronaut.
# - If information gain is low, the 'likes_dogs' feature doesn't help much with our prediction task.
# 
# This analysis helps us understand the predictive power of the 'likes_dogs' feature in our decision tree model.