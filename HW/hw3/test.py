# Homework 4 (Neural Networks)
# COSC 3337 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Part 1: California Housing Dataset (Regression)
print("="*80)
print("Part 1: California Housing Dataset (Regression)")
print("="*80)

# Load the California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Print basic statistics of the data
print("Basic Statistics of California Housing Dataset:")
print(X.describe())

# Print general information about the data
print("\nGeneral Information about California Housing Dataset:")
print(X.info())

# Check for missing values
print("\nMissing values in California Housing Dataset:")
print(X.isnull().sum())

# Answer: There are no missing values in this dataset as shown by the output above where all columns have 0 null values.

# Part 2: Visualization
print("\n" + "="*80)
print("Part 2: Visualization")
print("="*80)

# Histogram of median house value
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, edgecolor='black')
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Answer: The histogram shows that most house values are concentrated between 0 and 5, with fewer houses at the higher end
# of the price range. The distribution is right-skewed, which is common for housing prices.

# Scatter plot of Population vs. House Value
plt.figure(figsize=(10, 6))
plt.scatter(X['Population'], y, alpha=0.5)
plt.title('Population vs. House Value')
plt.xlabel('Population')
plt.ylabel('Median House Value ($100,000)')
plt.grid(True, alpha=0.3)
plt.show()

# Answer: There doesn't appear to be a strong linear relationship between population and house value.
# Most high-population areas don't necessarily have higher house values.

# Scatter plot of Longitude vs. Latitude, colored by house value
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X['Longitude'], X['Latitude'], c=y, cmap='viridis', 
                      alpha=0.6, s=10, edgecolors='none')
plt.colorbar(scatter, label='Median House Value ($100,000)')
plt.title('Geographic Distribution of House Values in California')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.show()

# Answer: This plot shows the geographical distribution of houses in California. We can see clusters of higher-valued 
# properties (yellow/green) in certain regions, likely corresponding to coastal areas and major cities like San Francisco 
# and Los Angeles. The lower-valued properties (dark blue/purple) are more scattered throughout the state.

# Heatmap of the data
plt.figure(figsize=(12, 8))
correlation_matrix = pd.concat([X, pd.Series(y, name='HouseValue')], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of California Housing Data')
plt.tight_layout()
plt.show()

# Answer: The heatmap shows that 'MedInc' (median income) has the strongest positive correlation with house value.
# 'Latitude' and 'Longitude' also show moderate correlations, which confirms our observation from the geographic plot.
# Other variables like 'AveRooms' show weaker correlations with house value.

# Creating scatter plots for each feature vs. house value
fig, axs = plt.subplots(3, 2, figsize=(18, 16))
axs = axs.flatten()

# 1. MedInc vs. house value
axs[0].scatter(X['MedInc'], y, alpha=0.5)
axs[0].set_title('MedInc vs. House Value')
axs[0].set_xlabel('Median Income')
axs[0].set_ylabel('House Value')
axs[0].grid(True, alpha=0.3)

# 2. HouseAge vs. house value
axs[1].scatter(X['HouseAge'], y, alpha=0.5)
axs[1].set_title('HouseAge vs. House Value')
axs[1].set_xlabel('House Age')
axs[1].set_ylabel('House Value')
axs[1].grid(True, alpha=0.3)

# 3. AveRooms vs. house value
axs[2].scatter(X['AveRooms'], y, alpha=0.5)
axs[2].set_title('AveRooms vs. House Value')
axs[2].set_xlabel('Average Rooms')
axs[2].set_ylabel('House Value')
axs[2].grid(True, alpha=0.3)

# 4. AveBedrms vs. house value
axs[3].scatter(X['AveBedrms'], y, alpha=0.5)
axs[3].set_title('AveBedrms vs. House Value')
axs[3].set_xlabel('Average Bedrooms')
axs[3].set_ylabel('House Value')
axs[3].grid(True, alpha=0.3)

# 5. AveOccup vs. house value
axs[4].scatter(X['AveOccup'], y, alpha=0.5)
axs[4].set_title('AveOccup vs. House Value')
axs[4].set_xlabel('Average Occupancy')
axs[4].set_ylabel('House Value')
axs[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Answer: From these scatter plots:
# - MedInc shows the strongest positive correlation with house value
# - HouseAge shows a weak positive correlation
# - AveRooms shows a moderate positive correlation, but with more variance
# - AveBedrms doesn't show a strong pattern with house value
# - AveOccup shows most values clustered at lower occupancy levels with a few outliers

# Histogram for HouseAge
plt.figure(figsize=(10, 6))
plt.hist(X['HouseAge'], bins=30, edgecolor='black')
plt.title('Distribution of House Age')
plt.xlabel('House Age (years)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Answer: The histogram for HouseAge shows several peaks, suggesting that housing development in California 
# occurred in waves. This might reflect periods of housing development booms in different decades.

# Part 3: Model Creation and Evaluation (Regression)
print("\n" + "="*80)
print("Part 3: Model Creation and Evaluation (Regression)")
print("="*80)

# 1. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

# 3. Create a neural network for regression
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(130, 64, 32, 16),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=True
)

# Fit the model
mlp_regressor.fit(X_train, y_train)

# 4. Make predictions
predictions = mlp_regressor.predict(X_test)

# 5. Print the R^2 score
r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.4f}")

# Answer: The model's R² score indicates how well the model explains the variance in the house values. 
# A score close to 1 would mean excellent predictive power.

# Experiment with different hidden layer sizes
print("\nExperimenting with different hidden layer sizes:")

hidden_layers = [
    (64, 32),
    (100, 50, 25),
    (130, 64, 32, 16),  # Original
    (256, 128, 64, 32, 16)
]

for layers in hidden_layers:
    model = MLPRegressor(
        hidden_layer_sizes=layers,
        activation='relu',
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    print(f"Layers {layers}: R² = {score:.4f}")

# Use GridSearchCV to find the best parameters
print("\nUsing GridSearchCV to find the best parameters:")

param_grid = {
    'hidden_layer_sizes': [(100, 50), (130, 64, 32, 16), (200, 100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPRegressor(max_iter=300, random_state=42),
    param_grid,
    cv=3,
    scoring='r2'
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.4f}")

# Use the best parameters found by GridSearchCV
best_mlp = MLPRegressor(
    **grid_search.best_params_,
    max_iter=300,
    random_state=42
)
best_mlp.fit(X_train, y_train)
best_predictions = best_mlp.predict(X_test)
best_r2 = r2_score(y_test, best_predictions)
print(f"R² score with best parameters on test set: {best_r2:.4f}")

# Remove outliers and see if that improves performance
print("\nRemoving outliers:")

# Identify outliers using IQR
def remove_outliers(X, y):
    df = pd.DataFrame(X)
    df['target'] = y
    
    # Calculate IQR for each feature
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    outlier_mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
    
    return df.loc[outlier_mask].drop('target', axis=1).values, df.loc[outlier_mask, 'target'].values

X_no_outliers, y_no_outliers = remove_outliers(X_scaled, y)

print(f"Original data size: {X_scaled.shape[0]}")
print(f"Data size after removing outliers: {X_no_outliers.shape[0]}")

# Split the filtered data
X_train_no_out, X_test_no_out, y_train_no_out, y_test_no_out = train_test_split(
    X_no_outliers, y_no_outliers, test_size=0.30, random_state=42
)

# Train model on data without outliers
no_outlier_model = MLPRegressor(
    **grid_search.best_params_,
    max_iter=300,
    random_state=42
)
no_outlier_model.fit(X_train_no_out, y_train_no_out)
no_outlier_pred = no_outlier_model.predict(X_test_no_out)
no_outlier_r2 = r2_score(y_test_no_out, no_outlier_pred)
print(f"R² score after removing outliers: {no_outlier_r2:.4f}")

# Best setting summary
print("\nBest Working Setting for Regression:")
print(f"Hidden layer sizes: {grid_search.best_params_['hidden_layer_sizes']}")
print(f"Activation function: {grid_search.best_params_['activation']}")
print(f"Alpha: {grid_search.best_params_['alpha']}")
print(f"With outlier removal: {'Yes' if no_outlier_r2 > best_r2 else 'No'}")
print(f"Best R² score: {max(best_r2, no_outlier_r2):.4f}")

# Part 4: Iris Dataset (Classification)
print("\n" + "="*80)
print("Part 4: Iris Dataset (Classification)")
print("="*80)

# Load the Iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

# Add species column temporarily for visualization
X_iris_viz = X_iris.copy()
X_iris_viz['species'] = pd.Categorical.from_codes(y_iris, iris.target_names)

# Print basic statistics of the data
print("Basic Statistics of Iris Dataset:")
print(X_iris.describe())

# Print general information about the data
print("\nGeneral Information about Iris Dataset:")
print(X_iris.info())

# Check for missing values
print("\nMissing values in Iris Dataset:")
print(X_iris.isnull().sum())

# Answer: There are no missing values in the Iris dataset as shown by the output above where all columns have 0 null values.

# Create a pairplot of the iris data
plt.figure(figsize=(12, 10))
sns.pairplot(X_iris_viz, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Answer: The pairplot shows clear separation between the species, especially for the setosa species which is well-separated
# from the other two. Versicolor and virginica show some overlap. Petal length and petal width seem to be the most
# discriminative features for separating the species.

# Create boxplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Boxplot of species vs. sepal length
sns.boxplot(x='species', y='sepal length (cm)', data=X_iris_viz, ax=axes[0, 0])
axes[0, 0].set_title('Species vs. Sepal Length')
axes[0, 0].grid(True, alpha=0.3)

# Answer: Setosa has smaller sepal length, while versicolor and virginica have similar but distinguishable sepal lengths.
# Virginica generally has the longest sepals.

# Boxplot of species vs. sepal width
sns.boxplot(x='species', y='sepal width (cm)', data=X_iris_viz, ax=axes[0, 1])
axes[0, 1].set_title('Species vs. Sepal Width')
axes[0, 1].grid(True, alpha=0.3)

# Answer: Setosa has wider sepals, while versicolor and virginica have similar but narrower sepal widths.
# Versicolor generally has the narrowest sepals.

# Boxplot of species vs. petal length
sns.boxplot(x='species', y='petal length (cm)', data=X_iris_viz, ax=axes[1, 0])
axes[1, 0].set_title('Species vs. Petal Length')
axes[1, 0].grid(True, alpha=0.3)

# Answer: There's a clear distinction in petal length across all three species. Setosa has the shortest petals,
# versicolor has medium-length petals, and virginica has the longest petals. There's minimal overlap between species.

# Boxplot of species vs. petal width
sns.boxplot(x='species', y='petal width (cm)', data=X_iris_viz, ax=axes[1, 1])
axes[1, 1].set_title('Species vs. Petal Width')
axes[1, 1].grid(True, alpha=0.3)

# Answer: Similar to petal length, petal width shows clear separation between the species. Setosa has the narrowest petals,
# versicolor has medium-width petals, and virginica has the widest petals. There's some overlap between versicolor and virginica.

plt.tight_layout()
plt.show()

# Create a scatterplot of sepal length vs. sepal width
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_iris['sepal length (cm)'], X_iris['sepal width (cm)'], 
                     c=y_iris, cmap='viridis', alpha=0.8, s=100)
plt.title('Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)
plt.show()

# Answer: This scatterplot shows that setosa is well-separated from the other two species based on sepal measurements,
# while versicolor and virginica show some overlap. Setosa tends to have wider but shorter sepals.

# Part 5: Model Creation and Evaluation (Classification)
print("\n" + "="*80)
print("Part 5: Model Creation and Evaluation (Classification)")
print("="*80)

# Scale the data
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# Split the data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris_scaled, y_iris, test_size=0.30, random_state=42
)

# Create a neural network for classification
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)

# Fit the model
mlp_classifier.fit(X_train_iris, y_train_iris)

# Make predictions
predictions_iris = mlp_classifier.predict(X_test_iris)

# Print the accuracy
accuracy = accuracy_score(y_test_iris, predictions_iris)
print(f"Accuracy: {accuracy:.4f}")

# Answer: The model achieves high accuracy on the Iris dataset, which is not surprising given the clear separation
# between species shown in our exploratory data analysis.

# Experiment with different hidden layer sizes
print("\nExperimenting with different hidden layer sizes:")

hidden_layers_iris = [
    (32, 16),
    (64, 32, 16),
    (128, 64, 32),
    (256, 128, 64, 32)  # Original
]

for layers in hidden_layers_iris:
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        max_iter=300,
        random_state=42
    )
    model.fit(X_train_iris, y_train_iris)
    pred = model.predict(X_test_iris)
    score = accuracy_score(y_test_iris, pred)
    print(f"Layers {layers}: Accuracy = {score:.4f}")

# Use GridSearchCV to find the best parameters
print("\nUsing GridSearchCV to find the best parameters:")

param_grid_iris = {
    'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search_iris = GridSearchCV(
    MLPClassifier(max_iter=300, random_state=42),
    param_grid_iris,
    cv=3,
    scoring='accuracy'
)
grid_search_iris.fit(X_train_iris, y_train_iris)

print(f"Best parameters: {grid_search_iris.best_params_}")
print(f"Best accuracy score: {grid_search_iris.best_score_:.4f}")

# Use the best parameters found by GridSearchCV
best_mlp_iris = MLPClassifier(
    **grid_search_iris.best_params_,
    max_iter=300,
    random_state=42
)
best_mlp_iris.fit(X_train_iris, y_train_iris)
best_predictions_iris = best_mlp_iris.predict(X_test_iris)
best_accuracy = accuracy_score(y_test_iris, best_predictions_iris)
print(f"Accuracy score with best parameters on test set: {best_accuracy:.4f}")

# Create a confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_iris, best_predictions_iris)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Iris Classification')
plt.grid(False)
plt.show()

# Best setting summary
print("\nBest Working Setting for Classification:")
print(f"Hidden layer sizes: {grid_search_iris.best_params_['hidden_layer_sizes']}")
print(f"Activation function: {grid_search_iris.best_params_['activation']}")
print(f"Alpha: {grid_search_iris.best_params_['alpha']}")
print(f"Best accuracy score: {best_accuracy:.4f}")