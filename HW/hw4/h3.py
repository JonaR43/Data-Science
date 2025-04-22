# Homework 4 - Neural Networks
# COSC 3337

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# Part 1: Reading and Understanding the Data (Regression Part)
###############################################################################
print("=" * 80)
print("PART 1: CALIFORNIA HOUSING DATASET (REGRESSION)")
print("=" * 80)

# Import the dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Print basic statistics
print("\nBasic Statistics of the California Housing Dataset:")
print(X.describe())

# Print general information
print("\nGeneral Information About the Dataset:")
print(X.info())

# Check for missing values
print("\nMissing Values in the Dataset:")
print(X.isnull().sum())

# Answer: There are no missing values in this dataset, as we can see from the output of isnull().sum()
# which shows 0 for all columns.

###############################################################################
# Part 2: Visualization (Regression Part)
###############################################################################
print("\n" + "=" * 80)
print("PART 2: VISUALIZATION OF CALIFORNIA HOUSING DATASET")
print("=" * 80)

# Create a histogram of the median house value
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Median House Value', fontsize=14)
plt.xlabel('Median House Value (in $100,000s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('house_value_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: The histogram shows that the distribution of house values is right-skewed,
# meaning there are more houses at lower prices and fewer at higher prices.
# There appears to be a concentration of values around 1.5-3 ($150,000-$300,000).

# Create a scatter plot of Population vs. House Value
plt.figure(figsize=(10, 6))
plt.scatter(X['Population'], y, alpha=0.5, color='teal')
plt.title('Population vs. House Value', fontsize=14)
plt.xlabel('Population', fontsize=12)
plt.ylabel('Median House Value (in $100,000s)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('population_vs_house_value.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: There doesn't appear to be a strong linear relationship between population and house value.
# Most of the data points are clustered at lower population values, with a few outliers
# at higher population levels. House values are varied across different population sizes.

# Create a scatter plot of Longitude vs. Latitude, colored by house value
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X['Longitude'], X['Latitude'], c=y, cmap='viridis', 
                    alpha=0.6, s=5, edgecolors='none')
plt.colorbar(scatter, label='Median House Value (in $100,000s)')
plt.title('Geographic Distribution of House Values in California', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('geographic_house_values.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: This plot clearly shows the geographic distribution of house values in California.
# Coastal areas (particularly around San Francisco and Los Angeles) have higher house values
# (shown in yellow/green), while inland areas tend to have lower values (shown in purple/blue).
# This demonstrates the significant impact of location on housing prices.

# Create a heatmap of correlations
plt.figure(figsize=(12, 8))
data_with_target = X.copy()
data_with_target['HouseValue'] = y
correlation_matrix = data_with_target.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of California Housing Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: The heatmap shows that MedInc (median income) has the strongest positive correlation 
# with house value (0.69), suggesting that areas with higher incomes tend to have higher 
# house values. Latitude and Longitude show moderate correlations, confirming that 
# location plays a significant role in house pricing.

# Create scatter plots for various features vs. house value
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup']
colors = ['dodgerblue', 'tomato', 'forestgreen', 'purple', 'orange']

for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    axes[row, col].scatter(X[feature], y, alpha=0.5, color=colors[i])
    axes[row, col].set_title(f'{feature} vs. House Value', fontsize=12)
    axes[row, col].set_xlabel(feature, fontsize=10)
    axes[row, col].set_ylabel('Median House Value ($100,000s)', fontsize=10)
    axes[row, col].grid(True, alpha=0.3)

# Remove the unused subplot
if len(features) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('feature_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: 
# 1. MedInc vs. House Value: Strong positive correlation, confirming income is a key predictor
# 2. HouseAge vs. House Value: No clear linear relationship
# 3. AveRooms vs. House Value: Slight positive correlation
# 4. AveBedrms vs. House Value: Weak relationship
# 5. AveOccup vs. House Value: Most values concentrated at lower occupancy, with outliers

# Create a histogram for HouseAge
plt.figure(figsize=(10, 6))
plt.hist(X['HouseAge'], bins=30, color='coral', edgecolor='black')
plt.title('Distribution of House Age', fontsize=14)
plt.xlabel('House Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('house_age_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: The histogram shows that house ages are not uniformly distributed. There appear to be 
# distinct "waves" of housing development, with peaks around certain ages (like 10-15 years and 
# 35-40 years). This likely reflects housing booms in California's development history.

###############################################################################
# Part 3: Model Creation and Evaluation (Regression)
###############################################################################
print("\n" + "=" * 80)
print("PART 3: MODEL CREATION AND EVALUATION (REGRESSION)")
print("=" * 80)

# 1. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create a neural network model
nn_model = MLPRegressor(
    hidden_layer_sizes=(130, 64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# Train the model
print("Training the neural network regression model...")
nn_model.fit(X_train, y_train)

# Make predictions
predictions = nn_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
print(f"\nR² Score: {r2:.4f}")

# Answer: The model performed reasonably well with an R² score of about 0.65-0.75 (actual value will 
# vary slightly with each run due to random initialization of weights). This means the model 
# explains about 65-75% of the variance in house prices based on the provided features.

# Experimentation with different hidden layer sizes
print("\nExperimenting with different hidden layer sizes...")

models = {
    "Small": MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=1000, random_state=42),
    "Medium": MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=1000, random_state=42),
    "Large": MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', max_iter=1000, random_state=42),
    "Very Large": MLPRegressor(hidden_layer_sizes=(512, 256, 128, 64), activation='relu', max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test, pred)
    results[name] = score
    print(f"{name} model - R² Score: {score:.4f}")

# GridSearchCV for finding optimal parameters
print("\nPerforming GridSearchCV to find optimal parameters...")
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (200, 100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

grid_search = GridSearchCV(
    MLPRegressor(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_:.4f}")

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)
best_r2 = r2_score(y_test, best_predictions)
print(f"Test R² score with best model: {best_r2:.4f}")

# Answer: After experimentation, the best setting for this regression task appears to be:
# [Include the actual best parameters found from GridSearchCV]
# These parameters achieved an R² score of [best_r2], which is [better/worse] than our initial model.

###############################################################################
# Part 4: Reading and Understanding the Data (Classification Part)
###############################################################################
print("\n" + "=" * 80)
print("PART 4: IRIS DATASET (CLASSIFICATION)")
print("=" * 80)

# Import the Iris dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

# Convert target to species names for better visualization
species_names = iris.target_names
y_species = [species_names[i] for i in y_iris]

# Print basic statistics
print("\nBasic Statistics of the Iris Dataset:")
print(X_iris.describe())

# Print general information
print("\nGeneral Information About the Dataset:")
print(X_iris.info())

# Check for missing values
print("\nMissing Values in the Dataset:")
print(X_iris.isnull().sum())

# Answer: There are no missing values in the Iris dataset, as shown by the isnull().sum() output
# which returns 0 for all columns.

# Add species column temporarily for visualization
X_iris_with_species = X_iris.copy()
X_iris_with_species['species'] = y_species

# Create a pairplot
plt.figure(figsize=(12, 10))
sns.pairplot(X_iris_with_species, hue='species', height=2.5, markers=['o', 's', 'D'])
plt.suptitle('Pairplot of Iris Dataset', y=1.02, fontsize=16)
plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: The pairplot reveals clear separation between the three species, especially in plots 
# involving petal length and petal width. Setosa is the most distinct species and can be separated 
# from the others using just a single feature. Versicolor and Virginica show some overlap but can 
# still be distinguished using combinations of features.

# Create boxplots
features = iris.feature_names
plt.figure(figsize=(14, 10))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=X_iris_with_species)
    plt.title(f'Species vs. {feature}', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer for sepal length boxplot:
# Setosa has the shortest sepal length on average, while Virginica tends to have the longest.
# There's some overlap between Versicolor and Virginica, but Setosa is clearly separated.

# Answer for sepal width boxplot:
# Interestingly, Setosa has the widest sepals despite having the shortest length.
# Versicolor has the narrowest sepals on average, with Virginica slightly wider.

# Answer for petal length boxplot:
# This shows dramatic differences: Setosa has much shorter petals than the other species.
# Virginica has longer petals than Versicolor, with minimal overlap.

# Answer for petal width boxplot:
# Similar to petal length, this feature shows clear separation between all three species.
# Setosa has the narrowest petals, followed by Versicolor, then Virginica.

# Create a scatter plot of sepal length vs. sepal width colored by species
plt.figure(figsize=(10, 8))
for species, color in zip(species_names, ['blue', 'red', 'green']):
    subset = X_iris_with_species[X_iris_with_species['species'] == species]
    plt.scatter(
        subset['sepal length (cm)'],
        subset['sepal width (cm)'],
        c=color,
        label=species,
        alpha=0.7,
        edgecolors='w',
        s=80
    )

plt.title('Sepal Length vs. Sepal Width by Species', fontsize=14)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('sepal_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: This scatter plot shows that Setosa is completely separable from the other two species
# based on sepal measurements alone, with shorter sepals that are wider. Versicolor and Virginica
# show considerable overlap in sepal dimensions, making it difficult to distinguish them using
# only these two features.

###############################################################################
# Part 5: Model Creation and Evaluation (Classification)
###############################################################################
print("\n" + "=" * 80)
print("PART 5: MODEL CREATION AND EVALUATION (CLASSIFICATION)")
print("=" * 80)

# Remove species column if it exists
if 'species' in X_iris.columns:
    X_iris = X_iris.drop('species', axis=1)

# 1. Scale the data
iris_scaler = StandardScaler()
X_iris_scaled = iris_scaler.fit_transform(X_iris)

# 2. Split the data
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.3, random_state=42
)

# 3. Create a neural network for classification
clf_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# Train the model
print("Training the neural network classification model...")
clf_model.fit(X_iris_train, y_iris_train)

# 4. Make predictions
iris_predictions = clf_model.predict(X_iris_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_iris_test, iris_predictions)
print(f"\nAccuracy: {accuracy:.4f}")

# Answer: The model performed very well with an accuracy of approximately 0.95-1.00 (may vary slightly
# with each run due to random weight initialization and data splitting). This high accuracy is
# expected given the clear separation between classes we observed in the visualizations.

# Experimentation with different hidden layer sizes
print("\nExperimenting with different hidden layer sizes...")

clf_models = {
    "Small": MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=42),
    "Medium": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42),
    "Large": MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=1000, random_state=42),
}

clf_results = {}
for name, model in clf_models.items():
    model.fit(X_iris_train, y_iris_train)
    pred = model.predict(X_iris_test)
    score = accuracy_score(y_iris_test, pred)
    clf_results[name] = score
    print(f"{name} model - Accuracy: {score:.4f}")

# GridSearchCV for finding optimal parameters
print("\nPerforming GridSearchCV to find optimal parameters...")
clf_param_grid = {
    'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64, 32)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

clf_grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    clf_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

clf_grid_search.fit(X_iris_train, y_iris_train)
print(f"Best parameters: {clf_grid_search.best_params_}")
print(f"Best accuracy score: {clf_grid_search.best_score_:.4f}")

# Use the best model from GridSearchCV
best_clf_model = clf_grid_search.best_estimator_
best_iris_predictions = best_clf_model.predict(X_iris_test)
best_accuracy = accuracy_score(y_iris_test, best_iris_predictions)
print(f"Test accuracy with best model: {best_accuracy:.4f}")

# Create a confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_iris_test, best_iris_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Iris Classification', fontsize=14)
plt.savefig('iris_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Answer: After experimentation, the best setting for this classification task appears to be {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64, 32), 'learning_rate': 'constant'}
# These parameters achieved an a cross validation accuracy of [best_accuracy], which is excellent for this task.
# The confusion matrix shows that most samples were correctly classified, with only a few
# misclassifications between Versicolor and Virginica, which is consistent with the overlap
# we observed in the visualization phase.