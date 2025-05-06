# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import gradio as gr
import pickle

# Initialize MinMaxScaler for feature scaling and a dictionary to store trained models
scaler = MinMaxScaler()
trained_models = {}

# Function to load dataset and remove unnecessary or missing data
def load_and_clean_data(filepath):
    data = pd.read_csv(filepath).drop(columns='id')  # Drop the 'id' column
    data.dropna(inplace=True)  # Remove rows with missing values
    return data

# Function to encode categorical variables into numerical values using predefined mappings
def encode_data(data):
    mapping = {
        'gender': {'Male': 0, 'Female': 1, 'Other': 2},
        'ever_married': {'Yes': 0, 'No': 1},
        'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
        'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
        'Residence_type': {'Urban': 0, 'Rural': 1}
    }
    return data.replace(mapping)

# Function to scale numerical features using MinMaxScaler
def scale_features(X):
    global scaler
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Function to display a confusion matrix heatmap for model predictions
def plot_confusion_matrix(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='', cmap="Greens", xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Function to train a model, make predictions, and evaluate using classification metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    t1 = datetime.now()  # Start time
    model.fit(X_train, y_train)  # Train the model
    t2 = datetime.now()  # End time
    duration = round((t2 - t1).total_seconds(), 3)  # Training duration
    y_pred = model.predict(X_test)  # Make predictions
    score = round(model.score(X_test, y_test), 3)  # Accuracy score
    plot_confusion_matrix(y_test, y_pred)  # Visualize results
    print(metrics.classification_report(y_test, y_pred))  # Print detailed metrics
    return score, duration

# Function to perform hyperparameter tuning using GridSearchCV, then evaluate the best model
def model_pipeline(estimator, param_grid, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(estimator, param_grid, cv=10).fit(X_train, y_train)  # Cross-validated grid search
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_}")
    model = estimator.set_params(**grid.best_params_)  # Apply best parameters to model
    return evaluate_model(model, X_train, X_test, y_train, y_test), model


# Main function to execute the training and evaluation pipeline
def main():
    warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
    sns.set_style("darkgrid")  # Set Seaborn plot style
    pd.set_option('display.max_columns', None)  # Show all columns in dataframes

    # Load and preprocess the dataset
    data = load_and_clean_data('healthcare-dataset-stroke-data.csv')
    data_encoded = encode_data(data)
    
    # Split features and target, then scale features
    X = data_encoded.drop(columns='stroke')
    y = data_encoded['stroke']
    X = scale_features(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Define models and corresponding hyperparameter grids
    models = [
        ("RandomForestClassifier", RandomForestClassifier(), {
            'n_estimators': [50, 100, 250, 500],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2']
        }),
        ("LogisticRegression", LogisticRegression(), {
            'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            'class_weight': ['balanced'],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        }),
        ("SVC", SVC(), {
            'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            'gamma': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
        }),
        ("DecisionTreeClassifier", DecisionTreeClassifier(), {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': list(np.arange(4, 30, 1))
        }),
        ("KNeighborsClassifier", KNeighborsClassifier(), {
            'n_neighbors': list(np.arange(3, 20, 2)),
            'p': [1, 2, 3, 4]
        })
    ]

    # Train and evaluate each model, storing the results and trained models
    global trained_models
    results = []
    for name, model, params in models:
        print(f"\n--- {name} ---")
        (score, duration), trained_model = model_pipeline(model, params, X_train, y_train, X_test, y_test)
        trained_models[name] = trained_model
        results.append((name, score, duration))

    # Create a dataframe to show performance metrics of all models
    result_df = pd.DataFrame(results, columns=['Algorithm', 'Score', 'Delta_t'])
    print(result_df)

    # Plot bar charts for model scores and training times
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=result_df, x='Algorithm', y='Score', ax=ax[0]).bar_label(ax[0].containers[0], fmt='%.3f')
    ax[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=result_df, x='Algorithm', y='Delta_t', ax=ax[1]).bar_label(ax[1].containers[0], fmt='%.3f')
    ax[1].tick_params(axis='x', rotation=45)
    plt.show()

    # Save the trained models and scaler to files for later use
    print("SAVING MODELS AND SCALER VARIABLES")
    with open("models.pickle", mode='wb') as buffer_write:
        pickle.dump(trained_models, buffer_write)

    with open("scaler.pickle", mode='wb') as buffer_write:
        pickle.dump(scaler, buffer_write)

# Entry point to run the main function
if __name__ == "__main__":
    main()
