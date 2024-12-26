import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.load.data_loading import load_and_split_data
from src.models.models import (
    train_linear_regression, train_lasso_regression, train_ridge_regression, train_logistic_binary_regression,
    train_logistic_multi_regression, train_random_forest, train_gradient_boosting
)
from src.eval.eval_plotting import plot_classification_results, plot_regression_results  # Assuming these functions exist
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from src.might_delete.export import render_mpl_table

X_train, X_test, y_train, y_test = load_and_split_data('data/data_1.csv')
# X_train, X_test, y_train, y_test = load_and_split_data('data/data_1_augmented.csv')
# X_train, X_test, y_train, y_test = load_and_split_data('data/data_1_processed.csv')



models = {
    "Linear Regression": train_linear_regression(X_train, y_train),
    "Lasso Regression": train_lasso_regression(X_train, y_train),
    "Ridge Regression": train_ridge_regression(X_train, y_train),
    "Logistic Regression": train_logistic_multi_regression(X_train, y_train)[0],
    "Random Forest": train_random_forest(X_train, y_train),
    "Gradient Boosting": train_gradient_boosting(X_train, y_train)
}

regression_models = ["Linear Regression", "Lasso Regression", "Ridge Regression"]
classification_models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
columns = ['Model', 'Accuracy', 'F1 Score', 'MAE', 'R² Score']
results = []

classification_results = {}
regression_results = {}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    
    if model_name in regression_models:
        mae = round(mean_absolute_error(y_test, y_pred),4)
        r2 = round(r2_score(y_test, y_pred),4)
        regression_results[model_name] = {"Mean Absolute Error": mae, "R2 Score": r2}
        print(f'{model_name} - Mean Absolute Error: {mae:.4f}, R2 Score: {r2:.4f}')
        results.append([model_name, "-", "-", mae, r2])
    
    elif model_name in classification_models:
        accuracy = round(accuracy_score(y_test, y_pred),4)
        f1 = round(f1_score(y_test, y_pred, average='weighted'),4)
        classification_results[model_name] = {"Accuracy": accuracy, "F1 Score": f1}
        print(f'{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        results.append([model_name, accuracy, f1, "-", "-"])

df_results = pd.DataFrame(results, columns=columns)
print(df_results)   

plot_classification_results(classification_results)
plot_regression_results(regression_results)

render_mpl_table(df_results, header_columns=0, col_width=2.0)