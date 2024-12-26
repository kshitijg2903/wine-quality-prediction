import matplotlib.pyplot as plt
import numpy as np

def plot_classification_results(results):
    labels = list(results.keys())
    accuracy = [results[model]['Accuracy'] for model in labels]
    f1_scores = [results[model]['F1 Score'] for model in labels]
    
    x = np.arange(len(labels))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')

    ax.set_xlabel('Classification Models')
    ax.set_title('Classification Model Performance (Accuracy and F1 Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.ylim(0, 1)  # Set limits between 0 and 1 for accuracy and F1
    plt.show()

def plot_regression_results(results):
    labels = list(results.keys())
    mae = [results[model]['Mean Absolute Error'] for model in labels]
    r2_scores = [results[model]['R2 Score'] for model in labels]
    
    x = np.arange(len(labels))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, mae, width, label='Mean Absolute Error')
    rects2 = ax.bar(x + width/2, r2_scores, width, label='R2 Score')

    ax.set_xlabel('Regression Models')
    ax.set_title('Regression Model Performance (MAE and R² Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.ylim(0, max(mae) * 1.1)  # Adjust the Y-limit for visibility
    plt.show()
