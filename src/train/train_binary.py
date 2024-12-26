import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_binary_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    print("Data Head:")
    print(data.head())
    if 'quality' not in data.columns:
        print("Error: 'quality' column not found in the dataset")
        return None
    X = data.drop('quality', axis=1)  
    y = data['quality']  
    print("\nFeatures (X) Head:")
    print(X.head())
    print("\nTarget (y) Head:")
    print(y.head())
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encode "lower quality" and "finest quality" as 0 and 1

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, le

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_perceptron(X_train, y_train):
    model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear') 
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, pos_label='finest quality')  # Assuming 'finest quality' is the positive class
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_pairplot(data, label_encoder):
    sns.pairplot(data, hue="quality", palette="coolwarm", markers=["o", "s"])
    plt.title("Feature Interaction Pairplot")
    plt.show()


if __name__ == "__main__":
    file_path = 'data/data_binary.csv'
    data = load_binary_data(file_path)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data)

    plot_pairplot(data, label_encoder)

    print("Logistic Regression:")
    logistic_model = train_logistic_regression(X_train, y_train)
    evaluate_model(logistic_model, X_test, y_test, label_encoder)
    
    print("\nPerceptron Model:")
    perceptron_model = train_perceptron(X_train, y_train)
    evaluate_model(perceptron_model, X_test, y_test, label_encoder)
    
    print("\nSVM Model:")
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, label_encoder)