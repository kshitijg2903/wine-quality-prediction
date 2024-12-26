import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.load.data_loading import load_data, load_and_split_data
from src.preprocess.scale import scale_data
from src.preprocess.missing_vals import handle_missing_values
from src.preprocess.encode_cat import encode_categorical
from src.models.dl_models import NN_Shallowest, NN_Shallow, NN_Dropout, NN_Deep, NN_BatchNorm

import wandb

sweep_config = {
    "method": "random",  # "grid" / "random" / "bayes"
    "metric": {"name": "val_accuracy", "goal": "maximize"},  
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0001, 0.00001]},  
        "hidden_size": {"values": [32, 64, 128]},
        "batch_size": {"values": [16, 32, 64]}, 
        "dropout": {"values": [0.3, 0.5]},  
        "epochs": {"value": 250} 
    }
}

sweep_id = wandb.sweep(sweep_config, project="Wine Quality Prediction")


# def train_regression():
#     # forget regression models for now


# def train_classification():

def train_NN(filepath="data/data_1.csv", NN_type="NN_Shallow"):

    with wandb.init() as run:
        config = wandb.config  # Access hyperparameters dynamically from WandB sweep

        if (filepath == "data/data_1.csv"):
            data = load_data(filepath)
            X = data.drop(columns=['Id', 'quality'])
            y = data['quality']
            X_scaled = scale_data(X, method="standard")

        elif (filepath == "data/data_1_processed.csv" or filepath == "data/data_1_augmented.csv"):
            data = load_data(filepath)
            X_scaled = data.drop(columns =['quality'])
            y = data['quality']

        else:
            raise NotImplementedError("Support for other datasets is not implemented yet.")
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42)

        y_train = y_train - y_train.min()
        y_val = y_val - y_val.min()
        y_test = y_test - y_test.min()

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False)

        if NN_type == "NN_Shallowest":
            model = NN_Shallowest(input_size=X_train.shape[1], hidden_size=config.hidden_size, output_size=len(y.unique()))
        elif NN_type == "NN_Shallow":
            model = NN_Shallow(input_size=X_train.shape[1], hidden_size=config.hidden_size, output_size=len(y.unique()))
        elif NN_type == "NN_Dropout":
            model = NN_Dropout(input_size=X_train.shape[1], hidden_size=config.hidden_size, output_size=len(y.unique()), dropout=config.dropout)
        elif NN_type == "NN_Deep":
            model = NN_Deep(input_size=X_train.shape[1], hidden_size=config.hidden_size, output_size=len(y.unique()))
        elif NN_type == "NN_BatchNorm":
            model = NN_BatchNorm(input_size=X_train.shape[1], hidden_size=config.hidden_size, output_size=len(y.unique()))
        else:
            raise ValueError(f"Unknown NN_type: {NN_type}")

        wandb.watch(model, log="all")

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                predictions = torch.argmax(outputs, axis=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct / total

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    predictions = torch.argmax(outputs, axis=1)
                    val_correct += (predictions == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })

            print(
                f"Epoch {epoch + 1}/{config.epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

        # torch.save(model.state_dict(), f"model_{NN_type}_lr_{config.learning_rate}_hs_{config.hidden_size}.pth")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, axis=1)
            y_pred = predictions.numpy()

            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

            test_accuracy = accuracy_score(y_test, y_pred)

            report = classification_report(y_test, y_pred, zero_division=1)
            cm = confusion_matrix(y_test, y_pred)

            print("Test Classification Report:")
            print(report)
            print("Test Confusion Matrix:")
            print(cm)
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # Log test metrics to WandB
            wandb.log({
                "test_accuracy": test_accuracy,
                "test_classification_report": report,
                "test_confusion_matrix": cm.tolist()
            })


if __name__ == "__main__":

    filepath = "data/data_1_processed.csv"

    # wandb.agent(sweep_id, function=lambda: train_NN(filepath=filepath, NN_type="NN_Shallowest"), count=10)
    # wandb.agent(sweep_id, function=lambda: train_NN(filepath=filepath, NN_type="NN_Dropout"), count=10)
    # wandb.agent(sweep_id, function=lambda: train_NN(filepath=filepath, NN_type="NN_Shallow"), count=10)
    # wandb.agent(sweep_id, function=lambda: train_NN(filepath=filepath, NN_type="NN_Deep"), count=10)
    wandb.agent(sweep_id, function=lambda: train_NN(filepath=filepath, NN_type="NN_BatchNorm"), count=10)

    # train_NN(filepath=filepath, NN_type="NN_Shallow")
    # train_NN(filepath=filepath, NN_type="NN_Dropout")
    # train_NN(filepath=filepath, NN_type="NN_Deep")
    # train_NN(filepath=filepath, NN_type="NN_BatchNorm")