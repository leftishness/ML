import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import ramanspy as rp

# Set the path to the local folder where data is stored
DATA_PATH = "./data/ramanspy"

def load_and_preprocess_data(dataset="train", data_path=DATA_PATH):
    # Load the dataset using ramanspy with the specified folder path
    spectra_data, labels = rp.datasets.bacteria(dataset=dataset, folder=data_path)

    # Convert to numpy array for easier manipulation
    X = spectra_data.spectral_data
    y = labels

    # Encode labels as integers for classification
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, label_encoder

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def train_autoencoder(X_train, input_dim, latent_dim, epochs=100):
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_X, in progress_bar:
            optimizer.zero_grad()
            _, reconstructed = autoencoder(batch_X)
            loss = criterion(reconstructed, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            progress_bar.set_postfix(loss=avg_loss)

    # Extract latent representations
    with torch.no_grad():
        X_train_latent = autoencoder.encoder(X_train)

    return autoencoder, X_train_latent

class AdaptivePartialClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdaptivePartialClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def train_classifier(X_train_latent, y_train, latent_dim, output_dim, hidden_dim=32, epochs=50):
    classifier = AdaptivePartialClassifier(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train_latent, y_train), batch_size=64, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)  # Ensure progress bar shows up
        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            logits = classifier(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Accumulate correct predictions for accuracy tracking
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_y).sum().item()
            total_predictions += batch_y.size(0)

            # Update loss in the progress bar during each batch
            avg_loss = epoch_loss / len(train_loader)
            progress_bar.set_postfix(loss=avg_loss)

        # Calculate accuracy after all batches are processed
        accuracy = correct_predictions / total_predictions

        # Update the progress bar with final loss and accuracy for the epoch
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)
        progress_bar.close()  # Close the progress bar to ensure it updates correctly at the end of each epoch

    return classifier

def evaluate_model(classifier, autoencoder, X_test, y_test, label_encoder):
    # Get latent representations for test data
    with torch.no_grad():
        X_test_latent = autoencoder.encoder(X_test)
        test_logits = classifier(X_test_latent)
        test_predictions = torch.argmax(test_logits, dim=1)

    # Convert label encoder classes to strings for classification report
    target_names = [str(class_label) for class_label in label_encoder.classes_]

    # Calculate accuracy and print classification report
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, test_predictions, target_names=target_names))

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(dataset="train", data_path=DATA_PATH)

    # Set dimensions
    input_dim = X_train.shape[1]
    latent_dim = 16  # Adjust as needed based on experimentation

    # Train autoencoder and extract latent representations
    autoencoder, X_train_latent = train_autoencoder(X_train, input_dim=input_dim, latent_dim=latent_dim)

    # Train classifier on latent representations
    output_dim = len(np.unique(y_train))
    classifier = train_classifier(X_train_latent, y_train, latent_dim=latent_dim, output_dim=output_dim)

    # Evaluate model on test set
    evaluate_model(classifier, autoencoder, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()