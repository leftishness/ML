import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import ramanspy as rp

# Set the path to the local folder where data is stored
DATA_PATH = "./data/ramanspy"

def load_and_preprocess_data(dataset="train", data_path=DATA_PATH):
    spectra_data, labels = rp.datasets.bacteria(dataset=dataset, folder=data_path)
    X = spectra_data.spectral_data
    y = labels

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, label_encoder

class ConvAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * (input_dim // 8), latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * (input_dim // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.fc(encoded_flat)
        decoded_flat = self.decoder_fc(latent).view(encoded.size())
        reconstructed = self.decoder(decoded_flat)
        return latent, reconstructed.squeeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        logits = self.fc(lstm_out[:, -1, :])
        return logits

def train_conv_autoencoder(model, data_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Autoencoder Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_X, in progress_bar:
            optimizer.zero_grad()
            latent, reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            avg_loss = epoch_loss / len(data_loader)
            progress_bar.set_postfix(loss=avg_loss)

def train_lstm_classifier(model, autoencoder, data_loader, criterion, optimizer, epochs=100):
    model.train()
    autoencoder.eval()  # Ensure the autoencoder is in evaluation mode
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        progress_bar = tqdm(data_loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()

            # Get the latent representation from the autoencoder
            with torch.no_grad():
                latent, _ = autoencoder(batch_X)

            # Pass the latent representation to the LSTM classifier
            logits = model(latent)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_y).sum().item()
            total_predictions += batch_y.size(0)

            avg_loss = epoch_loss / len(data_loader)
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

def evaluate_model(classifier, autoencoder, X_test, y_test, label_encoder):
    autoencoder.eval()
    classifier.eval()
    with torch.no_grad():
        X_test_latent, _ = autoencoder(X_test)
        logits = classifier(X_test_latent)
        predictions = torch.argmax(logits, dim=1)

    test_accuracy = accuracy_score(y_test, predictions)
    target_names = [str(class_label) for class_label in label_encoder.classes_]
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, predictions, target_names=target_names))

def main():
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data()
    input_dim = X_train.shape[1]
    latent_dim = 64
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    autoencoder = ConvAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    classifier = LSTMClassifier(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    ae_criterion = nn.MSELoss()
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    clf_criterion = nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)
    clf_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    train_conv_autoencoder(autoencoder, train_loader, ae_criterion, ae_optimizer)
    train_lstm_classifier(classifier, autoencoder, clf_loader, clf_criterion, clf_optimizer)
    evaluate_model(classifier, autoencoder, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
