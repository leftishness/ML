import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


# Orthogonal regularization
def orthogonal_regularization(layer_weights):
    rows = layer_weights.size(0)
    identity = torch.eye(rows, device=layer_weights.device)
    gram_matrix = torch.mm(layer_weights, layer_weights.T)
    return ((gram_matrix - identity) ** 2).sum()


# Covariance penalty
def covariance_penalty(features):
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-6)
    cov_matrix = torch.mm(features.T, features) / (features.size(0) - 1)
    penalty = cov_matrix - torch.diag(torch.diag(cov_matrix))
    return penalty.pow(2).sum()


# Entropy calculation
def entropy(features):
    prob = torch.softmax(features, dim=1)
    log_prob = torch.log(prob + 1e-6)
    return -(prob * log_prob).sum(dim=1).mean()


class EntropyMaximizingLinear(nn.Module):
    def __init__(self, in_features, out_features, use_orthogonality=True, use_entropy=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_orthogonality = use_orthogonality
        self.use_entropy = use_entropy

    def forward(self, x):
        features = self.linear(x)
        reg_loss = 0.0

        if self.use_orthogonality:
            reg_loss += orthogonal_regularization(self.linear.weight)

        if self.use_entropy:
            reg_loss -= entropy(features)  # Maximize entropy (negative sign for loss minimization)

        return features, reg_loss


class AdaptiveDropout(nn.Module):
    def __init__(self, base_rate=0.5):
        super().__init__()
        self.base_rate = base_rate

    def forward(self, x):
        with torch.no_grad():
            redundancy_score = (x.T @ x).abs().mean()
        adjusted_rate = self.base_rate + redundancy_score.clamp(max=0.3)
        mask = torch.bernoulli(torch.ones_like(x) * adjusted_rate)
        return x * mask / (1 - adjusted_rate)


class AttentionWithDiversity(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.attention_dim = attention_dim  # Store attention_dim as a class attribute

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Use self.attention_dim here
        scores = torch.softmax(torch.mm(q, k.T) / (self.attention_dim ** 0.5), dim=1)
        diversity_loss = entropy(scores)
        output = torch.mm(scores, v)
        return output, diversity_loss


class HistopathologyClassifierWithDiversity(nn.Module):
    def __init__(self, num_classes):
        super(HistopathologyClassifierWithDiversity, self).__init__()
        # ResNet50 as feature extractor
        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()

        # Diversity-promoting layers
        self.projector = nn.Sequential(
            EntropyMaximizingLinear(num_ftrs, 512),
            nn.ReLU(),
            EntropyMaximizingLinear(512, 256)
        )

        # Entropy-based attention
        self.attention = AttentionWithDiversity(256, 128)

        # Classifier head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)

        # Apply diversity-promoting projector
        proj_features, proj_loss = self.projector[0](x)
        proj_features = self.projector[1](proj_features)
        proj_features, diversity_loss = self.projector[2](proj_features)

        # Apply attention with diversity
        attention_output, attention_loss = self.attention(proj_features)

        # Final classification logits
        logits = self.classifier(proj_features)

        # Sum all diversity-promoting losses
        total_diversity_loss = proj_loss + diversity_loss + attention_loss
        return logits, proj_features, total_diversity_loss


def custom_loss(output_logits, entropy_features, labels, total_diversity_loss, entropy_lambda=0.1):
    ce_loss = nn.CrossEntropyLoss()(output_logits, labels)
    total_loss = ce_loss + entropy_lambda * total_diversity_loss
    return total_loss


def load_data(train_dir, test_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    class_weights = np.bincount(train_targets)
    class_weights = class_weights.max() / class_weights
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=[class_weights[label] for label in train_targets],
        num_samples=len(train_targets),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(
        train_dataset.dataset.classes), train_dataset.dataset.classes, class_weights


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            phase_pbar = tqdm(loader, desc=f"{phase.capitalize()} Phase [Epoch {epoch + 1}/{num_epochs}]")
            for i, (inputs, labels) in enumerate(phase_pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits, entropy_features, diversity_loss = model(inputs)
                    loss = custom_loss(logits, entropy_features, labels, diversity_loss)
                    _, preds = torch.max(logits, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total += batch_size

                avg_loss = running_loss / total
                avg_acc = (running_corrects.double() / total) * 100

                phase_pbar.set_postfix({
                    "iter": f"{i + 1}/{len(loader)}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "avg_acc": f"{avg_acc:.2f}%"
                })

            if phase == 'val' and avg_acc > best_acc:
                best_acc = avg_acc
                best_model_wts = model.state_dict()

        scheduler.step()

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)

            logits, _, _ = model(inputs)
            _, preds = torch.max(logits, 1)

            running_corrects += torch.sum(preds == labels.data)
            total += batch_size

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(all_labels, all_preds, target_names=class_names))


def main():
    train_dir = './NCT-CRC-HE-100K'
    test_dir = './CRC-VAL-HE-7K'
    batch_size = 32
    num_workers = 4
    num_epochs = 3
    learning_rate = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, num_classes, class_names, class_weights = load_data(
        train_dir, test_dir, batch_size, num_workers
    )

    model = HistopathologyClassifierWithDiversity(num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device)
    evaluate_model(model, test_loader, device, class_names)


if __name__ == '__main__':
    main()
