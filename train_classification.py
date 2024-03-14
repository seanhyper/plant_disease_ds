import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from classification_utils import CNN, get_dataloader
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, lr, beta):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title(f'Loss (lr={lr}, beta={beta})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy (lr={lr}, beta={beta})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/lr_{lr}_beta_{beta}_metrics.png')
    # plt.show()

def grid_search(train_loader, val_loader, num_epochs=10, learning_rates=[ 0.0001], b1_options = [ 0.9, 0.95], b2_options = [0.99, 0.999]):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"
        print ("MPS device not found.")
    print(device)
    for lr in learning_rates:
        for b1 in b1_options:
            for b2 in b2_options:
                beta = (b1, b2)
                model = CNN(num_classes=10)
                model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=beta)
                
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []

                for epoch in tqdm(range(num_epochs), desc="Training"):
                    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
                    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accuracies.append(train_accuracy)
                    val_accuracies.append(val_accuracy)

                    print(f"Epoch [{epoch + 1}/{num_epochs}], lr={lr}, beta={beta}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, lr, beta)


if __name__ == '__main__':
    train_loader, val_loader, _ = get_dataloader(batch_size=75 ,show= True)
    grid_search(train_loader, val_loader, num_epochs=20)
