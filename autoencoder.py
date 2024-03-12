from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
from torch import nn
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_DIR = 'data/PlantVillage/'
HEALTHY_FOLDER = 'Tomato_healthy'

CLASS_2_NUM = {
    "Tomato_healthy": 0,
    "Tomato__Tomato_YellowLeaf__Curl_Virus": 1,
    "Tomato_Early_blight": 2,
    "Tomato__Target_Spot": 3,
    "Tomato_Leaf_Mold": 4,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 5,
    "Tomato_Septoria_leaf_spot": 6,
    "Tomato__Tomato_mosaic_virus": 7,
    "Tomato_Bacterial_spot": 8,
    "Tomato_Late_blight": 9
}




class AutoencoderDataSet(Dataset):
    def __init__(self, root_dir=DATA_DIR, n_samples_from_each_class=None, random_state=42, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [os.path.join(root_dir + HEALTHY_FOLDER, file) for file in os.listdir(root_dir + HEALTHY_FOLDER)]
        self.labels = [CLASS_2_NUM[HEALTHY_FOLDER]]*len(self.samples)
        self.binary_labels = []
        self.idx_to_label = {v: k for k, v in CLASS_2_NUM.items()}
        self.random_state = random_state
        self.n_samples_from_each_class = n_samples_from_each_class

        self._sample_n_from_each_class()

    def _sample_n_from_each_class(self):
        for dir in os.listdir(self.root_dir):
            if 'tomato' in dir.lower() and dir != HEALTHY_FOLDER:

                subfolder_path = os.path.join(DATA_DIR, dir)

                np.random.seed(self.random_state)

                if self.n_samples_from_each_class:
                    class_sample = list(np.random.choice([os.path.join(subfolder_path, file) for file in os.listdir(subfolder_path)], 
                                                         min(self.n_samples_from_each_class, len(os.listdir(subfolder_path)))))
                    self.samples += class_sample
                    self.labels += [CLASS_2_NUM[dir]]*self.n_samples_from_each_class


    def convert_labels_to_binary(self):
        self.binary_labels = [0 if self.idx_to_label[label] == 'Tomato_healthy' else 1 for label in self.labels]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_splits(self, train_size=0.7, val_size=0.3):
        self.convert_labels_to_binary()

        healthy_indices = [i for i, label in enumerate(self.binary_labels) if label == 0]
        unhealthy_indices = [i for i, label in enumerate(self.binary_labels) if label == 1]

        train_idx, val_idx = train_test_split(healthy_indices, train_size=train_size, test_size=val_size, random_state=42)
        
        val_idx_partition_for_test = list(np.random.choice(val_idx, int(len(val_idx) / 2)))

        test_idx = unhealthy_indices + val_idx_partition_for_test

        train_dataset = Subset(self, train_idx)
        val_dataset = Subset(self, [idx for idx in val_idx if idx not in val_idx_partition_for_test])
        test_dataset = Subset(self, test_idx)

        return train_dataset, val_dataset, test_dataset



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # [batch, 16, 128, 128]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [batch, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # [batch, 256, 8, 8]
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # [batch, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # [batch, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # [batch, 32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # [batch, 16, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), # [batch, 3, 256, 256]
            nn.Sigmoid() # Use Sigmoid to bring the output in the range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def save_loss_plot(train_losses, val_losses, lr, beta1, beta2, epoch, folder='results/autoencoder'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f'loss_plot_lr{lr}_beta1{beta1}_beta2{beta2}_epochs{epoch}.png'
    filepath = os.path.join(folder, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'LR: {lr}, Beta1: {beta1}, Beta2: {beta2}, Epochs: {epoch}')
    plt.legend()
    plt.savefig(filepath)
    plt.show()
    plt.close()

def grid_search(autoencoder, train_loader, val_loader, learning_rates, betas, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for lr in learning_rates:
        for beta1, beta2 in betas:
            model = autoencoder().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

            train_losses, val_losses = [], []
            for epoch in tqdm(range(num_epochs), desc=f'Training LR={lr}, Beta1={beta1}, Beta2={beta2}'):
                model.train()
                train_loss = 0
                for images, _ in tqdm(train_loader, leave=False, desc='Train Batch'):
                    images = images.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_losses.append(train_loss / len(train_loader))

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, _ in tqdm(val_loader, leave=False, desc='Val Batch'):
                        images = images.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, images)
                        val_loss += loss.item()
                val_losses.append(val_loss / len(val_loader))

            save_loss_plot(train_losses, val_losses, lr, beta1, beta2, num_epochs)

def visualize_reconstructions(original, reconstructed, n=10):
    original = original.to('cpu').numpy()
    reconstructed = reconstructed.to('cpu').detach().numpy()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.transpose(original[i], (1, 2, 0)))
        plt.title("Original")
        plt.axis('off')

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = AutoencoderDataSet(n_samples_from_each_class=2, transform=transform)

    train_dataset, val_dataset, test_dataset = dataset.get_splits(train_size=0.7, val_size=0.3)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    grid_search(Autoencoder, train_loader, val_loader, [0.001, 0.0001], [(0.9, 0.999), (0.95, 0.999)], num_epochs=20)
