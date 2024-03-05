import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dict, target_count, transform=None):
        self.data_dict = data_dict
        self.target_count = target_count
        self.transform = transform

        # Initialize lists to hold images and labels
        self.images = []
        self.labels = []

        # Augment images and store them in the dataset
        self.augment_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def augment_images(self):
        for class_name, images in self.data_dict.items():
            if len(images) < self.target_count:
                # Augment images in low classes
                num_images_to_augment = self.target_count - len(images)
                augmented_images = self.perform_augmentation(images, num_images_to_augment)
                self.images.extend(augmented_images)
                self.labels.extend([class_name] * len(augmented_images))
            else:
                self.images.extend(images)
                self.labels.extend([class_name] * len(images))

    def perform_augmentation(self, images, num_images_to_augment):
        augmented_images = []
        num_augmentations_per_image = max(1, num_images_to_augment // len(images))

        for img in images:
            pil_img = transforms.ToPILImage()(img)  # Convert to PIL Image
            for _ in range(num_augmentations_per_image):
                datagen = transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor()
                ])
                augmented_image = datagen(pil_img)
                augmented_images.append(augmented_image)

        # Shuffle the augmented images to ensure randomness
        random.shuffle(augmented_images)

        return augmented_images


def load_dataset(dataset_path):
    data_dict = {}
    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            # Initialize an empty list for each subfolder
            data_dict[dir_name] = []

            # Get the full path to the subfolder
            subfolder_path = os.path.join(root, dir_name)

            # Iterate through the files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Construct the full path to the image
                    image_path = os.path.join(subfolder_path, file)

                    # Load the image using OpenCV
                    image = cv2.imread(image_path)
                    # Convert from BGR to RGB color space
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Append the image to the list associated with the subfolder name
                    data_dict[dir_name].append(image)

    return data_dict


def main():
    # Path to the dataset directory
    dataset_dir = 'data/PlantVillage/'

    # Load your dataset
    data_dict = load_dataset(dataset_dir)
    # Specify the target count for each class
    target_count = 3000

    # Create a custom dataset instance with augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    custom_dataset = CustomDataset(data_dict, target_count, transform=transform)

    # Create a PyTorch DataLoader
    batch_size = 32
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the dataloader
    for images, labels in dataloader:
        print(f"Batch size: {images.shape[0]}")
        # Your training code here


if __name__ == "__main__":
    main()
