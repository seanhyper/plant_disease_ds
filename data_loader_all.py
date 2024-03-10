import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

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

NUM_2_CLASS = {v: k for k, v in CLASS_2_NUM.items()}

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

class DatasetAll(Dataset):
    def __init__(self, data_dict, target_count):
        self.data_dict = data_dict
        self.target_count = target_count

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

        image = transforms.ToTensor()(image)

        return image, label

    def augment_images(self):
        for class_name, images in self.data_dict.items():
            if len(images) < self.target_count:
                # Augment images in low classes
                augmented_images = self.perform_augmentation(images, self.target_count)
                self.images.extend(augmented_images)
                self.labels.extend([class_name] * len(augmented_images))
            elif len(images) > self.target_count:
                # Randomly drop images in classes with more samples
                sampled_images = random.sample(images, self.target_count)
                self.images.extend(sampled_images)
                self.labels.extend([class_name] * len(sampled_images))
            else:
                self.images.extend(images)
                self.labels.extend([class_name] * len(images))


    def perform_augmentation(self, images, num_images_to_augment):
        augmented_images = []
        num_images_per_image = num_images_to_augment // len(images)
        remainder = num_images_to_augment % len(images)

        for img in images:
            pil_img = transforms.ToPILImage()(img)  # Convert to PIL Image
            
            # Apply augmentations for the number of times determined
            for _ in range(num_images_per_image):
                pil_img_aug = transforms.RandomRotation(20)(pil_img)
                pil_img_aug = transforms.RandomHorizontalFlip()(pil_img_aug)
                pil_img_aug = transforms.RandomVerticalFlip()(pil_img_aug)
                augmented_images.append(pil_img_aug)

        # Randomly sample the remaining images
        remaining_images = random.sample(images, remainder)
        for img in remaining_images:
            pil_img = transforms.ToPILImage()(img)
            pil_img_aug = transforms.RandomRotation(20)(pil_img)
            pil_img_aug = transforms.RandomHorizontalFlip()(pil_img_aug)
            pil_img_aug = transforms.RandomVerticalFlip()(pil_img_aug)
            augmented_images.append(pil_img_aug)

        # Shuffle the augmented images to ensure randomness
        random.shuffle(augmented_images)

        return augmented_images


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # need to impliment

    def forward(self, x):
        # need to impliment
        return x

def test_model():
    # need to impliment
    return
def test_data():
    # Path to the dataset directory
    dataset_dir = 'data/PlantVillage/'

    # Load your dataset
    data_dict_str = load_dataset(dataset_dir)
    data_dict = {}
    for key, value in data_dict_str.items():
        data_dict[CLASS_2_NUM[key]] = value

    # Specify the target count for each class
    target_count = 2200

    custom_dataset = DatasetAll(data_dict, target_count  )

    # Create a PyTorch DataLoader
    batch_size = 32
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    counter = {
        "0":0, "1":0, "2":0, "3":0,"4":0,"5":0 ,"6":0,"7":0,"8":0,"9":0
    }
    # Iterate through the dataloader
    for images, labels in dataloader:
        # Convert labels to strings
        labels = [str(label.item()) for label in labels]

        # Update the counter
        for label in labels:
            counter[label] += 1

        # Print batch size
        print(f"Batch: {images.shape}")

    # Print the final count
    print("Class Counts:")
    for class_label, count in counter.items():
        print(f"Class {class_label}: {count}")

    return dataloader
def main():
   dataloader = test_data()

if __name__ == "__main__":
    main()
