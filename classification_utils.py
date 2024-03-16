import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

class DatasetSplit(Dataset):
    def __init__(self, images , labels):
        super(DatasetSplit, self).__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # plt.imshow(image)
        # plt.title(f"Label: {label}")
        # plt.show()
        imagetensor = transforms.ToTensor()(image)

        return imagetensor, label


class DatasetAll():
    def __init__(self, data_dict, target_count):
        
        self.data_dict = data_dict
        self.target_count = target_count

        # Initialize lists to hold images and labels
        self.images = []
        self.labels = []

        # Augment images and store them in the dataset
        self.augment_images()

        self.train_ratio = 0.7
        self.val_ratio = 0.15

        total_samples = len(self.images)
        num_train = int(total_samples * self.train_ratio)
        num_val = int(total_samples * self.val_ratio)
        num_test = total_samples - num_train - num_val

        # Shuffle the data
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images[:], self.labels[:] = zip(*combined)

        # Split data into train, val, test sets
        train_images, train_labels = self.images[:num_train], self.labels[:num_train]
        val_images, val_labels = self.images[num_train:num_train + num_val], self.labels[num_train:num_train + num_val]
        test_images, test_labels = self.images[num_train + num_val:], self.labels[num_train + num_val:]

        self.train_data = DatasetSplit(train_images, train_labels)
        self.val_data = DatasetSplit(val_images,val_labels)
        self.test_data = DatasetSplit(test_images,test_labels)

    

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
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Placeholder size, will be updated in forward pass
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)  # Flatten along dimensions 1, 2, and 3 
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Softmax activation for classification
        x = F.softmax(x, dim=1)
        
        return x



def get_dataloader(dataset_dir = 'data/PlantVillage/' , target_count = 2200 ,batch_size = 32 ,show = True):
    # Load your dataset
    data_dict_str = load_dataset(dataset_dir)
    data_dict = {}
    for key, value in data_dict_str.items():
        data_dict[CLASS_2_NUM[key]] = value

    custom_dataset = DatasetAll(data_dict, target_count  )

    train_data = DataLoader(custom_dataset.train_data, batch_size=batch_size, shuffle=True )
    val_data = DataLoader(custom_dataset.val_data, batch_size=batch_size, shuffle=True )
    test_data = DataLoader(custom_dataset.test_data, batch_size=batch_size, shuffle=True )

    # for print 
    if show:
        counter = {
            "0":0, "1":0, "2":0, "3":0,"4":0,"5":0 ,"6":0,"7":0,"8":0,"9":0
        }
        # Iterate through the dataloader
        for images, labels in train_data:
            # Convert labels to strings
            labels = [str(label.item()) for label in labels]

            # Update the counter
            for label in labels:
                counter[label] += 1

        # Print the final count
        print("Class Counts train:")
        for class_label, count in counter.items():
            print(f"Class {class_label}: {count}")
        counter = {
            "0":0, "1":0, "2":0, "3":0,"4":0,"5":0 ,"6":0,"7":0,"8":0,"9":0
        }
        for images, labels in test_data:
                # Convert labels to strings
                labels = [str(label.item()) for label in labels]

                # Update the counter
                for label in labels:
                    counter[label] += 1

        # Print the final count
        print("Class Counts test:")
        for class_label, count in counter.items():
            print(f"Class {class_label}: {count}")
        counter = {
            "0":0, "1":0, "2":0, "3":0,"4":0,"5":0 ,"6":0,"7":0,"8":0,"9":0
        }
        for images, labels in val_data:
            # Convert labels to strings
            labels = [str(label.item()) for label in labels]

            # Update the counter
            for label in labels:
                counter[label] += 1


        # Print the final count
        print("Class Counts val:")
        for class_label, count in counter.items():
            print(f"Class {class_label}: {count}")

    return train_data , val_data , test_data
def main():
    train_data , val_data , test_data = get_dataloader()
    model = CNN(num_classes=10)

    # Print model architecture
    print(model)

    # Perform a forward pass on a test batch
    for images, labels in test_data:
        print(images.shape)
        outputs = model(images)
        print("Output shape:", outputs.shape)
        break  # Just perform one forward pass for demonstration
if __name__ == "__main__":
    main()
