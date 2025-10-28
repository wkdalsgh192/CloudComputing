from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import time
import os


class CatDogDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.root_dir, file)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if file.startswith("cat") else 1
        return image, label


def load_data(data_dir):
    full_dataset = CatDogDataset(data_dir)
    
    print("Loaded dataset with", len(full_dataset), "images")
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    return train_loader, test_loader

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        # Convolutional layers
        # Conv1: RGB (3) → 16 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Conv2: 16 → 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully connected layers
        # After two 2×2 pools: 64 → 32 → 16 → feature map size = 16×16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary output (cat vs dog)

    def forward(self, x):
        # Conv1 → ReLU → MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Conv2 → ReLU → MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = torch.flatten(x, 1)

        # FC1 → ReLU
        x = F.relu(self.fc1(x))
        
        # FC2 → Sigmoid for binary classification
        x = torch.sigmoid(self.fc2(x))

        return x
    
def train(device, model, train_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_epoch = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_epoch
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss/len(train_loader):.4f}  Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

def evaluate(device, model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)

            predicted = (outputs > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    train_loader, test_loader = load_data('/mnt/c/Users/minhojang/Downloads/train/train')

    # Peek at one batch
    images, labels = next(iter(train_loader))
    print("Image batch:", images.shape)   # [32, 3, 64, 64]
    print("Label batch:", labels.shape)   # [32]
    
    start = time.time()
    model = CatDogCNN().to(device)
    train(device, model, train_loader, 10, 0.001)
    torch.save(model.state_dict(), "model_epoch10.pth")
    evaluate(device, model, test_loader)

    print(f"Total runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
