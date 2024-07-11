import os
import tarfile
import urllib.request
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Enable GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Download and Extract Dataset
def download_and_extract_dataset(data_dir, url):
    filename = os.path.join(data_dir, 'images.tar')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(filename):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("Dataset already downloaded.")
    with tarfile.open(filename) as tar:
        tar.extractall(path=data_dir)
    print("Dataset extracted.")

# Step 2: Split Dataset
def split_dataset(data_dir, train_dir, val_dir):
    images_dir = os.path.join(data_dir, 'Images')
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        for breed in os.listdir(images_dir):
            breed_dir = os.path.join(images_dir, breed)
            images = os.listdir(breed_dir)
            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
            os.makedirs(os.path.join(train_dir, breed), exist_ok=True)
            os.makedirs(os.path.join(val_dir, breed), exist_ok=True)
            for img in train_images:
                shutil.move(os.path.join(breed_dir, img), os.path.join(train_dir, breed, img))
            for img in val_images:
                shutil.move(os.path.join(breed_dir, img), os.path.join(val_dir, breed, img))
        print("Dataset split into train and validation sets.")
    else:
        print("Dataset already split.")

# Step 3: Custom Dataset
class DogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                self.images.append((os.path.join(cls_folder, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 4: Data Augmentation and Preparation
def prepare_data_loaders(train_dir, val_dir):
    print("Preparing data loaders...")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DogDataset(train_dir, transform=transform_train)
    val_dataset = DogDataset(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Data loaders prepared.")
    return train_loader, val_loader, len(train_dataset.classes)

# Step 5: Load Pretrained ResNet50 Model
def build_model(num_classes):
    print("Building model...")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    print("Model built.")
    return model

# Step 6: Train the Model
def train_model(model, train_loader, val_loader, num_epochs=20):
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.topk(outputs, 3, dim=1)
            total += labels.size(0)
            correct += sum([labels[i].item() in predicted[i].tolist() for i in range(labels.size(0))])
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.topk(outputs, 3, dim=1)
                total += labels.size(0)
                correct += sum([labels[i].item() in predicted[i].tolist() for i in range(labels.size(0))])
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Save class indices
    class_indices = train_loader.dataset.class_to_idx
    with open('class_indices.npy', 'wb') as f:
        np.save(f, class_indices)
    print("Class indices saved.")

    # Save training history
    with open('training_history.npy', 'wb') as f:
        np.save(f, history)
    print("Training history saved.")
    return history

# Step 7: Predict the Breed of a New Image
def predict_breed(model, img_path):
    print(f"Predicting breed for image: {img_path}")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities, predicted_indices = torch.topk(torch.softmax(outputs, dim=1), 3)

    with open('class_indices.npy', 'rb') as f:
        class_indices = np.load(f, allow_pickle=True).item()
    class_labels = {v: k for k, v in class_indices.items()}

    top3_predictions = [(class_labels[idx.item()], prob.item()) for idx, prob in zip(predicted_indices[0], probabilities[0])]
    for breed, prob in top3_predictions:
        print(f"Predicted breed: {breed}, Probability: {prob:.4f}")

    return top3_predictions

def plot_training_history():
    print("Plotting training history...")
    with open('training_history.npy', 'rb') as f:
        history = np.load(f, allow_pickle=True).item()

    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy (Top-3)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save plots as image
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'training_history.png'))
    plt.show()
    print("Training history plotted.")

def evaluate_model(model, val_loader):
    print("Evaluating model...")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.topk(outputs, 3, dim=1)
            total += labels.size(0)
            correct += sum([labels[i].item() in predicted[i].tolist() for i in range(labels.size(0))])
    
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    metrics = {
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }
    
    # Save metrics as text file
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print("Model evaluation complete. Metrics saved.")


def main():
    print("Starting main...")
    data_dir = 'data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

    if not os.path.exists('best_model.pth') or not os.path.exists('class_indices.npy'):
        print("Preparing data and model...")
        download_and_extract_dataset(data_dir, url)
        split_dataset(data_dir, train_dir, val_dir)
        train_loader, val_loader, num_classes = prepare_data_loaders(train_dir, val_dir)
        model = build_model(num_classes)
        history = train_model(model, train_loader, val_loader)
        
        plot_training_history()
        evaluate_model(model, val_loader)
    else:
        print("Loading existing model...")
        model = build_model(num_classes=120)  # Assuming 120 classes
        model.load_state_dict(torch.load('best_model.pth'))

    # Predict the breed of the new image 'dog.jpg'
    img_paths = ['dog.jpg']
    for img_path in img_paths:
        predict_breed(model, img_path)
    print("Main complete.")

if __name__ == "__main__":
    main()
