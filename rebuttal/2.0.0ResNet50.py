import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

class ISICDataset(Dataset):
    
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(dataframe['class'].unique()))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        img_path = row['img_path']
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
        full_img_path = os.path.join(parent_dir, img_path)

        image = Image.open(full_img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.class_to_idx[row['class']]
        
        return image, label

def create_output_folder(base_name="resnet50_results"):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    return output_dir

def create_data_loaders(csv_path, batch_size=32, val_ratio=0.2):
    
    
    df = pd.read_csv(csv_path)
    
    test_df = df[df['dataset_split'] == 'test'].reset_index(drop=True)
    print(f"Total test images: {len(test_df)}")
    print(f"Class distribution:\n{test_df['class'].value_counts()}")
    
    train_df, val_df = train_test_split(test_df, test_size=val_ratio, 
                                        stratify=test_df['class'], 
                                        random_state=42)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ISICDataset(train_df, transform=train_transform)
    val_dataset = ISICDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_dataset.class_to_idx

def create_model(num_classes):
    
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_training_log(output_dir, train_losses, train_accs, val_losses, val_accs, 
                     val_labels, val_preds, class_names, final_acc):
    
    log_path = os.path.join(output_dir, 'training_log.txt')
    
    with open(log_path, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Final validation accuracy: {final_acc:.2f}%\n\n")
        
        f.write("Epoch-wise Performance:\n")
        for epoch in range(len(train_losses)):
            f.write(f"Epoch {epoch+1}: ")
            f.write(f"Train Loss: {train_losses[epoch]:.4f}, ")
            f.write(f"Train Acc: {train_accs[epoch]:.2f}%, ")
            f.write(f"Val Loss: {val_losses[epoch]:.4f}, ")
            f.write(f"Val Acc: {val_accs[epoch]:.2f}%\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(val_labels, val_preds, 
                                    target_names=class_names, 
                                    digits=4))

def train_resnet50_classifier(csv_path, num_epochs=25, batch_size=32, learning_rate=0.001):
    
    
    output_dir = create_output_folder()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, class_to_idx = create_data_loaders(csv_path, batch_size)
    num_classes = len(class_to_idx)
    
    model = create_model(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_resnet50_model.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }, os.path.join(output_dir, 'checkpoint_best.pth'))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_resnet50_model.pth')))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    class_names = sorted(class_to_idx.keys())
    print("\nFinal Model Performance")
    print("=" * 30)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, 
                              target_names=class_names, 
                              digits=4))
    
    plot_confusion_matrix(val_labels, val_preds, class_names, output_dir)
    
    save_training_log(output_dir, train_losses, train_accs, val_losses, val_accs,
                     val_labels, val_preds, class_names, val_acc)
    
    config_path = os.path.join(output_dir, 'config.json')
    import json
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'csv_path': csv_path,
        'device': str(device),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nAll results saved to: {output_dir}")
    
    return model, val_acc, output_dir

if __name__ == "__main__":
    csv_path = "/mnt/SkinGenerativeModel/code/data/local/Dermofit/input/metadata_rebuttal_samesize_50.csv"
    
    try:
        model, final_acc, output_dir = train_resnet50_classifier(
            csv_path=csv_path,
            num_epochs=25,
            batch_size=32,
            learning_rate=0.001
        )
        
        print(f"\nTraining completed! Best validation accuracy: {final_acc:.2f}%")
        print(f"All outputs saved to: {output_dir}")
        print("Saved outputs:")
        print("- best_resnet50_model.pth (model weights)")
        print("- checkpoint_best.pth (full checkpoint)")
        print("- training_history.png (loss and accuracy plots)")
        print("- confusion_matrix.png (confusion matrix)")
        print("- training_log.txt (detailed training log)")
        print("- config.json (hyperparameters and configuration)")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()