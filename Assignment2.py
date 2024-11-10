# Import necessary libraries
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data augmentation and normalization for training (for ResNet and MobileNetV2)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Only normalization for validation and test (for ResNet and MobileNetV2)
transform_val_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Transforms for AlexNet
# Data augmentation and normalization for training (for AlexNet)
transform_train_alexnet = transforms.Compose([
    transforms.Resize(224),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  #
                         std=[0.229, 0.224, 0.225]),
])

# Only normalization for validation and test (for AlexNet)
transform_val_test_alexnet = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Helper function to train the model with early stopping
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10, patience=5):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping!')
                    time_elapsed = time.time() - since
                    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
                    print(f'Best Validation Accuracy: {best_acc:.4f} at epoch {best_epoch+1}')
                    # Load best model weights
                    model.load_state_dict(best_model_wts)
                    return model, val_acc_history, train_acc_history
            else:
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Accuracy: {best_acc:.4f} at epoch {best_epoch+1}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

# Helper function to evaluate the model
def evaluate_model(model, test_loader, classes):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    # Classification Report
    print(classification_report(true_labels, predictions, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Function to plot accuracy curves (optional)
def plot_accuracy(train_acc, val_acc, model_name):
    plt.figure()
    plt.plot(range(1, len(train_acc)+1), [acc.cpu() for acc in train_acc], label='Train Accuracy')
    plt.plot(range(1, len(val_acc)+1), [acc.cpu() for acc in val_acc], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main execution code
if __name__ == '__main__':
    # Load CIFAR-10 dataset
    train_dataset_full = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)

    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_val_test)

    # Train/Validation split
    train_size = int(0.85 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update transforms for validation dataset
    val_dataset.dataset.transform = transform_val_test

    # Data Loaders for AlexNet
    # Create new datasets and data loaders for AlexNet
    train_dataset_full_alexnet = datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_train_alexnet)

    test_dataset_alexnet = datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_val_test_alexnet)

    # Train/Validation split for AlexNet
    train_size_alexnet = int(0.85 * len(train_dataset_full_alexnet))
    val_size_alexnet = len(train_dataset_full_alexnet) - train_size_alexnet

    train_dataset_alexnet, val_dataset_alexnet = random_split(
        train_dataset_full_alexnet, [train_size_alexnet, val_size_alexnet],
        generator=torch.Generator().manual_seed(42)
    )

    # Update transforms for validation dataset
    val_dataset_alexnet.dataset.transform = transform_val_test_alexnet

    # List of classes in CIFAR-10
    classes = test_dataset.classes

    # List of models to train
    models_to_train = ['ResNet18', 'AlexNet', 'MobileNetV2']

    # Hyperparameter options for tuning
    hyperparams = {
        'ResNet18': {
            'learning_rates': [0.01, 0.001],  
            'batch_sizes': [128],             
            'optimizers': ['SGD', 'Adam']    
        },
        'AlexNet': {
            'learning_rates': [0.01, 0.001],  
            'batch_sizes': [32],             
            'optimizers': ['SGD', 'Adam']     
        },
        'MobileNetV2': {
            'learning_rates': [0.01, 0.001],  
            'batch_sizes': [128],             
            'optimizers': ['SGD', 'Adam']     
        }
    }

    if 'ResNet18' in models_to_train:
        def train_resnet18():
            print("Training ResNet-18...")
            best_val_acc = 0.0
            best_model = None
            best_hyperparams = {}

            for lr in hyperparams['ResNet18']['learning_rates']:
                for opt_name in hyperparams['ResNet18']['optimizers']:
                    batch_size = hyperparams['ResNet18']['batch_sizes'][0]
                    print(f'\nHyperparameters: LR={lr}, Batch Size={batch_size}, Optimizer={opt_name}')
                    # DataLoaders
                    num_workers = 2
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    dataloaders = {'train': train_loader, 'val': val_loader}

                    # Initialize the model
                    resnet18 = models.resnet18()
                    resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
                    resnet18 = resnet18.to(device)

                    # Define Loss Function and Optimizer
                    criterion = nn.CrossEntropyLoss()
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(resnet18.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(resnet18.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                    # Train the model with early stopping
                    resnet18_model, resnet18_val_acc, resnet18_train_acc = train_model(
                        resnet18, criterion, optimizer, scheduler, num_epochs=10, dataloaders=dataloaders, patience=5)

                    final_val_acc = resnet18_val_acc[-1].cpu().item()
                    if final_val_acc > best_val_acc:
                        best_val_acc = final_val_acc
                        best_model = resnet18_model
                        best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt_name}

            print(f'\nBest Hyperparameters for ResNet-18: {best_hyperparams}')
            # Save the best model
            torch.save(best_model.state_dict(), 'resnet18_best_model.pth')
            # Evaluate the best model
            evaluate_model(best_model, test_loader, classes)
 

        train_resnet18()

    if 'AlexNet' in models_to_train:
        def train_alexnet():
            print("Training AlexNet...")
            best_val_acc = 0.0
            best_model = None
            best_hyperparams = {}

            for lr in hyperparams['AlexNet']['learning_rates']:
                for opt_name in hyperparams['AlexNet']['optimizers']:
                    batch_size = hyperparams['AlexNet']['batch_sizes'][0]
                    print(f'\nHyperparameters: LR={lr}, Batch Size={batch_size}, Optimizer={opt_name}')
                    # DataLoaders
                    num_workers = 2
                    train_loader_alexnet = DataLoader(train_dataset_alexnet, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    val_loader_alexnet = DataLoader(val_dataset_alexnet, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    test_loader_alexnet = DataLoader(test_dataset_alexnet, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    dataloaders_alexnet = {'train': train_loader_alexnet, 'val': val_loader_alexnet}

                    # Initialize the model
                    alexnet = models.alexnet()
                    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)
                    alexnet = alexnet.to(device)

                    # Define Loss Function and Optimizer
                    criterion = nn.CrossEntropyLoss()
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(alexnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(alexnet.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                    # Train the model with early stopping
                    alexnet_model, alexnet_val_acc, alexnet_train_acc = train_model(
                        alexnet, criterion, optimizer, scheduler, num_epochs=10, dataloaders=dataloaders_alexnet, patience=5)

                    final_val_acc = alexnet_val_acc[-1].cpu().item()
                    if final_val_acc > best_val_acc:
                        best_val_acc = final_val_acc
                        best_model = alexnet_model
                        best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt_name}

            print(f'\nBest Hyperparameters for AlexNet: {best_hyperparams}')
            # Save the best model
            torch.save(best_model.state_dict(), 'alexnet_best_model.pth')
            # Evaluate the best model
            evaluate_model(best_model, test_loader_alexnet, classes)
            # Plot accuracy curves (optional)
            # plot_accuracy(alexnet_train_acc, alexnet_val_acc, 'AlexNet')

        train_alexnet()

    if 'MobileNetV2' in models_to_train:
        def train_mobilenetv2():
            print("Training MobileNetV2...")
            best_val_acc = 0.0
            best_model = None
            best_hyperparams = {}

            for lr in hyperparams['MobileNetV2']['learning_rates']:
                for opt_name in hyperparams['MobileNetV2']['optimizers']:
                    batch_size = hyperparams['MobileNetV2']['batch_sizes'][0]
                    print(f'\nHyperparameters: LR={lr}, Batch Size={batch_size}, Optimizer={opt_name}')
                    # DataLoaders
                    num_workers = 2
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    dataloaders = {'train': train_loader, 'val': val_loader}

                    # Initialize the model
                    mobilenet = models.mobilenet_v2()
                    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)
                    mobilenet = mobilenet.to(device)

                    # Define Loss Function and Optimizer
                    criterion = nn.CrossEntropyLoss()
                    if opt_name == 'SGD':
                        optimizer = optim.SGD(mobilenet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    elif opt_name == 'Adam':
                        optimizer = optim.Adam(mobilenet.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

                    # Train the model with early stopping
                    mobilenet_model, mobilenet_val_acc, mobilenet_train_acc = train_model(
                        mobilenet, criterion, optimizer, scheduler, num_epochs=10, dataloaders=dataloaders, patience=5)

                    final_val_acc = mobilenet_val_acc[-1].cpu().item()
                    if final_val_acc > best_val_acc:
                        best_val_acc = final_val_acc
                        best_model = mobilenet_model
                        best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt_name}

            print(f'\nBest Hyperparameters for MobileNetV2: {best_hyperparams}')
            # Save the best model
            torch.save(best_model.state_dict(), 'mobilenetv2_best_model.pth')
            # Evaluate the best model
            evaluate_model(best_model, test_loader, classes)

        train_mobilenetv2()

    # Call the training functions
    train_resnet18()
    train_alexnet()
    train_mobilenetv2()
