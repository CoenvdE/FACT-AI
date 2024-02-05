import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


def load(path):
    """
    Load the data and split it into train, validation and test sets
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # Cropping a central square patch of the image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    dataset = datasets.ImageFolder(root=path,transform = transform)
    
    train_size = int(0.80 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        prefetch_factor=4,  # Each worker will prefetch 4 batches in advance
        pin_memory=True     # Assuming you're using GPUs
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        prefetch_factor=4,  # Each worker will prefetch 4 batches in advance
        pin_memory=True     # Assuming you're using GPUs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        prefetch_factor=4,  # Each worker will prefetch 4 batches in advance
        pin_memory=True     # Assuming you're using GPUs
    )

    return train_loader, val_loader, test_loader

def load_stratified(path):
    """
    load the data and split it into train, validation and test sets
    same as load but with stratified split, unused in the code
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=path, transform=transform)
    
    # Stratified split
    targets = np.array([target for _, target in full_dataset.samples])
    train_indices, test_indices = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        shuffle=True,
        stratify=targets,
    )
    
    # Further split the test set into validation and test sets
    test_size = int(0.5 * len(test_indices))  # Split the test set in half
    val_indices, test_indices = test_indices[:test_size], test_indices[test_size:]
    
    # Create data subsets for train, validation, and test
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=64, num_workers=8, shuffle=True, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=8, shuffle=False, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=8, shuffle=False, pin_memory=True, prefetch_factor=4)

    return train_loader, val_loader, test_loader

def make_cifar100_dataloaders(path):
    """
    Create train and validation dataloaders for CIFAR-100 dataset
    not used in the code
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # Cropping a central square patch of the image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),  #TO-DO figure out impact & optimal values
    ])

    # Create the train dataset
    train_dataset = datasets.CIFAR100(path, download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create the validation dataset
    val_dataset = datasets.CIFAR100(path, download=True, train=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader