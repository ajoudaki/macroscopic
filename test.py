import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader

# Configuration
config = {
    'batch_size': 512,
    'epochs': 5,
    'lr': 0.001,
    'use_gpu_loader': False,  # Set to True to use GPU preloaded data loader
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

class GPUDataLoader:
    """Drop-in replacement for DataLoader that preloads data to GPU"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cpu', max_samples=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.max_samples = max_samples
        
        # Preprocess and load data to GPU
        self.data, self.targets = self._preload_data(dataset)
        self.n_samples = len(self.data)
        
    def _preload_data(self, dataset):
        """Preprocess entire dataset and load to GPU"""
        print(f"Preprocessing and loading data to {self.device}...")
        
        # Determine how many samples to use
        n_total = len(dataset)
        n_use = min(self.max_samples or n_total, n_total)
        
        if n_use < n_total:
            print(f"Using {n_use}/{n_total} samples")
            indices = torch.randperm(n_total)[:n_use]
            subset = torch.utils.data.Subset(dataset, indices)
            temp_loader = DataLoader(subset, batch_size=n_use, shuffle=False)
        else:
            temp_loader = DataLoader(dataset, batch_size=n_total, shuffle=False)
        
        try:
            # Load and preprocess all data at once
            data, targets = next(iter(temp_loader))
            
            # Move to device
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            print(f"Successfully loaded {len(data)} samples to {self.device}")
            return data, targets
            
        except torch.cuda.OutOfMemoryError:
            print("GPU OOM! Trying with fewer samples...")
            fallback_samples = min(10000, n_use // 2)
            return self._preload_data_with_limit(dataset, fallback_samples)
    
    def _preload_data_with_limit(self, dataset, max_samples):
        """Fallback method with limited samples"""
        indices = torch.randperm(len(dataset))[:max_samples]
        subset = torch.utils.data.Subset(dataset, indices)
        temp_loader = DataLoader(subset, batch_size=max_samples, shuffle=False)
        data, targets = next(iter(temp_loader))
        return data.to(self.device), targets.to(self.device)
    
    def __iter__(self):
        """Iterator that yields batches like standard DataLoader"""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.data[batch_indices], self.targets[batch_indices]
    
    def __len__(self):
        """Number of batches"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                         download=True, transform=transform)
    return trainset, testset

def get_dataloader(dataset, config, is_train=True):
    """Factory function to create appropriate dataloader"""
    if config['use_gpu_loader'] and is_train:
        # Use GPU loader for training with memory management
        return GPUDataLoader(dataset, 
                           batch_size=config['batch_size'], 
                           shuffle=True,
                           device=config['device'])
    else:
        # Use standard DataLoader
        return DataLoader(dataset, 
                         batch_size=config['batch_size'], 
                         shuffle=is_train)

def train_model(model, dataloader, criterion, optimizer, config):
    """Unified training function that works with both loaders"""
    model.train()
    
    start_time = time.time()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        n_batches = 0
        
        for inputs, labels in dataloader:
            # Move to device only if using standard DataLoader
            if not config['use_gpu_loader']:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/n_batches:.4f}')
    
    return time.time() - start_time

def evaluate(model, testset, config):
    """Evaluate model accuracy"""
    testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def main():
    loader_type = "GPU DataLoader" if config['use_gpu_loader'] else "Standard DataLoader"
    print(f"Using device: {config['device']}")
    print(f"Training with: {loader_type}")
    
    # Get data
    trainset, testset = get_data()
    
    # Create dataloader
    trainloader = get_dataloader(trainset, config, is_train=True)
    
    # Initialize model
    model = MLP().to(config['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Train
    train_time = train_model(model, trainloader, criterion, optimizer, config)
    
    # Evaluate
    accuracy = evaluate(model, testset, config)
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Samples used: {len(trainloader.data) if config['use_gpu_loader'] else len(trainset)}")

if __name__ == "__main__":
    print("=" * 60)
    print("STRATEGY 1: Standard DataLoader")
    config['use_gpu_loader'] = False
    main()
    
    # Clear GPU memory between strategies
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("STRATEGY 2: GPU Preloaded DataLoader")
    config['use_gpu_loader'] = True
    main()