import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, 
                 model : nn.Module, 
                 train_loader : DataLoader, 
                 val_loader : DataLoader, 
                 criterion, 
                 optimizer, 
                 device,
                 num_epochs=5,
                 run_name="mnist",
                 ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.run_name = run_name

    def training_epoch(self, data_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        return epoch_loss

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                val_loss += loss.item()

        val_loss /= len(data_loader)
        return val_loss
    
    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.training_epoch(self.train_loader)
            # val_loss, val_accuracy = self.evaluate(self.val_loader)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}')
                 
        torch.save(self.model.state_dict(), f'model_{self.run_name}.pth')

