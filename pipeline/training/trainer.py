"""
Core Training Loop with experiment tracking
"""
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_metric = float("inf")
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train(self, model, train_loader, val_loader=None, epochs=100):
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.get("amp", True))
        
        model.to(self.device)
        patience = self.config.get("patience", 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_loss, correct, total = 0, 0, 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.config.get("amp", True)):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = total_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation phase
            val_loss, val_acc = 0, 0
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                
                # Early stopping
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    patience_counter = 0
                    self._save_checkpoint(model, optimizer, epoch, val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
            
            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        return self.history
    
    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                total_loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), 100.0 * correct / total
    
    def _save_checkpoint(self, model, optimizer, epoch, loss):
        path = self.checkpoint_dir / "best.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, path)
    
    def _build_optimizer(self, model):
        name = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 1e-3)
        wd = self.config.get("weight_decay", 0.01)
        
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer: {name}")
    
    def _build_scheduler(self, optimizer, epochs):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
