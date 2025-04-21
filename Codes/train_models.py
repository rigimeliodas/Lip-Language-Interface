# train_models.py

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=200):
    model.to(device)
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
       
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(acc)

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {train_losses[-1]:.4f} | Val Acc: {acc:.4f}")
    
    return train_losses, val_accuracies
