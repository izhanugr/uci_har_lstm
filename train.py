import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm_model import LSTMClassifier
from utils.preprocessing import prepare_dataloaders
from utils.early_stopping import EarlyStopping

def run_training(batch_size=64, epochs=50, lr=1e-3, patience=5, num_layer=1, bidirectional=False, dropout=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, input_dim, output_dim = prepare_dataloaders(batch_size=batch_size)

    model = LSTMClassifier(input_dim, 128, output_dim, num_layer, dropout, bidirectional).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience, delta=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss ={avg_loss:.2f}, Accuracy={correct/total:.2%}")

        #validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_correct += (output.argmax(dim=1)==y).sum().item()
                val_total += y.size(0)

        val_avg_loss = val_loss / len(val_loader)
        val_accuracy = val_correct/val_total
        print(f"Val Loss={val_avg_loss:.2f}, Val Accuracy={val_accuracy:.2%}")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early Stopping.")
            break
    

    early_stopper.save_best_model("save_model.pth")
