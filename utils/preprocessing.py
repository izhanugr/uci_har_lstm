import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_signals(folder_path, type_test):
    signals = []
    for signal in ["body_acc", "body_gyro", "total_acc"]:
        for axis in ["x", "y", "z"]:
            path = f"{folder_path}/Inertial Signals/{signal}_{axis}_{type_test}.txt"
            signals.append(np.loadtxt(path))
        
        stacked = np.stack(signals, axis=-1) #shape(samples, time_steps, input_size/feature/Axes)
        return stacked
    
def load_labels(folder_path):
    labels = np.loadtxt(f"{folder_path}/y_{'train' if 'train' in folder_path else 'test'}.txt")
    return labels.astype(int) - 1 #zero-based labels

def prepare_dataloaders(base_path="./data/UCI_HAR_Dataset", batch_size=64):
    X_train = load_signals(f"{base_path}/train", "train")
    y_train = load_labels(f"{base_path}/train")

    X_test = load_signals(f"{base_path}/test", "test")
    y_test = load_labels(f"{base_path}/test")

    #train/val split
    val_split = int(0.8 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    def to_tensor(x ,y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    
    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, X_train.shape[2], int(y_train.max()) + 1