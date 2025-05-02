import torch
from models.lstm_model import LSTMClassifier
from utils.preprocessing import prepare_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_test(num_layer, dropout, bidirectional):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, input_dim, output_dim = prepare_dataloaders()
    model = LSTMClassifier(input_dim, 128, output_dim, num_layer, dropout, bidirectional)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    print(classification_report(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()