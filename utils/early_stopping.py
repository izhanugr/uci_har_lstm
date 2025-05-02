import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

        else:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self, path="best_model.pth"):
        torch.save(self.best_model_state, path)

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

