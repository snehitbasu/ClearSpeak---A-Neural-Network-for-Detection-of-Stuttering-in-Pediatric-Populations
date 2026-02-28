import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Load your data

X = np.load("X_padded.npy")  # (samples, timesteps, features)
y = np.load("y.npy")         # (samples, labels)

print(f"Original shapes: X={X.shape}, y={y.shape}")


# Flatten features for simplicity

X_flat = X.reshape(X.shape[0], -1)


# Multi-label oversampling
# Ensure each label has at least 2000 positive samples

def multi_label_oversample(X, y, target_count=2000):
    X_res, y_res = X.copy(), y.copy()
    for label in range(y.shape[1]):
        pos_idx = np.where(y_res[:, label] == 1)[0]
        n_needed = target_count - len(pos_idx)
        if n_needed > 0:
            idx = np.random.choice(pos_idx, size=n_needed, replace=True)
            X_res = np.vstack([X_res, X[idx]])
            y_res = np.vstack([y_res, y[idx]])
    return X_res, y_res

X_res, y_res = multi_label_oversample(X_flat, y, target_count=2000)
print(f"After oversampling: X={X_res.shape}, y={y_res.shape}")


# Train / Val split

X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)


# Neural Network

class StutterNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.size(0) == 1:  # BatchNorm requires >1 batch
            x = x.repeat(2, 1)
            out = torch.relu(self.bn1(self.fc1(x)))
            out = torch.relu(self.bn2(self.fc2(out)))
            out = torch.sigmoid(self.fc3(out))
            return out[0].unsqueeze(0)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))

input_dim = X_train.shape[1]
model = StutterNN(input_dim)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -------------------------
# Focal Loss
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


