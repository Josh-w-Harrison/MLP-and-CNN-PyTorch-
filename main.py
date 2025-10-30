import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
tfm = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
val_ds   = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def accuracy(logits, y): return (logits.argmax(1) == y).float().mean()

for epoch in range(10):
    model.train()
    for x,y in train_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # val
    model.eval(); val_loss=val_acc=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits,y).item()*x.size(0)
            val_acc  += accuracy(logits,y).item()*x.size(0)
    n=len(val_ds)
    print(f"epoch {epoch+1:02d} | val_loss={val_loss/n:.4f} | val_acc={val_acc/n:.4f}")
