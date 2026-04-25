import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# PART 1: PRUNABLE LINEAR
# -----------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# -----------------------------
# MODEL
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# SPARSITY LOSS
# -----------------------------
def compute_sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += torch.mean(torch.clamp(gates - 0.05, min=0) ** 2)
    return loss

# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train(model, loader, optimizer, lambda_sparse):
    model.train()
    total_loss = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        ce_loss = F.cross_entropy(output, target)
        sp_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

# -----------------------------
# TEST FUNCTION
# -----------------------------
def test(model, loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return 100. * correct / len(loader.dataset)

# -----------------------------
# SPARSITY CHECK
# -----------------------------
def compute_sparsity(model):
    total = 0
    pruned = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            pruned += (gates < 0.1).sum().item()

    return 100 * pruned / total

# -----------------------------
# EXPERIMENT WITH DIFFERENT λ
# -----------------------------
lambdas = [1.0, 5.0, 10.0]
results = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # keep small for demo
        loss = train(model, train_loader, optimizer, lam)
        acc = test(model, test_loader)
        print(f"Epoch {epoch+1}, Loss: {loss:.2f}, Acc: {acc:.2f}%")

    sparsity = compute_sparsity(model)
    acc = test(model, test_loader)

    results.append((lam, acc, sparsity))

# -----------------------------
# PRINT RESULTS TABLE
# -----------------------------
print("\nResults:")
print("Lambda | Accuracy | Sparsity (%)")
for lam, acc, sp in results:
    print(f"{lam} | {acc:.2f} | {sp:.2f}")

# -----------------------------
# PLOT GATE DISTRIBUTION
# -----------------------------
def plot_gates(model):
    all_gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()

# plot last model
plot_gates(model)