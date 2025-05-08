# TGDA and Gradient Matching Attack Implementation for MNIST
# Source Code from other papers/Modified by Me
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset

# Base model and loss
from models import LR  # You can replace with MLP or ConvNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilon = 0.03
epochs = 50
lr = 0.01
batch_size = 60000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset_clean = datasets.MNIST('./', train=True, download=True, transform=transform)
dataset_test = datasets.MNIST('./', train=False, transform=transform)
train_loader = DataLoader(dataset_clean, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=1000)

def run_tgda_attack():
    model = LR().to(device).double()
    model.train()

    for data, target in train_loader:
        data, target = data.to(device).double(), target.to(device)
        poison_count = int(epsilon * len(data))
        data_p = data[:poison_count].clone().detach().requires_grad_(True)
        target_p = target[:poison_count].clone().detach()

        optimizer = torch.optim.SGD([data_p], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data_p.view(poison_count, -1))
            loss = -nn.CrossEntropyLoss()(output, target_p)
            loss.backward()
            optimizer.step()
            print(f"[TGDA] Epoch {epoch}, Loss: {loss.item():.4f}")

        return data[poison_count:], target[poison_count:], data_p.detach(), target_p.detach()

def run_gradient_matching_attack():
    model = LR().to(device).double()
    model.train()

    for data, target in train_loader:
        data, target = data.to(device).double(), target.to(device)
        poison_count = int(epsilon * len(data))

        # Target gradient
        data_clean = data[poison_count:].view(-1, 784)
        target_clean = target[poison_count:]

        criterion = nn.CrossEntropyLoss()
        output_clean = model(data_clean)
        loss_clean = criterion(output_clean, target_clean)
        grad_target = torch.autograd.grad(loss_clean, model.parameters(), create_graph=True)

        data_p = data[:poison_count].clone().detach().requires_grad_(True)
        target_p = target[:poison_count].clone().detach()
        optimizer = torch.optim.SGD([data_p], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data_p.view(poison_count, -1))
            loss_poison = criterion(output, target_p)
            grad_poison = torch.autograd.grad(loss_poison, model.parameters(), create_graph=True)

            loss = 0.0
            for g1, g2 in zip(grad_poison, grad_target):
                loss += -torch.nn.functional.cosine_similarity(g1.view(1, -1), g2.view(1, -1))

            loss.backward()
            optimizer.step()
            print(f"[GM] Epoch {epoch}, Alignment Loss: {loss.item():.4f}")

        return data[poison_count:], target[poison_count:], data_p.detach(), target_p.detach()


def retrain_and_evaluate(data_clean, target_clean, data_poison, target_poison, test_loader):
    dataset_clean = TensorDataset(data_clean, target_clean)
    dataset_poison = TensorDataset(data_poison, target_poison)
    dataset_total = ConcatDataset([dataset_clean, dataset_poison])
    train_loader = DataLoader(dataset_total, batch_size=1000, shuffle=True)

    model = LR().to(device).double()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(5):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch.view(x_batch.size(0), -1))
            loss = nn.CrossEntropyLoss()(output, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x.view(x.size(0), -1))
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    print("TGDA Attack")
    data_c, target_c, data_tgda, target_tgda = run_tgda_attack()
    retrain_and_evaluate(data_c, target_c, data_tgda, target_tgda, test_loader)
    print("\n")
    print("Gradient Matching Attack")
    data_c, target_c, data_gm, target_gm = run_gradient_matching_attack()
    retrain_and_evaluate(data_c, target_c, data_gm, target_gm, test_loader)
