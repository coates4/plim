import os
import math
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from models import LR, LinearModel, ConvNet

def get_model(model_name, device):
    if model_name == 'lr':
        model = LR().to(device).double()
    elif model_name == 'mlp':
        model = LinearModel().to(device).double()
    elif model_name == 'cnn':
        model = ConvNet().to(device).double()
    else:
        raise ValueError("Unsupported model type.")
    return model

def autograd(outputs, inputs, create_graph=False):
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [g if g is not None else p.new_zeros(p.size()) for g, p in zip(grads, inputs)]

class PoisonedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])

def preprocess_input(data, model_name):
    if model_name in ['lr', 'mlp']:
        return data.view(data.size(0), -1)
    return data

def get_learning_rate(model_name):
    if model_name == 'lr':
        return 0.05
    elif model_name == 'mlp':
        return 0.02
    else:
        return 0.01

def pretrain_model(model, device, train_loader, test_loader, model_path, epochs=10, model_name='cnn'):
    lr = get_learning_rate(model_name)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {model_name} with learning rate: {lr}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device).double(), target.to(device).long()
            data = preprocess_input(data, model_name)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Pretrain] Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device).double(), target.to(device).long()
                data = preprocess_input(data, model_name)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"[Pretrain] Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Pretrained model saved to {model_path}")

def run_gc(args):
    torch.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_name = args.model
    epsilon_w = args.epsilon_w
    epochs = args.epochs
    lr = args.lr

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./', train=False, transform=transform)

    pre_loader = DataLoader(dataset1, batch_size=60000, shuffle=False)
    train_loader = DataLoader(dataset1, batch_size=60000, shuffle=False)
    test_loader = DataLoader(dataset2, batch_size=10000)

    results_path = f"results_{model_name}.csv"
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epsilon', 'Test Accuracy'])

        for epsilon in [0.03, 0.1, 1.0]:
            poison_path = f'poisoned_models/{model_name}'
            clean_grad_path = f'clean_gradients/clean_grad_{model_name}.pt'
            os.makedirs(poison_path + '/img', exist_ok=True)

            model = get_model(model_name, device)
            target_model_path = args.target_model or f"target_models/mnist_gd_{model_name}_{epsilon_w}.pt"
            if args.pretrain or not os.path.exists(target_model_path):
                print("==> Pretraining model on clean data")
                pretrain_model(model, device, pre_loader, test_loader, target_model_path, epochs=10, model_name=model_name)
            else:
                model.load_state_dict(torch.load(target_model_path))

            if not os.path.exists(clean_grad_path):
                os.makedirs(os.path.dirname(clean_grad_path), exist_ok=True)
                for data, target in pre_loader:
                    data, target = data.to(device).double(), target.to(device).long()
                    data = preprocess_input(data, model_name)
                    data.requires_grad = True
                    output = model(data)
                    loss = nn.CrossEntropyLoss(reduction='sum')(output, target)
                    grad_c = autograd(loss, tuple(model.parameters()), create_graph=False)
                    torch.save(grad_c[0], clean_grad_path)

            g1 = torch.load(clean_grad_path).to(device)
            loss_all = []

            def adjust_learning_rate(lr, epoch):
                return lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))

            def attack(epoch, lr):
                nonlocal g1
                lr = adjust_learning_rate(lr, epoch)
                for data, target in train_loader:
                    data, target = data.to(device).double(), target.to(device).long()
                    if epoch == 0:
                        data_p = Variable(data[:int(epsilon * len(data))])
                        target_p = Variable(target[:int(epsilon * len(target))])
                        torch.save(target_p, f'{poison_path}/target_p_{epsilon}_{epsilon_w}.pt')
                    else:
                        data_p = torch.load(f'{poison_path}/data_p_{epsilon}_{epsilon_w}.pt')
                        target_p = torch.load(f'{poison_path}/target_p_{epsilon}_{epsilon_w}.pt')

                    data_p.requires_grad = True
                    data_in = preprocess_input(data_p, model_name)
                    output_p = model(data_in)
                    loss_p = nn.CrossEntropyLoss(reduction='sum')(output_p, target_p)
                    grad_p = autograd(loss_p, tuple(model.parameters()), create_graph=True)[0]
                    grad_sum = g1 + grad_p
                    loss = torch.norm(grad_sum, 2).square()
                    loss_all.append(loss.item())

                    if loss < 1:
                        break

                    update = autograd(loss, data_p, create_graph=True)[0]
                    data_t = data_p - lr * update
                    data_t = torch.clamp(data_t, 0, 1)

                    torch.save(data_t, f'{poison_path}/data_p_{epsilon}_{epsilon_w}.pt')

                    print(f"epoch: {epoch}, loss: {loss.item():.4f}, lr: {lr:.4f}")

            print(f"==> Running GC attack for epsilon={epsilon}")
            for epoch in range(epochs):
                attack(epoch, lr)

            print("==> Retraining model with clean + poisoned data")
            data_p = torch.load(f'{poison_path}/data_p_{epsilon}_{epsilon_w}.pt')
            target_p = torch.load(f'{poison_path}/target_p_{epsilon}_{epsilon_w}.pt')
            dataset_p = PoisonedDataset(data_p, target_p)
            dataset_total = ConcatDataset([dataset1, dataset_p])
            train_loader_retrain = DataLoader(dataset_total, batch_size=10000, shuffle=True)

            model1 = get_model(model_name, device)
            retrain_lr = get_learning_rate(model_name)
            optimizer = optim.SGD(model1.parameters(), lr=retrain_lr)
            train_loss_all = []

            def retrain(epoch):
                model1.train()
                for data, target in train_loader_retrain:
                    data, target = data.to(device).double(), target.to(device).long()
                    data = preprocess_input(data, model_name)
                    optimizer.zero_grad()
                    output = model1(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss_all.append(loss.item())

            def evaluate():
                model1.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device).double(), target.to(device).long()
                        data = preprocess_input(data, model_name)
                        output = model1(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / len(dataset2)
                print(f"Test Accuracy (epsilon={epsilon}): {accuracy:.2f}%")
                return accuracy

            for epoch in range(10):
                retrain(epoch)

            final_acc = evaluate()
            writer.writerow([epsilon, final_acc])

            plt.plot(train_loss_all)
            plt.savefig(f'{poison_path}/img/retrain_loss_{epsilon}_{epsilon_w}.png')
            plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['lr', 'mlp', 'cnn'], required=True)
    parser.add_argument('--epsilon_w', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--target_model', type=str, default=None, help='Path to pre-trained model file (optional)')
    parser.add_argument('--pretrain', action='store_true', help='Train the target model before attacking')
    args = parser.parse_args()

    run_gc(args)
