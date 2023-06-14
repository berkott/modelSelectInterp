import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_gen import get_data
from models import TransformerModel
import wandb
import yaml
from munch import Munch

with open(f"configs/model_selection.yaml", "r") as yaml_file:
    args = Munch.fromYAML(yaml_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dict = get_data(alphas=args.data.data_alphas, N=args.data.N, d_d=args.data.d_d, train_samp_per_class=args.data.train_samp_per_class)

def nn_train(dataloader, model, loss_fn, optimizer, verbose=False):
    size = len(dataloader.dataset)
    model.train()
    avg_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
        if verbose:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return avg_loss/size

def nn_test(dataloader, model, loss_fn, verbose=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
        
        test_loss /= num_batches
        if verbose:
            print(f"Test Avg loss: {test_loss:>8f} \n")

    return test_loss

def train(model, X_train, y_train, X_test, y_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size)

    train_loss = []
    test_loss = []

    for t in range(args.training.epochs):
        new_train_loss = nn_train(train_dataloader, model, loss_fn, optimizer)
        new_test_loss = nn_test(test_dataloader, model, loss_fn)
        train_loss.append(new_train_loss)
        test_loss.append(new_test_loss)

        if t % 10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Train Loss: {new_train_loss}, Test Loss: {new_test_loss}")
        
        if t % args.training.save_every_epochs == 0:
            pass
        
        wandb.log(
            {
                "train_loss": new_train_loss,
                "test_loss": new_test_loss
            },
            step=t,
        )
    
    torch.save(model.state_dict(), f"{args.out_dir}/model.pth")

    return train_loss, test_loss

def format_data(data_dict):
    alphas_merged = torch.cat([torch.ones(args.data.train_samp_per_class)*a for a in data_dict], dim=0)
    X_merged = torch.cat([torch.cat([torch.unsqueeze(data_dict[a][b]["y_hat"], 2) for b in data_dict[a]] + [torch.unsqueeze(data_dict[a][a]["y_test"], 2)], dim=2) for a in data_dict], dim=0)
    X_merged[:, -1, -1] = 0
    y_merged = torch.cat([data_dict[a][a]["y_test"][:, -1] for a in data_dict], dim=0)

    randperm = torch.randperm(alphas_merged.shape[0])

    alphas = alphas_merged[randperm]
    X = X_merged[randperm]
    y = y_merged[randperm]

    print(f"Alphas: {alphas.shape}, X: {X.shape}, y: {y.shape}")

    return alphas, X, y

alphas, X, y = format_data(data_dict)

alphas_train = alphas[:int(alphas.shape[0]*args.data.train_prop)]
alphas_test = alphas[int(alphas.shape[0]*args.data.train_prop):]
X_train = X[:int(X.shape[0]*args.data.train_prop)]
X_test = X[int(X.shape[0]*args.data.train_prop):]
y_train = y[:int(y.shape[0]*args.data.train_prop)]
y_test = y[int(y.shape[0]*args.data.train_prop):]

model = TransformerModel(
    n_dims=len(args.data.data_alphas) + 1,
    n_positions=args.data.N,
    n_layer=args.model.n_layer,
    n_head=args.model.n_head,
    n_embd=args.model.n_embd
)

wandb.init(dir=args.out_dir,
    project=args.wandb.project,
    entity=args.wandb.entity,
    config=args.__dict__,
    notes=args.wandb.notes,
    name=args.wandb.name,
    resume=True
)

train_loss, test_loss = train(model, X_train, y_train, X_test, y_test)
