import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_gen import get_data, format_data
from models import TransformerModel
import wandb
import yaml
from munch import Munch
import time

with open(f"configs/model_selection.yaml", "r") as yaml_file:
    args = Munch.fromYAML(yaml_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        if verbose:
            loss, current = loss.item(), batch * len(X)
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

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr)
    loss_fn = nn.MSELoss()

    train_loss = []
    test_loss = []

    for t in range(args.training.epochs):
        data_dict = get_data(alphas=args.data.data_alphas, N=args.data.N, d_d=args.data.d_d, train_samp_per_class=args.data.train_samp_per_class)

        alphas, X, y = format_data(data_dict, train_samp_per_class=int(args.data.train_samp_per_class / len(args.data.data_alphas)))

        alphas_train = alphas[:int(alphas.shape[0]*args.data.train_prop)]
        alphas_test = alphas[int(alphas.shape[0]*args.data.train_prop):]

        X_train = X[:int(X.shape[0]*args.data.train_prop)]
        X_test = X[int(X.shape[0]*args.data.train_prop):]
        y_train = y[:int(y.shape[0]*args.data.train_prop)]
        y_test = y[int(y.shape[0]*args.data.train_prop):]

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size)

        new_train_loss = nn_train(train_dataloader, model, loss_fn, optimizer)
        new_test_loss = nn_test(test_dataloader, model, loss_fn)
        train_loss.append(new_train_loss)
        test_loss.append(new_test_loss)

        if t % 10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Train Loss: {new_train_loss}, Test Loss: {new_test_loss}")
        
        if t % args.training.save_every_epochs == 0:
            print(f"Saving model at epoch {t} to {args.out_dir}/model{int(time.time()*10)}.pth")
            torch.save(model.state_dict(), f"{args.out_dir}/model{int(time.time()*10)}.pth")
        
        wandb.log(
            {
                "train_loss": new_train_loss,
                "test_loss": new_test_loss
            },
            step=t,
        )
    
    torch.save(model.state_dict(), f"{args.out_dir}/model{int(time.time()*10)}.pth")

    return train_loss, test_loss

if __name__ == "__main__":
    model = TransformerModel(
        n_dims=len(args.data.data_alphas) + 1,
        n_positions=args.data.N,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        n_embd=args.model.n_embd
    ).to(device)

    wandb.init(dir=args.out_dir,
        project=args.wandb.project,
        entity=args.wandb.entity,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        resume=True
    )

    train_loss, test_loss = train(model)
