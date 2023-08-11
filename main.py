import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data_gen import format_data, get_regression_data, get_classification_data
from models import TransformerModel
import wandb
import yaml
from munch import Munch
import time

with open(f"configs/model_selection.yaml", "r") as yaml_file:
    args = Munch.fromYAML(yaml_file)

LOAD_MODEL = False
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

def get_manual_select_model(X, y):
    # loss_fn_none = nn.MSELoss(reduction="none")
    # loss_fn_none = nn.MSELoss(reduction="none")
    loss_fn_none = nn.BCELoss(reduction="none")
    
    min_indices = torch.argmin(torch.cat([torch.unsqueeze(loss_fn_none(X[:, -1, i], y), 1) for i in range(len(args.data.data_alphas))], dim=1), dim=1)

    return X[torch.arange(min_indices.shape[0]), -1, min_indices]

def get_ensemble_model(X, y):
    loss_fn_none = nn.MSELoss(reduction="none")
    softmax = nn.Softmax(dim=1)

    weights = softmax(-torch.cat([torch.unsqueeze(loss_fn_none(X[:, -1, i], y), 1) for i in range(len(args.data.data_alphas))], dim=1))

    return (weights * X[:, -1, :-1]).sum(dim=1)

def get_other_losses(dataloader, loss_fn, verbose=False):
    num_batches = len(dataloader)
    model.eval()

    manual_loss = 0
    average_loss = 0
    ensemble_loss = 0
    alpha_losses = [0 for _ in range(len(args.data.data_alphas))]

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            manual_loss += loss_fn(get_manual_select_model(X, y), y).item()
            average_loss += loss_fn(torch.mean(X[:, -1, :-1], dim=1), y).item()
            ensemble_loss += loss_fn(get_ensemble_model(X, y), y).item()
            alpha_losses = [sum(i) for i in zip(alpha_losses, [loss_fn(X[:, -1, i], y).item() for i in range(len(args.data.data_alphas))])]  
        
        manual_loss /= num_batches
        ensemble_loss /= num_batches
        alpha_losses = [alpha_loss/num_batches for alpha_loss in alpha_losses]

        if verbose:
            print(f"Manual loss: {manual_loss:>8f} Ensemble loss: {ensemble_loss:>8f} Alpha losses: {[round(alpha_loss, 8) for alpha_loss in alpha_losses]}")

    return manual_loss, average_loss, ensemble_loss, alpha_losses

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
        if args.data.classification:
            data_dict = get_classification_data(alphas=args.data.data_alphas, N=args.data.N, d_d=args.data.d_d, train_samp_per_class=args.data.train_samp_per_class, data_gen_factor=args.data.data_gen_factor)
        else:
            data_dict = get_regression_data(alphas=args.data.data_alphas, N=args.data.N, d_d=args.data.d_d, train_samp_per_class=args.data.train_samp_per_class)

        alphas, X, y = format_data(data_dict, train_samples_per_alpha=int(args.data.train_samp_per_class / len(args.data.data_alphas)))

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
        manual_loss, average_loss, ensemble_loss, alpha_losses = get_other_losses(test_dataloader, loss_fn)

        train_loss.append(new_train_loss)
        test_loss.append(new_test_loss)

        if t % 10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            print(f"Train Loss: {new_train_loss}, Test Loss: {new_test_loss}")
        
        if t % args.training.save_every_epochs == 0 and t > 0:
            save_path = f"{args.out_dir}/model_epoch{t}_time{int(time.time()*10)}.pt"
            print(f"Saving model at epoch {t} to {save_path}")
            torch.save(model.state_dict(), save_path)
            wandb.log(
                {
                    "save_name": save_path
                },
                step=t,
            )
        
        wandb.log(
            {
                "train_loss": new_train_loss,
                "test_loss": new_test_loss,
                "manual_loss": manual_loss,
                "average_loss": average_loss,
                "ensemble_loss": ensemble_loss
            } | {
                f"alpha_{args.data.data_alphas[i]}_loss": alpha_loss for i, alpha_loss in enumerate(alpha_losses)
            },
            step=t,
        )
    
    torch.save(model.state_dict(), f"{args.out_dir}/model{int(time.time()*10)}.pt")

    return train_loss, test_loss

if __name__ == "__main__":
    model = TransformerModel(
        n_dims=len(args.data.data_alphas) + 1,
        n_positions=args.data.N,
        n_layer=args.model.n_layer,
        n_head=args.model.n_head,
        n_embd=args.model.n_embd
    ).to(device)

    if LOAD_MODEL:
        model.load_state_dict(torch.load("/burg/home/bto2106/code/modelSelectInterp/models/model_epoch1750_time16869439352.pth"))

    wandb.init(dir=args.out_dir,
        project=args.wandb.project,
        entity=args.wandb.entity,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        # resume=True
        resume=False
    )

    train_loss, test_loss = train(model)
