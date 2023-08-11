import torch
from utils.models import TransformerModel
from utils.training import *
import wandb
from munch import Munch
import argparse

# Command line parser
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
cmd_args = vars(parser.parse_args())

args = Munch.fromYAML(open(cmd_args.config_file, "r"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(n_dims=len(args.data.data_alphas) + 1, 
                         n_positions=args.data.N,
                         n_layer=args.model.n_layer,
                         n_head=args.model.n_head,
                         n_embd=args.model.n_embd)
model.to(device)

wandb.init(dir=args.out_dir,
           project=args.wandb.project,
           entity=args.wandb.entity,
           config=args.__dict__,
           notes=args.wandb.notes,
           name=args.wandb.name,
           resume=False)

train_loss, test_loss = train(model, device, args)
