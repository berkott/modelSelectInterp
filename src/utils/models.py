import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=64, n_layer=3, n_head=2, n_out=1):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_out=n_out,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_out = n_out
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_out)

    def forward(self, X, inds=None, output_attentions=False):
        embeds = self._read_in(X)

        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return torch.squeeze(prediction[:, -1])