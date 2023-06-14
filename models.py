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
        # if output_attentions:
        #     return self._backbone(inputs_embeds=embeds, output_attentions=True)
        # else:
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return torch.squeeze(prediction[:, -1])
            # if self.n_out == 1:
            #     return prediction[:, ::2, 0][:, inds]  # predict only on xs
            # else:
            #     # return prediction
            #     return prediction[:, ::2, :self.n_out][:, inds]
        # if inds is None:
        #     inds = torch.arange(ys.shape[1])
        # else:
        #     inds = torch.tensor(inds)
        #     if max(inds) >= ys.shape[1] or min(inds) < 0:
        #         raise ValueError("inds contain indices where xs and ys are not defined")
        # zs = self._combine(xs, ys)
        # embeds = self._read_in(zs)
        # if output_attentions:
        #     return self._backbone(inputs_embeds=embeds, output_attentions=True)
        # else:
        #     output = self._backbone(inputs_embeds=embeds).last_hidden_state
        #     prediction = self._read_out(output)
        #     if self.n_out == 1:
        #         return prediction[:, ::2, 0][:, inds]  # predict only on xs
        #     else:
        #         return prediction[:, ::2, :self.n_out][:, inds]

# class TransformerModel(nn.Module):
#     def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_out=1):
#         super(TransformerModel, self).__init__()
#         configuration = GPT2Config(
#             n_positions=2 * n_positions,
#             n_embd=n_embd,
#             n_layer=n_layer,
#             n_head=n_head,
#             n_out=n_out,
#             resid_pdrop=0.0,
#             embd_pdrop=0.0,
#             attn_pdrop=0.0,
#             use_cache=False,
#         )
#         self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.n_out = n_out
#         self._read_in = nn.Linear(n_dims, n_embd)
#         self._backbone = GPT2Model(configuration)
#         self._read_out = nn.Linear(n_embd, n_out)

#     @staticmethod
#     def _combine(xs_b, ys_b):
#         """Interleaves the x's and the y's into a single sequence."""
#         bsize, points, dim = xs_b.shape
#         if xs_b.shape == ys_b.shape:
#             zs = torch.stack((xs_b, ys_b), dim=2)
#         else:
#             ys_b_wide = torch.cat(
#                 (
#                     ys_b.view(bsize, points, 1),
#                     torch.zeros(bsize, points, dim - 1, device=ys_b.device),
#                 ),
#                 axis=2,
#             )
#             zs = torch.stack((xs_b, ys_b_wide), dim=2)
#         zs = zs.view(bsize, 2 * points, dim)
#         return zs

#     def forward(self, xs, ys, inds=None, output_attentions=False):
#         if inds is None:
#             inds = torch.arange(ys.shape[1])
#         else:
#             inds = torch.tensor(inds)
#             if max(inds) >= ys.shape[1] or min(inds) < 0:
#                 raise ValueError("inds contain indices where xs and ys are not defined")
#         zs = self._combine(xs, ys)
#         embeds = self._read_in(zs)
#         if output_attentions:
#             return self._backbone(inputs_embeds=embeds, output_attentions=True)
#         else:
#             output = self._backbone(inputs_embeds=embeds).last_hidden_state
#             prediction = self._read_out(output)
#             if self.n_out == 1:
#                 return prediction[:, ::2, 0][:, inds]  # predict only on xs
#             else:
#                 return prediction[:, ::2, :self.n_out][:, inds]