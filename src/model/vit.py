import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, NamedTuple, Optional

import torch
import torch.nn as nn


class ConvNormActivation(torch.nn.Sequential):
    """
    The following code is sourced from Torchvision

    BSD 3-Clause License

    Copyright (c) Soumith Chintala 2016, 
    All rights reserved.

    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """
    The following code is sourced from Torchvision

    BSD 3-Clause License

    Copyright (c) Soumith Chintala 2016, 
    All rights reserved.

    Transformer Model Encoder for sequence to sequence translation.
    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


def kl_divergence(z, mu, std):
    """
    Copyright 2020 William Falcon
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl


class MAE_FS(nn.Module):
    """Masked Autoencoders for Few-Shot Learning"""

    def __init__(
        self,
        ways: int,
        shots: int,
        query_size: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        masking_ratio: float = 0.75,
        num_channels: int = 3,
        decoder_dim: int = 512,
        num_decoder_layers: int = 1,
        conv_embed = None,
        prototype_dim = None,
    ):
        super().__init__()
        # torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.ways = ways
        self.shots = shots
        self.query_size = query_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.masking_ratio = masking_ratio
        self.num_channels = num_channels
        self.decoder_dim = decoder_dim
        self.conv_embed = conv_embed

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    ConvNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        self.seq_length = ways * (shots + query_size)
        self.masked_seq_length = self.ways
        self.masked_seq_remainder = self.seq_length - self.masked_seq_length

        if prototype_dim is not None:
            self.prototype_dim = prototype_dim
        else:
            self.prototype_dim = (image_size // patch_size) ** 2 * hidden_dim

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.decoder_dim))

        # # Add a class token
        # self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.seq_length += 1

        self.encoder = Encoder(
            self.masked_seq_length,
            num_layers,
            num_heads,
            self.prototype_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.enc_pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, self.prototype_dim).normal_(std=0.02))

        self.decoder = Encoder(
            self.seq_length,
            num_decoder_layers,
            num_heads,
            self.decoder_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.dec_pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, self.decoder_dim).normal_(std=0.02))

        self.decoder_projection = nn.Linear(self.prototype_dim, self.decoder_dim)
        self.pixel_projection = nn.Linear(self.decoder_dim, self.prototype_dim)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        self.linear_logits = nn.Linear(self.prototype_dim, self.num_classes)


    def embed_imgs(self, x: torch.Tensor):
        x = x.squeeze(0)
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # Decompose image into a series of patches
        # Ex: (N, C, H, W) -> (N, Embed_dim, H//P_size, W//P_size)
        x = self.conv_proj(x)

        # Flatten to a single dimension
        x = x.reshape(n, self.hidden_dim * n_h * n_w).unsqueeze(0)

        return x


    def shuffle_tensor(self, x, shots, ways):
        batch = x.shape[0]
        seq = x.shape[1]
        dim = x.shape[2]

        # Generate the shuffled indices
        randperm = torch.rand(batch, seq).to(x.device)
        shuffled = torch.argsort(randperm, dim=1)
        unshuffled = torch.argsort(shuffled, dim=1)

        # Select the kept portion of the shuffled indices.
        # Shots * Ways are kept, which represents the support set
        x_masked = torch.gather(x, 
            index=shuffled[:, :(shots * ways)].unsqueeze(-1).repeat(1, 1, dim), dim=1)
        query_indices = shuffled[:, (shots * ways):]

        # Generate the respective masks
        masks = torch.ones([batch, seq]).to(x.device)
        masks[:, :self.masked_seq_length] = 0
        masks = torch.gather(masks, index=unshuffled, dim=1)
        
        return x_masked, unshuffled, masks, query_indices

    def unshuffle_tensor(self, x, unshuffled):
        assert x.shape[0] == unshuffled.shape[0]
        assert x.shape[1] == unshuffled.shape[1]

        x_unshuffled = torch.gather(x, 
            index=unshuffled.unsqueeze(-1).repeat(1, 1, x.shape[2]), dim=1)
        
        return x_unshuffled


    def forward(self, x: torch.Tensor, shots, query_size, ways, fs_mode=False, embed=True, add_noise=False, augment=False):
        # Reshape and permute the input tensor
        if embed:
            if self.conv_embed is not None:
                prototypes = self.conv_embed(x)
            else:
                prototypes = self.embed_imgs(x)
        else:
            prototypes = x
        
        if augment:
            a = prototypes.reshape(ways, (shots + query_size), -1)
            b = a[torch.randperm(ways), :, :]
            prototypes = ((a + b)/2).reshape(ways * (shots + query_size), -1)

        orig_prototypes = prototypes
        if add_noise:
            prototypes = prototypes + torch.rand_like(prototypes)

        # Embed position, exclude class token
        # so that the shape matches the input at this stage
        if int(prototypes.shape[0]) <= int(self.enc_pos_embedding.shape[1]):
            x = prototypes + self.enc_pos_embedding[:, :prototypes.shape[0]]
        else:
            # If the positional embedding is smaller than the number of prototypes, repeat
            repeat_amount = int(prototypes.shape[0] / self.enc_pos_embedding.shape[1]) + 1
            temp_pos_embedding = self.enc_pos_embedding.repeat(1, repeat_amount, 1)
            x = prototypes + temp_pos_embedding[:, :prototypes.shape[0]]

        if fs_mode:
            # Extract the support set
            # Support elements are before every query element section in the sequence
            # Ex: A 5-way, 5-shot sequenece with 15 query items would have elements
            # 0-4 as the support set and elements 5-19 as the query set
            if shots == 1:
                x = x[:, ::(shots + query_size)]
            else:
                section_len = shots + query_size
                temp = torch.zeros(x.shape[0], shots*ways, x.shape[2]).to(x.device)
                for i in range(ways):
                    temp[:, i*shots:(i*shots)+shots] = x[:, i*section_len:(i*section_len)+shots]
                x = temp
        else:
            x, unshuffled, masks, query_indices = self.shuffle_tensor(x, shots, ways)

        # Apply the encoder
        encoded_support = self.encoder.ln(self.encoder.layers(self.encoder.dropout(x)))

        # Project the encoded input
        x = self.decoder_projection(encoded_support)

        # Create a sequence of query masks and insert the encoded support tokens
        # at the same positions as in the original prototypes
        seq_length = ways * (shots + query_size)
        if fs_mode:
            b, _, _ = x.shape
            query_masks = self.mask_token.repeat((x.shape[0], (ways * query_size), 1)).reshape(x.shape[0], ways, query_size, -1)
            x = x.reshape(b, ways, shots, -1)
            x = torch.cat([x, query_masks], dim=2)
            x = x[:, :, torch.randperm(shots + query_size)]
            x = x.reshape(b, seq_length, -1)
        else:
            query_masks = self.mask_token.repeat((x.shape[0], (ways * query_size), 1))
            # Append query mask tokens
            x = torch.cat([x, query_masks], dim=1)
            # Unshuffle hidden dimension
            x = self.unshuffle_tensor(x, unshuffled)

        # Embed position for the decoder
        # x = query_masks + self.dec_pos_embedding[:, :query_masks.shape[0]]
        x = x.squeeze(0)
        if int(x.shape[0]) <= int(self.dec_pos_embedding.shape[1]):
            x = x + self.dec_pos_embedding[:, :x.shape[0]]
        else:
            # If the positional embedding is smaller than the number of prototypes, repeat
            repeat_amount = int(x.shape[0] / self.dec_pos_embedding.shape[1]) + 1
            temp_pos_embedding = self.dec_pos_embedding.repeat(1, repeat_amount, 1)
            x = x + temp_pos_embedding[:, :x.shape[0]]

        # Apply the decoder
        x = self.decoder.ln(self.decoder.layers(self.decoder.dropout(x)))

        # Upsample the last dim to match (patch_size * patch_size * num_channels)
        x = self.pixel_projection(x)

        logits = self.linear_logits(x)

        if fs_mode:
            return x, orig_prototypes, encoded_support, logits
        else:
            return x, orig_prototypes, encoded_support, logits, masks


if __name__ == "__main__":
    model = MAE_FS(
        image_size=84,
        patch_size=6,
        num_layers=12,
        num_heads=12,
        hidden_dim=1020,
        mlp_dim=3060,
        decoder_dim=504,
    )

    print(model(torch.randn(5, 3, 84, 84))[0].shape)

    # # Count the number of parameters in the model
    # count = 0
    # for param in model.parameters():
    #     count += param.numel()
    # print(f"Number of parameters: {count}")