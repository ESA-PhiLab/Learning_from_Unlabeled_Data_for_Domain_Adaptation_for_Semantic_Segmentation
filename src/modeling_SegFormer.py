#!/usr/bin/env python 
# coding: utf-8

# We modify: http://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.     

""" The PyTorch SegFormer model. """

# # We have modified: http://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

import math 
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
 
import torch.nn.functional as F

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_segformer import SegformerConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SegformerConfig"

_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]
 
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
 
SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # # See all SegFormer models at https://huggingface.co/models?filter=segformer
]
 
class SegFormerImageClassifierOutput(ImageClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

class SegformerDropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class SegformerOverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width

class SegformerEfficientSelfAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            hidden_states = self.sr(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SegformerAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]   
        return outputs 

class SegformerDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states

class SegformerMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SegformerLayer(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states), 
            height,
            width,
            output_attentions=output_attentions,
        )  

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  

        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs
        return outputs

class SegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        blocks, cur = [], 0
        for i in range(config.num_encoder_blocks):
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            hidden_states, height, width = embedding_layer(hidden_states)
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            hidden_states = norm_layer(hidden_states)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class SegformerPreTrainedModel(PreTrainedModel):
    config_class = SegformerConfig
    base_model_prefix = "segformer"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

SEGFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
class SegformerModel(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = SegformerEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
    states) e.g. for ImageNet.
    """,
    SEGFORMER_START_DOCSTRING,
)
class SegformerForImageClassification(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config) 

        #import pdb; pdb.set_trace() 

        self.num_labels = config.num_labels
        #self.num_labels2 = config.num_labels2
        self.segformer = SegformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SegFormerImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # # global average pooling 
        sequence_output = sequence_output.mean(dim=1)

        logits = self.classifier(sequence_output)

        #import pdb; pdb.set_trace() 

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss() 
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class SegformerMLP(nn.Module):
    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config, alpha):
        super().__init__(config)
        # # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = [] 
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # # the following 3 layers implement the ConvModule of the original implementation   
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        #self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1) 

        #self.classifier2 = nn.Conv2d(config.decoder_hidden_size, config.num_labels2, kernel_size=1)      
        #config.num_labels2 = 2  
        config.num_labels2 = 3  
        self.classifier2 = nn.Linear(config.decoder_hidden_size, config.num_labels2)
        
        self.config = config 

    def forward(self, encoder_hidden_states: torch.FloatTensor, alpha: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = () 
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # # unify channel dimension   
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            
            # # # upsample        
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            
            all_hidden_states += (encoder_hidden_state,) 

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # # logits are of shape (batch_size, num_labels, height / 4, width / 4)                    
        #logits = self.classifier(hidden_states)    

        logits = self.classifier(hidden_states)

        #logits2 = self.classifier2(hidden_states)

        hidden_states = F.adaptive_avg_pool2d(hidden_states, 1).reshape(batch_size, -1)        
        
        #print(alpha)  
        
        hidden_states = ReverseLayerF.apply(hidden_states, alpha)
    
        logits2 = self.classifier2(hidden_states) 
        
        logits2 = F.softmax(logits2) 

        #return logits   
        return logits, logits2

@add_start_docstrings(
    """SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes. """,
    SEGFORMER_START_DOCSTRING,
)  
class SegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config, alpha):
        super().__init__(config)
        self.segformer = SegformerModel(config) 
        
        # # SegformerForSemanticSegmentation  

        #print(config)     
        #print(config.num_labels)     
        
        #self.decode_head = SegformerDecodeHead(config)   
        
        #self.alpha = alpha 
        
        #self.decode_head = SegformerDecodeHead(config, self.alpha)
        #self.decode_head, self.decode_head2 = SegformerDecodeHead(config, self.alpha)

        self.alpha = alpha
        self.decode_head = SegformerDecodeHead(config, self.alpha)

        #self.decode_head2 = SegformerDecodeHead(config, alpha)   
        
        #self.decode_head = SegformerDecodeHead(config, alpha)     

        # # Initialize weights and apply final processing                  
        self.post_init() 

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        alpha: Optional[torch.FloatTensor] = None,
        image_patches: Optional[torch.FloatTensor] = None,
        data2: Optional[torch.FloatTensor] = None,
        target2: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:     

        Examples:    

        ```python   
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # # shape (batch_size, num_labels, height / 4, width / 4)     
        >>> list(logits.shape)
        [1, 150, 128, 128]      
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # # we need the intermediate hidden states                    
            return_dict=return_dict,
        ) 

        #print(return_dict) 
        # # True    

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        #logits = self.decode_head(encoder_hidden_states)   
        #logits, logits2 = self.decode_head(encoder_hidden_states) 

        #print(alpha)  
        
        logits, logits2 = self.decode_head(encoder_hidden_states, alpha)
        
        #logits, logits2 = self.decode_head(encoder_hidden_states)  

        #logits2 = self.decode_head2(encoder_hidden_states)        

        #print(logits2)    
        #print(logits2.shape)      

        import segmentation_models_pytorch as smp
        
        loss = None  
        
        if labels is not None:
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1: 
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

                # print(upsampled_logits.shape)                                                                                          
                # print(labels.shape)        
                
                # # Dice Loss 
                
                #loss += smp.losses.JaccardLoss(mode="multiclass", classes=6)(y_pred=upsampled_logits, y_true=labels)    
                loss += smp.losses.DiceLoss(mode="multiclass", classes=6)(y_pred=upsampled_logits, y_true=labels)
                
                batch_size = labels.shape[0]
                domain_label = torch.zeros(batch_size).long().cuda()
                
                err_s_domain = nn.CrossEntropyLoss()(logits2, domain_label)
                domain_label = torch.ones(batch_size).long().cuda()

                #_, domain_output = net(image_patches, alpha)  
                #_, domain_output = self.decode_head(encoder_hidden_states, alpha)

                #_, domain_output = net(image_patches, alpha) 

                # outputs2 = self.segformer(
                #     pixel_values,
                #     output_attentions=output_attentions,
                #     output_hidden_states=True,  # we need the intermediate hidden states                     
                #     return_dict=return_dict,
                # )   

                outputs2 = self.segformer(
                    image_patches,
                    output_attentions=output_attentions,
                    output_hidden_states=True,  # we need the intermediate hidden states                            
                    return_dict=return_dict,
                )

                # outputs2 = self.segformer(
                #     pixel_values,
                #     output_attentions=output_attentions,
                #     output_hidden_states=True,  # # we need the intermediate hidden states                             
                #     return_dict=return_dict,
                # )           

                domain_output = outputs2.hidden_states
                _, domain_output = self.decode_head(domain_output, alpha)

                err_t_domain = nn.CrossEntropyLoss()(domain_output, domain_label)
                
                loss += err_s_domain + err_t_domain

                #loss += err_s_domain + err_t_domain

                #print(loss)
                #print(err_s_domain + err_t_domain)    
                
                #import pdb; pdb.set_trace() 
                
                #output2, oo222 = net(data2, alpha)    
                    
                # # # data2, target2   
                
                outputs3 = self.segformer(
                    data2,
                    output_attentions=output_attentions,
                    output_hidden_states=True,  # # we need the intermediate hidden states                            
                    return_dict=return_dict,
                )
                outputt22 = outputs3.hidden_states
                output2, logitss22 = self.decode_head(outputt22, alpha)
                upsampled_logitss22 = nn.functional.interpolate(
                    output2, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                #loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)   
                #losss22 = loss_fct(upsampled_logitss22, target2)          
                loss += loss_fct(upsampled_logitss22, target2)

                #loss += CrossEntropy2d(output2, target2, weight=weights)            
                
                #loss += smp.losses.JaccardLoss(mode="multiclass", classes=6)(y_pred=output2, y_true=target2) 

                #loss += smp.losses.JaccardLoss(mode="multiclass", classes=6)(y_pred=upsampled_logitss22, y_true=target2)
                loss += smp.losses.DiceLoss(mode="multiclass", classes=6)(y_pred=upsampled_logitss22, y_true=target2)

                #domain_label2 = (2*torch.ones(BATCH_SIZE).long()).cuda()         
                
                domain_label2 = (2*torch.ones(batch_size).long()).cuda()
                
                #loss += nn.CrossEntropyLoss()(oo222, domain_label2)                  
                
                loss += nn.CrossEntropyLoss()(logitss22, domain_label2)

                #import segmentation_models_pytorch as smp      
                #loss += smp.losses.JaccardLoss(mode="multiclass", classes=6)(y_pred=upsampled_logits, y_true=labels)

                #jaccard_distance_loss1 = JaccardDistanceLoss()                   
                #loss += jaccard_distance_loss1(labels, upsampled_logits)                                       

                # from torchmetrics import JaccardIndex    
                # #pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]                
                # #upsampled_logits = 1 - upsampled_logits
                # jaccard = JaccardIndex(num_classes=150)
                # loss += jaccard(upsampled_logits, labels)     
                
                #print(loss) 

                #loss += lovaszloss(upsampled_logits, labels)  
                
                #import segmentation_models_pytorch as smp 
                #loss += smp.losses.JaccardLoss(mode="multiclass", classes=151)(y_pred=upsampled_logits, y_true=labels)

                #import pdb; pdb.set_trace() 

                #print(loss)          
                
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
            
            else:
                raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )  

