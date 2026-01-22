"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, latent_size, dropout=0.0, layer_norm_eps=1e-12
    ):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size, eps=layer_norm_eps),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)


class TIGER(T5ForConditionalGeneration):
    """
    A wrapper class for T5ForConditionalGeneration that adds extra functionality while preserving core T5 behavior.
    """

    def __init__(
        self,
        config: T5Config,
        n_semantic_codebook: int,
        max_items_per_seq: int,
        flag_use_learnable_text_embed: bool = False,
        flag_use_output_embedding: bool = False,
        embedding_head_dict: dict = None,
    ):

        self.flag_use_output_embedding = flag_use_output_embedding
        self.embedding_head_dict = embedding_head_dict

        super().__init__(config)

        if flag_use_learnable_text_embed:
            if embedding_head_dict["embed_proj_type"] == "mlp":
                self.emb_proj = MLP(
                    embedding_head_dict["text_embedding_dim"],
                    embedding_head_dict["hidden_sizes"][::-1],
                    config.d_model,
                    dropout=embedding_head_dict["embd_proj_in_dropout_rate"],
                    layer_norm_eps=config.layer_norm_epsilon,
                )
            elif embedding_head_dict["embed_proj_type"] == "linear":
                self.emb_proj = nn.Linear(
                    embedding_head_dict["text_embedding_dim"], config.d_model
                )
            else:
                raise ValueError(
                    f"Invalid embedding projection type: {embedding_head_dict['embed_proj_type']}"
                )
            self.input_embed_dropout = nn.Dropout(
                p=embedding_head_dict["embd_proj_dropout_rate"]
            )
            self.input_embed_layernorm = nn.LayerNorm(
                config.d_model, eps=config.layer_norm_epsilon
            )

        if "text_embedding_dim" in embedding_head_dict:
            if embedding_head_dict["embed_proj_type"] == "mlp":
                self.context_proj = MLP(
                    embedding_head_dict["text_embedding_dim"],
                    embedding_head_dict["hidden_sizes"][::-1],
                    config.d_model,
                    dropout=embedding_head_dict["embd_proj_in_dropout_rate"],
                    layer_norm_eps=config.layer_norm_epsilon,
                )
            elif embedding_head_dict["embed_proj_type"] == "linear":
                self.context_proj = nn.Linear(
                    embedding_head_dict["text_embedding_dim"], config.d_model
                )
            else:
                raise ValueError(
                    f"Invalid embedding projection type: {embedding_head_dict['embed_proj_type']}"
                )
        else:
            self.context_proj = None

        self.n_semantic_codebook = n_semantic_codebook
        self.semantic_pos = nn.Embedding(n_semantic_codebook + 1, config.d_model)
        self.pos_embedding = nn.Embedding(max_items_per_seq, config.d_model)

        # Cross-attention layer for context fusion
        self.context_cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.context_cross_attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        if embedding_head_dict["use_new_init"]:
            # this is only applied for dense retrieval
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        initializer_range = factor
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        encoder_attention_mask = attention_mask
        
        # Process per-position input context tokens (new feature)
        input_context_tokens = getattr(self, "input_context_tokens", None)
        if input_context_tokens is not None:
            # input_context_tokens shape: [batch_size, seq_len, d_model]
            # hidden_states shape: [batch_size, seq_len, d_model]
            input_context_mask = getattr(self, "input_context_mask", None)
            
            # Check if batch sizes match (they won't match during beam search expansion)
            # During beam search, batch_size gets multiplied by num_beams
            # Skip context application in this case as it was already applied in initial encoding
            if input_context_tokens.shape[0] != hidden_states.shape[0]:
                # Batch size mismatch (likely beam search) - skip context application
                self.input_context_tokens = None
                self.input_context_mask = None
            elif input_context_tokens.shape[:2] != hidden_states.shape[:2]:
                # If shapes don't match, we need to align them
                # This can happen if encoder processes multiple codebooks per item
                batch_size, hidden_seq_len, d_model = hidden_states.shape
                context_seq_len = input_context_tokens.shape[1]
                
                # Calculate the expansion factor (e.g., n_codebook for SID mode)
                if hidden_seq_len % context_seq_len == 0:
                    expansion_factor = hidden_seq_len // context_seq_len
                    # Expand context tokens to match hidden states length
                    # Each context token is repeated for its corresponding codebook positions
                    expanded_context = input_context_tokens[:, :, None, :].repeat(
                        1, 1, expansion_factor, 1
                    ).reshape(batch_size, hidden_seq_len, d_model)
                    
                    # Expand mask similarly
                    if input_context_mask is not None:
                        expanded_mask = input_context_mask[:, :, None].repeat(
                            1, 1, expansion_factor
                        ).reshape(batch_size, hidden_seq_len)
                    else:
                        expanded_mask = None
                    
                    # Add context to hidden states (only for valid positions)
                    if expanded_mask is not None:
                        hidden_states = hidden_states + expanded_context * expanded_mask[:, :, None]
                    else:
                        hidden_states = hidden_states + expanded_context
                else:
                    # Fallback: if we can't align, just add to matching positions
                    min_len = min(hidden_seq_len, context_seq_len)
                    if input_context_mask is not None:
                        hidden_states[:, :min_len] = (
                            hidden_states[:, :min_len] + 
                            input_context_tokens[:, :min_len] * input_context_mask[:, :min_len, None]
                        )
                    else:
                        hidden_states[:, :min_len] = hidden_states[:, :min_len] + input_context_tokens[:, :min_len]
            else:
                # Shapes match, directly add context to hidden states
                if input_context_mask is not None:
                    hidden_states = hidden_states + input_context_tokens * input_context_mask[:, :, None]
                else:
                    hidden_states = hidden_states + input_context_tokens
            
            # Clean up
            self.input_context_tokens = None
            self.input_context_mask = None
        
        # Process single context token via cross-attention (for label context)
        context_token = getattr(self, "context_token", None)
        if context_token is not None:
            if context_token.dim() == 2:
                context_token = context_token[:, None, :]
            if context_token.dim() != 3:
                raise ValueError("context_token must have shape [batch, 1, d_model].")
            if context_token.shape[2] != hidden_states.shape[2]:
                raise ValueError("context_token shape mismatch with encoder outputs.")
            if context_token.shape[0] != hidden_states.shape[0]:
                repeat_factor = hidden_states.shape[0] // context_token.shape[0]
                if hidden_states.shape[0] % context_token.shape[0] != 0:
                    raise ValueError("context_token batch mismatch with encoder outputs.")
                context_token = context_token.repeat_interleave(repeat_factor, dim=0)
            
            # Cross-attention: sequence attends to context (Q=seq, K=V=context)
            attn_out, _ = self.context_cross_attn(
                query=hidden_states,
                key=context_token,
                value=context_token,
            )
            hidden_states = self.context_cross_attn_norm(hidden_states + attn_out)
            
            if return_dict:
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
            else:
                encoder_outputs = (hidden_states,) + encoder_outputs[1:]
            self.context_token = None
        elif input_context_tokens is not None:
            # If we added per-position context but no label context, still update encoder_outputs
            if return_dict:
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
            else:
                encoder_outputs = (hidden_states,) + encoder_outputs[1:]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        self.predicted_embedding = None
        if self.flag_use_output_embedding:
            if encoder_attention_mask is None:
                raise ValueError("attention_mask is required for predicted_embedding.")
            item_seq_len = encoder_attention_mask.sum(-1)
            predicted_embedding = self.gather_indexes(hidden_states, item_seq_len - 1)
            self.predicted_embedding = predicted_embedding

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
