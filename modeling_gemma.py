import torch
from torch import nntplib
from typing import Optional, Tuple, List
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionModel, SiglipVisionConfig


class GemmaConfig():
    def __init__(self,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 num_key_value_heads,
                 head_dim=256,
                 max_position_embeddings=8192,
                 rms_norm_eps=1e-6,
                 rope_theta=10000.0,
                 attention_bias=False,
                 attention_dropout=0.0,
                 pad_token_id=None,
                 **kwargs,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


 class PaliGemmaConfig():
     def __init__(self,
                  vision_config: None,
                  text_config= None,
                  ignore_index = -100,
                  image_token_index = 256000,
                  vocab_size = 257152,
                  projection_dim = 2048,
                  hidden_size = 2048,
                  pad_token_id = None,
                  **kwargs,
                  ):
         super().__init__()
         self.ignore_index = ignore_index
         self.image_token_index = image_token_index
         self.vocab_size = vocab_size
         self.projection_dim = projection_dim
         self.hidden_size = hidden_size
         self.pad_token_id = pad_token_id
         self.vision_config = vision_config
         self.is_encoder_decoder = False

         self.vision_config = SiglipVisionConfig(**vision_config)
         self.text_config = text_config

         self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
         self.vocab_size = self.text_config.vocab_size

         self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
         self.vision_config.projection_dim = projection_dim



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        """
            Tie the input token embedding weights and output projection weights of the language model.

            This method calls `tie_weights()` on the underlying `language_model` to enforce parameter sharing
            between the token embedding layer and the final output (logit) layer.

            Weight tying reduces the number of parameters in the model and improves generalization by ensuring
            that the same representations used to encode input tokens are also used when projecting hidden states
            back to vocabulary logits for prediction. This technique is commonly used in transformer-based language
            models and is known to improve training stability and performance.

            Returns:
                None
            """
        return self.language_model.tie_weights()


    def _merge_input_ids_with_image_features(self,
                                             image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                             kv_cache: Optional[KVCache] = None):

        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Shape [batch_size, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # Combine the embeddings of the iamge tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # The below are just the masks that tell which element in the final_embedding are image and text and padding. For example,
        # If final_embedding is [img_token1, img_token2, img_token3, bos_token, txt_token1, txt_token_2, \n],
        # Then text_mask will be [0,0,0,1,1,1,1], img_mask will be [1,1,1,0,0,0,0], pad_token will be [0,0,0,0,0,0,0]. No padding for now.
        # shape (batch_size, seq_len). True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

        # shape (batch_size, seq_len), True for image tokens
        image_mask = input_ids == self.config.image_token_index

        # shape (batch_size, seq_len). True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we cant use them in torch.
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings to the placeholder
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We cant use torch.where because the seq length of the scaled_image_features is not the same as
        # the seq length of the final image embeddings
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)



    def forward(self,
                 input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,) ->Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings
        # shape: (batch_Size, seq_len, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        # We get the contexualized image embeddings here. This is the embeddings after going through the different attention layers
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, hidden_size]
        # The below is done by using the multi_modal_project which is basically a linear layer.
        # This layer is added so that the vision embeddings are made into the same shape as that of the language encoder output.
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        # inputs_embeds already contain the placeholders for the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)


        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs



