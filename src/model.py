import torch
import torch.nn as nn

from transformer import (
    TimeSeriesTransformer,
    TimeSeriesTransformerConfig
)

class ProbabilisticTransformer(nn.Module):
    def __init__(
            self,
            input_size,
            context_length=96,
            prediction_length=24,
            d_model=64,
            n_head=4,
            encoder_layers=2,
            decoder_layers=2,
    ):
        super().__init__()
        self.config = TimeSeriesTransformerConfig(
            input_size=input_size,

            context_length=context_length,
            prediction_length=prediction_length,

            d_model=d_model,

            encoder_attention_heads=n_head,
            decoder_attention_heads=n_head,

            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,

            lags_sequence=[1, 2, 3, 24],
        )

        self.transformer = TimeSeriesTransformer(self.config)

    def forward(
        self, 
        pass_values,
        past_time_features,
        past_observed_mask,
        feature_values=None,
        future_time_features=None
        ):

        output = self.transformer(
            pass_values=pass_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            feature_values=feature_values,
            future_time_features=future_time_features
        )
        return output
               
        