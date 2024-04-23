# mix tsmixerx & itransformer
__all__ = ['ModelMixer']
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.losses.pytorch import MAE
from neuralforecast.common._base_multivariate import BaseMultivariate
from itransformerm import iTransformer
from txmixerxm import TSMixerx, ReversibleInstanceNorm1d

class ModelMixer(BaseMultivariate):
    """
    ModelMixer
    """

    # Class attributes
    SAMPLING_TYPE = "multivariate"

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        revin=True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        optimizer=None,
        optimizer_kwargs=None,
        **trainer_kwargs
    ):

        # Inherit BaseMultvariate class
        super(ModelMixer, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            **trainer_kwargs
        )
        # Reversible InstanceNormalization layer
        self.revin = revin
        if self.revin:
            self.norm = ReversibleInstanceNorm1d(n_series=n_series)

        self.txmixer = TSMixerx(            
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list
        )

        self.itrans = iTransformer(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=None,
            hist_exog_list=None,
            stat_exog_list=None
        )

        self.out = nn.Linear(2*h, h, bias=True)
        

    def forward(self, windows_batch):
        # Parse batch
        x = windows_batch[
            "insample_y"
        ]  #   [batch_size (B), input_size (L), n_series (N)]
        hist_exog = windows_batch["hist_exog"]  #   [B, hist_exog_size (X), L, N]
        futr_exog = windows_batch["futr_exog"]  #   [B, futr_exog_size (F), L + h, N]
        stat_exog = windows_batch["stat_exog"]  #   [N, stat_exog_size (S)]
        batch_size, _ = x.shape[:2]

        # Add channel dimension to x
        x = x.unsqueeze(1)  #   [B, L, N] -> [B, 1, L, N]

        # Apply revin to x
        if self.revin:
            x = self.norm(x)  #   [B, 1, L, N] -> [B, 1, L, N]

        # apply txmixer
        x1 = x
        x1 = self.txmixer(x1, hist_exog, futr_exog, stat_exog) #  [B, 1, L, N] -> [B, h, N * n_outputs]
        # apply itrans
        x2 = x.squeeze(1)  # [B, 1, L, N] -> [B, L, N] 
        x2 = self.itrans(x2) # [B, L, N] -> [B, h, N]

        # concat x1, x2
        x = torch.cat((x1, x2), 1) # [B, 2h, N]
        x = x.permute(0, 2, 1) # [B, N, 2h]

        # Fully connected output layer
        x = self.out(x)  #   [B, N, 2h] -> [B, N, h]

        # Reverse Instance Normalization on output
        if self.revin:
            x = x.reshape(
                batch_size, self.h, self.loss.outputsize_multiplier, -1
            )  #   [B, h, N * n_outputs] -> [B, h, n_outputs, N]
            x = self.norm.reverse(x)
            x = x.reshape(
                batch_size, self.h, -1
            )  #   [B, h, n_outputs, N] -> [B, h, n_outputs * N]

        # Map to loss domain
        forecast = self.loss.domain_map(x)

        # domain_map might have squeezed the last dimension in case n_series == 1
        # Note that this fails in case of a tuple loss, but Multivariate does not support tuple losses yet.
        if forecast.ndim == 2:
            return forecast.unsqueeze(-1)
        else:
            return forecast
