"""
We will used pytorch lightning to train the models.

Those models will be trained using the matching flow methodology (close to diffusion models).
Also we will use a modify version of matching to enable fast sampling.

"""

import lightning as L
import torch
import torch.nn.functional as F

class RewardGuidanceModel(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, init_states: torch.Tensor, future_states: torch.Tensor, time_flags: torch.Tensor) -> torch.Tensor:
        return self.model(init_states, future_states, time_flags)

    def training_step(self, batch, batch_idx):
        init_states, future_states, time_flags, rewards = batch
        pass
    
    def validation_step(self, batch, batch_idx):
        init_states, future_states, time_flags, rewards = batch
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
