"""
We will used pytorch lightning to train the models.

Those models will be trained using the matching flow methodology (close to diffusion models).
Also we will use a modify version of matching to enable fast sampling.

"""

import lightning as L
import torch
import torch.nn.functional as F

from torch.distributions.dirichlet import Dirichlet

cuda_available = torch.cuda.is_available()
DEVICE = "cuda" if cuda_available else "cpu"


class PLRewardGuidanceModel(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        # dirichlet distribution parameter
        self.dirichlet_alpha = 0.5
        self.nb_dim_dirichlet = 6

        self.dirichlet_dist = Dirichlet(
            torch.tensor([self.dirichlet_alpha] * self.nb_dim_dirichlet)
        )

    def forward(
        self,
        init_states: torch.Tensor,
        future_states: torch.Tensor,
        time_flags: torch.Tensor,
        shortcut_value: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(init_states, future_states, time_flags, shortcut_value)

    def training_step(self, batch, batch_idx):
        """
        Training step for the reward guidance model.

        We will use a matching flow setup to train the model.


        We will use a time flags and a shortcut value to enable fast sampling.


        In batch we have init_states, future_states and rewards.
            init_states is of shape (batch_size, nb_init_seq, 9*6, 6)
            future_states is of shape (batch_size, nb_future_seq, 9*6, 6)
            rewards is of shape (batch_size, 1)
        """
        init_states, future_states, rewards = (
            batch["state_past"],
            batch["state_future"],
            batch["reward"],
        )

        batch_size = init_states.shape[0]

        # here we should generate a time flags
        time_flags = torch.rand(batch_size, 1).to(DEVICE)

        # here we should generate a shortcut value but for the first batch_size * 3 / 4
        # we set it to 0 and the rest random between 0 and 1
        shortcut_value = torch.zeros(batch_size, 1).to(DEVICE)

        # create a noisy future_states (matching flow setup)
        pur_noise = self.dirichlet_dist.sample(
            sample_shape=(batch_size, future_states.shape[1], 9 * 6)
        ).to(DEVICE)

        future_states_noisy = future_states * time_flags.unsqueeze(-1).unsqueeze(
            -1
        ) + pur_noise * (1.0 - time_flags.unsqueeze(-1).unsqueeze(-1))

        # forward pass
        state_final_logits = self.forward(
            init_states, future_states_noisy, time_flags, shortcut_value
        )

        target = future_states - pur_noise

        # output_target = torch.zeros_like(target)

        # compute the loss for the reward and the state_final_logits for
        # the shortcut value 0
        output_target = (
            torch.softmax(state_final_logits, dim=-1) - 1.0 / self.nb_dim_dirichlet
        )

        loss_regularization = F.mse_loss(output_target, target)

        # log the loss
        self.log("train_loss", loss_regularization)

        return loss_regularization

    def generate(self, nb_batch, nb_iter=100.0, init_states=None):
        """
        Method to generate samples from the model.
        """
        dirichlet_dist_inference = Dirichlet(
            torch.tensor([5.0] * self.nb_dim_dirichlet)
        )

        with torch.inference_mode():
            # we start from time flag 0
            time_flags = torch.zeros(nb_batch, 1).to(DEVICE)

            # we start from shortcut value 0
            shortcut_value = torch.zeros(nb_batch, 1).to(DEVICE)

            # we generate the future states using dirichlet distribution
            future_states = dirichlet_dist_inference.sample(
                sample_shape=(nb_batch, self.model.nb_future_states, 9 * 6)
            ).to(DEVICE)

            # loop with nb_iter
            for i in range(int(nb_iter)):
                # forward pass
                time_flags = torch.ones(nb_batch, 1).to(DEVICE) * float(i) / nb_iter
                noisy_state_final_logits = self.forward(
                    init_states, future_states, time_flags, shortcut_value
                )

                # from logit to probabilities
                velocity_final = (
                    torch.softmax(noisy_state_final_logits, dim=-1)
                    - 1.0 / self.nb_dim_dirichlet
                )

                future_states = future_states + velocity_final * 1.0 / nb_iter

        return future_states

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
