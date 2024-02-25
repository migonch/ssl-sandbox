from typing import Any, Tuple
import itertools

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ssl_sandbox.nn.encoder import encoder, EncoderArchitecture
from ssl_sandbox.nn.blocks import MLP


class WassersteinReg(pl.LightningModule):
    def __init__(
            self,
            encoder_architecture: EncoderArchitecture = 'resnet50',
            projector_hidden_dim: int = 2048,
            projector_out_dim: int = 512,
            critic_hidden_dim: int = 1024,
            gp_weight: float = 10.0,
            num_critic_steps: int = 5,
            wasreg_weight: float = 1.0,
            lr: float = 3e-4,
            betas: Tuple[float, float] = (.5, .9),
            weight_decay: float = 1e-6,
            **hparams: Any  # will be dumped to yaml in logs folder
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder, self.embed_dim = encoder(encoder_architecture)
        self.projector = MLP(
            input_dim=self.embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_out_dim,
            num_hidden_layers=2,
            bias=False
        )
        self.critic = MLP(
            input_dim=projector_out_dim,
            hidden_dim=critic_hidden_dim,
            output_dim=1,
            num_hidden_layers=4,
            norm='none',
            bias=False
        )
        self.gp_weight = gp_weight
        self.num_critic_steps = num_critic_steps
        self.wasreg_weight = wasreg_weight
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        self.automatic_optimization = False

    def forward(self, images):
        return self.encoder(images)

    def _gradient_penalty(self, z, noise):
        alpha = torch.rand((len(z), 1), device=self.device)
        z_hat = alpha * z + (1 - alpha) * noise
        z_hat.requires_grad = True
        critic_outputs = self.critic(z_hat).sum()
        gradients = torch.autograd.grad(
            outputs=critic_outputs,
            inputs=z_hat,
            grad_outputs=torch.ones_like(critic_outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        return torch.mean(torch.relu(gradients.norm(2, dim=1) - 1.0) ** 2)

    def _critic_loss(self, z):
        noise = torch.randn_like(z)

        gp = self._gradient_penalty(z, noise)
        self.log('pretrain/critic_gp', gp, on_epoch=True, on_step=True)

        delta = self.critic(z).mean() - self.critic(noise).mean()
        self.log('pretrain/critic_delta', delta, on_epoch=True, on_step=True)

        loss = delta + self.gp_weight * gp
        self.log('pretrain/critic_loss', loss, on_epoch=True, on_step=True)

        return loss

    def training_step(self, batch, batch_idx):
        encoder_optimizer, critic_optimizer = self.optimizers()

        if batch_idx % self.num_critic_steps == 0:
            # train encoder on this iteration
            for p in self.encoder.parameters():
                p.requires_grad = True
            for p in self.projector.parameters():
                p.requires_grad = True

        (_, x_1, x_2), _ = batch

        z_1 = self.projector(self.encoder(x_1))  # (batch_size, proj_dim)

        critic_optimizer.zero_grad()
        critic_loss = self._critic_loss(z_1.detach())
        self.manual_backward(critic_loss)
        critic_optimizer.step()

        if batch_idx % self.num_critic_steps == 0:
            # train encoder on this iteration
            for p in self.critic.parameters():
                p.requires_grad = False

            # train encoder on this iteration
            z_2 = self.projector(self.encoder(x_2))  # (batch_size, proj_dim)

            i_reg = F.mse_loss(z_1, z_2)
            self.log(f'pretrain/i_reg', i_reg, on_epoch=True)

            was_reg = (-self.critic(z_1).mean() - self.critic(z_2).mean()) / 2.0
            self.log('pretrain/wasserstein_reg', was_reg, on_epoch=True, on_step=True)

            encoder_loss = i_reg + self.wasreg_weight * was_reg
            self.manual_backward(encoder_loss)
            encoder_optimizer.step()

            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.projector.parameters():
                p.requires_grad = False
            for p in self.critic.parameters():
                p.requires_grad = True

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        encoder_projector_params = itertools.chain(self.encoder.parameters(), self.projector.parameters())
        encoder_optimizer = torch.optim.AdamW(encoder_projector_params, lr=self.lr, betas=self.betas)
        critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr, betas=self.betas)
        return encoder_optimizer, critic_optimizer
