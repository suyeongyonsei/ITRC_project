import torch
from torch import nn
from torch.nn import functional as F

def sample_noise(batch_size, noise_size):
    noise = (-2)*torch.rand((batch_size, noise_size)) +1
    return noise

def get_discriminator(input_size, hidden_dim):

    model = nn.Sequential(
      nn.Linear(input_size, hidden_dim),
      nn.LeakyReLU(0.01),
      nn.Linear(hidden_dim, hidden_dim),
      nn.LeakyReLU(0.01),
      nn.Linear(hidden_dim, 1))
    return model


def get_generator(noise_size, hidden_dim, input_size):

    model = nn.Sequential(
      nn.Linear(noise_size, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, input_size),
      nn.Tanh()
    )
    return model


def discriminator_loss(logits_real, logits_fake):
    true_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    true_loss = F.binary_cross_entropy_with_logits(logits_real, true_labels)
    fake_loss = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    loss = true_loss + fake_loss
    return loss

def generator_loss(logits_fake):

    fake_labels = torch.ones_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    return loss
