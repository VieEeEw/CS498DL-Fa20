import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits as bce_loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss_real = bce_loss(logits_real, torch.ones_like(logits_real), reduction='mean')
    loss_fake = bce_loss(1 - logits_fake, torch.ones_like(logits_fake), reduction='mean')

    return loss_real + loss_fake


def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = bce_loss(logits_fake, torch.ones_like(logits_fake), reduction='mean')

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss_real = mse_loss(scores_real, torch.ones_like(scores_real), reduction='mean')
    loss_fake = mse_loss(1 - scores_fake, torch.ones_like(scores_fake), reduction='mean')

    return loss_real + loss_fake


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = mse_loss(scores_fake, torch.ones_like(scores_fake), reduction='mean')

    return loss
