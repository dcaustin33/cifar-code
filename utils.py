import torch
import jax.numpy as jnp
import jax
from typing import Optional


def get_accuracy(predictions, labels):
    if len(predictions) > 0:
        _, predicted = torch.max(predictions, 1)
        acc1 = (predicted == labels).sum()

        _, pred = predictions.topk(5)
        labels = labels.unsqueeze(1).expand_as(pred)
        acc5 = (labels == pred).any(dim = 1).sum()
        return acc1, acc5
    return 0, 0


'''#taken from https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/sam_jax/training_utils/flax_training.py#L274
def top_k_error_rate_metric(logits: jnp.ndarray,
                            one_hot_labels: jnp.ndarray,
                            k: int = 5) -> jnp.ndarray:
    """Returns the top-K error rate between some predictions and some labels.
    Args:
        logits: Output of the model.
        one_hot_labels: One-hot encoded labels. Dimensions should match the logits.
        k: Number of class the model is allowed to predict for each example.
        mask: Mask to apply to the loss to ignore some samples (usually, the padding
        of the batch). Array of ones and zeros.
    Returns:
        The error rate (1 - accuracy), averaged over the first dimension (samples).
    """
    mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = 1 - ((hit * mask).sum() / mask.sum())
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)'''


#taken from https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/sam_jax/training_utils/flax_training.py#L274
@jax.jit
def top_5_error_rate_metric(logits: jnp.ndarray,
                            one_hot_labels: jnp.ndarray,
                            mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Returns the top-K error rate between some predictions and some labels.
    Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.
    k: Number of class the model is allowed to predict for each example.
    mask: Mask to apply to the loss to ignore some samples (usually, the padding
      of the batch). Array of ones and zeros.
    Returns:
    The error rate (1 - accuracy), averaged over the first dimension (samples).
    """
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -5:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = ((hit * mask).sum() / mask.sum())
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)

@jax.jit
def top_1_error_rate_metric(logits: jnp.ndarray,
                            one_hot_labels: jnp.ndarray,
                            mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Returns the top-K error rate between some predictions and some labels.
    Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.
    k: Number of class the model is allowed to predict for each example.
    mask: Mask to apply to the loss to ignore some samples (usually, the padding
      of the batch). Array of ones and zeros.
    Returns:
    The error rate (1 - accuracy), averaged over the first dimension (samples).
    """
    if mask is None:
        mask = jnp.ones([logits.shape[0]])
    mask = mask.reshape([logits.shape[0]])
    true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -1:]
    hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
    error_rate = ((hit * mask).sum() / mask.sum())
    # Set to zero if there is no non-masked samples.
    return jnp.nan_to_num(error_rate)