# Implementation is adapted from https://github.com/ExplainableML/uot-fm/blob/main/train.py

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as sharding
import numpy as np
import optax
from tqdm import tqdm

from .mlp import MLP
from .utils import (
    BatchResampler,
    get_loss_builder,
    get_optimizer,
)

import torch
import functools as ft
    
def train(X_sampler, Y_sampler, config) -> eqx.Module:
    """Training script."""
    jax.config.update("jax_threefry_partitionable", True)
    # create rng keys
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)
    model_key, train_key, eval_key = jr.split(key, 3)
    # set up sharding
    num_devices = len(jax.devices())
    # shard needs to have same number of dimensions as the input
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    shard = sharding.PositionalSharding(devices)
    # get data
    batch_size = config.training.batch_size

    batch_resampler = BatchResampler(
        batch_size=batch_size,
        tau_a=config.training.tau_a,
        tau_b=config.training.tau_b,
        epsilon=config.training.epsilon,
    )
    # build model and optimization functions
    model = MLP(512, key=model_key)
    loss_builder = get_loss_builder(config)
    loss_fn = loss_builder.get_batch_loss_fn()
    opt = get_optimizer(config)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    train_step_fn = loss_builder.get_train_step_fn(loss_fn, opt.update)
    if config.optim.ema_decay > 0.0:
        assert config.optim.ema_decay < 1.0
        opt_ema = optax.ema(config.optim.ema_decay, debias=False)
        ema_state = opt_ema.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit(donate="all-except-first")
        def update_ema(curr_model, curr_ema_state):
            _, ema_state = opt_ema.update(eqx.filter(curr_model, eqx.is_array), curr_ema_state)
            return ema_state

    else:
        ema_state = None

    steps = config.training.num_steps
    print(
        f"Number of parameters: {sum(param.size for param in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))}"
    )
    total_train_loss = 0
    total_steps = 0
    for step in tqdm(range(steps), total=steps):
        train_key, resample_key = jr.split(train_key, 2)
        src_batch = X_sampler.sample(batch_size).cpu().numpy()
        tgt_batch = Y_sampler.sample(batch_size).cpu().numpy()
        src_batch, tgt_batch = jnp.array(src_batch), jnp.array(tgt_batch)
        src_batch, tgt_batch = batch_resampler(resample_key, src_batch, tgt_batch)
        # shard data
        src_batch, tgt_batch = jax.device_put([src_batch, tgt_batch], shard)
        # train step
        train_loss, model, train_key, opt_state = train_step_fn(
            model,
            tgt_batch,
            src_batch,
            train_key,
            opt_state,
        )
        if config.optim.ema_decay > 0.0:
            ema_state = update_ema(model, ema_state)
        total_train_loss += train_loss
        total_steps += 1
        if (step % config.training.print_freq) == 0 and step != 0 or step == steps - 1:
            # log train loss
            print(f"Step {step}, Loss: {total_train_loss.item() / total_steps}")
            total_train_loss = 0
            total_steps = 0
    return model

def evaluate(model: eqx.Module, X_test: torch.Tensor, config) -> torch.Tensor:
    """Test the model on the X_test shape: (N, 512)"""
    jax.config.update("jax_threefry_partitionable", True)
    # create rng keys
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)
    _, _, eval_key = jr.split(key, 3)
    # set up sharding
    num_devices = len(jax.devices())
    # shard needs to have same number of dimensions as the input
    devices = mesh_utils.create_device_mesh((num_devices, 1, 1, 1))
    shard = sharding.PositionalSharding(devices)

    loss_builder = get_loss_builder(config)
    sample_fn = loss_builder.get_sample_fn()
    inference_model = eqx.tree_inference(model, value=True)
    eval_key, _ = jr.split(eval_key, 2) # jax random key
    
    #### Sample from model ####
    # create vmap functions
    partial_sample_fn = ft.partial(sample_fn, inference_model)
    # compute metrics batch-wise
    device = X_test.device
    X_test = X_test.cpu().numpy()
    src_batch = jax.device_put(X_test, shard)
    sample_batch, _ = jax.vmap(partial_sample_fn)(src_batch)
    sample_batch = jnp.clip(sample_batch, -1.0, 1.0)
    # convert to tensor
    sample_batch = torch.tensor(np.array(sample_batch), device=device)
    return sample_batch
