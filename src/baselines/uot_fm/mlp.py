import math
from collections.abc import Callable
from typing import Optional, Union

from einops import rearrange
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb

class MLP(eqx.Module):
    time_pos_emb: SinusoidalPosEmb
    mlp: eqx.nn.MLP

    def __init__(self, data_shape: tuple[int, int], *, key):
        keys = jax.random.split(key, 7)
        del key
        t_emb_dim = 512
        self.time_pos_emb = SinusoidalPosEmb(t_emb_dim)

        self.mlp = eqx.nn.MLP(
            in_size=512,
            out_size=512,
            width_size=512,
            depth=2,
            activation=jax.nn.relu,
            key=keys[0],
        )

    def __call__(self, t, x_t, *, key=None):
        # Ensure t has shape [batch]
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 1:
            t = t  # shape [batch]
        elif t.ndim > 1:
            raise ValueError("t should be a scalar or 1D array")

        t_emb = self.time_pos_emb(t)  # shape [batch, 512]
        x = x_t + t_emb
        return self.mlp(x)