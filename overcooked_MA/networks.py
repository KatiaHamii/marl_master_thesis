"""
networks.py — Improved actor-critic for OvercookedV2 (7-channel compact obs)
=============================================================================

Problem with the original network (overcooked_v2_rebuild/networks.py):
  The 7-channel compact obs feeds raw bit-packed integers directly into
  convolutions.  Channels like inventory (values: 4, 8, 16, 12, …) and
  dynamic items are not in [0, 1], causing gradient spikes and a plateau.

Fix: ObsPreprocessor decodes each channel into proper binary / normalised
form BEFORE the spatial encoder.  The resulting ~51-channel binary obs is
clean input for the Conv stack.

Decoded channel layout (51 total):
  static_type  (16)  — one-hot over StaticObject values 0-15
  self_dir     ( 5)  — one-hot: 0=no-agent, 1=UP, 2=DOWN, 3=RIGHT, 4=LEFT
  other_dir    ( 5)  — same for teammate
  self_inv     ( 8)  — bit-decomposition of inventory bit-pack
  other_inv    ( 8)  — bit-decomposition of teammate inventory
  dynamic      ( 8)  — bit-decomposition of on-grid dynamic item
  extra        ( 1)  — pot timer / recipe, normalised to [0, 1]
  ─────────────────
  total        (51)

Architecture:
  (H, W, 7) → preprocess → (H, W, 51)
  → Conv(32,3×3) + ReLU
  → Conv(64,3×3) + ReLU
  → Conv(64,3×3) + ReLU
  → Flatten → Dense(256) + LayerNorm + ReLU → Dense(128) + LayerNorm + ReLU
  → actor: Dense(n_actions)   [logits]
  → critic: Dense(1)          [value]

No GRU for full-observability mode.  Pass use_rnn=True to restore the
GRU(128) recurrent head for partial-observability experiments.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import numpy as np


# ── Constants (must match settings.py / common.py) ────────────────────────────
_N_STATIC   = 16    # max StaticObject enum value + 1
_N_DIR      = 5     # 0 = no agent, 1-4 = directions
_N_BITS     = 8     # bits to extract from each bit-packed channel
_MAX_EXTRA  = 64.0  # safe upper bound for pot-timer / recipe normalisation


# ── Observation preprocessor ──────────────────────────────────────────────────

class SimpleObsNormalizer(nn.Module):
    """
    Minimal preprocessing: just normalize raw values to [0, 1].
    Keeps obs shape as (H, W, 7) and lets network learn its own encoding.
    """

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        obs : (H, W, 7)  float32 — raw compact observation
        returns (H, W, 7) float32 — normalized to [0, 1]
        """
        # Normalize each channel to [0, 1]
        # ch0, ch2: directions (0-5) → divide by 5
        # ch1, ch3: inventories (0-16) → divide by 16
        # ch4: static objects (0-15) → divide by 15
        # ch5: dynamic items (0-255) → divide by 255
        # ch6: extra (0-64) → divide by 64

        normalized = jnp.concatenate([
            obs[..., 0:1] / 5.0,      # ch0: direction
            obs[..., 1:2] / 16.0,     # ch1: inventory
            obs[..., 2:3] / 5.0,      # ch2: other direction
            obs[..., 3:4] / 16.0,     # ch3: other inventory
            obs[..., 4:5] / 15.0,     # ch4: static type
            obs[..., 5:6] / 255.0,    # ch5: dynamic items
            obs[..., 6:7] / 64.0,     # ch6: timer/recipe
        ], axis=-1)
        return normalized


class ObsPreprocessor(nn.Module):
    """
    Converts the raw 7-channel compact obs into 51 binary / normalised channels.
    All output values are in [0, 1].
    """

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        obs : (H, W, 7)  float32 — raw compact observation
        returns (H, W, 51) float32 — clean binary/normalised encoding
        """
        # ch4 — static cell type → one-hot (16)
        static_oh = jax.nn.one_hot(
            obs[..., 4].astype(jnp.int32).clip(0, _N_STATIC - 1),
            _N_STATIC,
        )

        # ch0, ch2 — direction (0-4) → one-hot (5) each
        dir_self  = jax.nn.one_hot(obs[..., 0].astype(jnp.int32).clip(0, 4), _N_DIR)
        dir_other = jax.nn.one_hot(obs[..., 2].astype(jnp.int32).clip(0, 4), _N_DIR)

        # ch1, ch3, ch5 — bit-packed integers → 8 binary channels each
        # def _bits(x: chex.Array) -> chex.Array:
        #     x = x.astype(jnp.int32)
        #     return jnp.stack([(x >> i) & 1 for i in range(_N_BITS)], axis=-1).astype(jnp.float32)
        
        def _bits(x: chex.Array) -> chex.Array:
            x = x.astype(jnp.int32)[..., None] # add new dimension (H, W, 1)
            shifts = jnp.arange(_N_BITS)       # [0, 1, 2, 3, 4, 5, 6, 7]
            return ((x >> shifts) & 1).astype(jnp.float32) # Automatic broadcasting to (H, W, 8)

        inv_self  = _bits(obs[..., 1])   # (H, W, 8)
        inv_other = _bits(obs[..., 3])   # (H, W, 8)
        dyn       = _bits(obs[..., 5])   # (H, W, 8)
        
        

        # ch6 — pot timer + recipe → scalar in [0, 1]
        extra = (obs[..., 6] / _MAX_EXTRA).clip(0.0, 1.0)[..., None]  # (H, W, 1)

        return jnp.concatenate(
            [static_oh, dir_self, dir_other, inv_self, inv_other, dyn, extra],
            axis=-1,
        )  # (H, W, 51)


# ── Dense encoder (rl4mapf style) ──────────────────────────────────────────────

class DenseEncoder(nn.Module):
    """
    Simple dense network that flattens the observation (like rl4mapf).
    No spatial structure exploitation — just learn from flat features.
    Input: (H, W, C) — any shape
    Output: (128,) feature vector
    """

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        # Flatten spatial dimensions
        #x = x.reshape(-1)  # (H*W*C,) flat
        x = x.reshape((x.shape[:-3] + (-1,)))

        # Dense layers
        x = nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        return x  # (128,)


# ── Spatial encoder ────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Three-layer 3×3 Conv stack, followed by two Dense+LN layers.
    Input: (H, W, C_in) — the 51-channel preprocessed obs.
    Output: (256,) feature vector.
    """

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME",
                            kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME",
                            kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME",
                            kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
        #x = x.reshape(-1)
        x = x.reshape((x.shape[:-3] + (-1,)))

        x = nn.relu(nn.LayerNorm()(
            nn.Dense(256, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        ))
        x = nn.relu(nn.LayerNorm()(
            nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        ))
        return x   # (128,)


# ── Recurrent wrapper (optional) ──────────────────────────────────────────────

GRU_HIDDEN = 128


class GRUEncoder(nn.Module):
    """Wraps a flat feature vector with a GRU cell. For partial-obs experiments."""
    hidden_size: int = GRU_HIDDEN

    @nn.compact
    def __call__(
        self,
        features: chex.Array,   # (128,)
        hidden:   chex.Array,   # (hidden_size,)
    ) -> Tuple[chex.Array, chex.Array]:
        new_hidden, out = nn.GRUCell(self.hidden_size)(hidden, features)
        return out, new_hidden  # (hidden_size,), (hidden_size,)


# ── Actor-Critic ───────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Actor-Critic for OvercookedV2 IPPO.

    Call signature differs depending on use_rnn:

    use_rnn=False (default — full observability):
      logits, value = net(obs)
      obs    : (H, W, 7)
      logits : (n_actions,)
      value  : ()

    use_rnn=True (partial observability):
      logits, value, new_hidden = net(obs, hidden)
      hidden     : (gru_hidden,)
      new_hidden : (gru_hidden,)
    """
    n_actions:  int
    use_rnn:    bool = False
    gru_hidden: int  = GRU_HIDDEN
    obs_mode:   str  = "rich"  # "rich" (ObsPreprocessor) or "simple" (SimpleObsNormalizer)

    @nn.compact
    def __call__(
        self,
        obs:    chex.Array,
        hidden: Optional[chex.Array] = None,
    ):
        # Choose observation preprocessing and encoder
        if self.obs_mode == "simple":
            # Simple rl4mapf-style: minimal normalization + dense
            x = SimpleObsNormalizer()(obs)   # (H, W, 7) normalized
            x = DenseEncoder()(x)             # Flatten + dense layers
        else:
            # Rich: full preprocessing + spatial convolutions
            x = ObsPreprocessor()(obs)       # (H, W, 51) fully decoded
            x = ConvEncoder()(x)             # Conv spatial encoder

        if self.use_rnn:
            assert hidden is not None, "hidden required when use_rnn=True"
            x, new_hidden = GRUEncoder(self.gru_hidden)(x, hidden)

        logits = nn.Dense(
            self.n_actions,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(x).squeeze(-1)

        if self.use_rnn:
            return logits, value, new_hidden
        return logits, value

    @staticmethod
    def init_hidden(batch: int = 1, gru_hidden: int = GRU_HIDDEN) -> chex.Array:
        h = jnp.zeros((gru_hidden,))
        return jnp.tile(h[None], (batch, 1)) if batch > 1 else h
