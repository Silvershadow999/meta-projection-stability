from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class NoisySignificanceConfig:
    """
    Generator config for a noisy external human-context / significance-like proxy.

    Notes:
    - This is an external proxy signal generator for simulation experiments.
    - It is NOT the same as the adapter's internal human_significance anchor.
    """
    base_value: float = 0.85
    ou_theta: float = 0.05          # mean-reversion strength
    ou_sigma: float = 0.08          # diffusion / volatility
    spike_prob: float = 0.005       # probability of rare spike per step
    spike_mag: Tuple[float, float] = (-0.4, 0.3)  # asymmetric spikes (drops stronger)
    floor: float = 0.0
    ceiling: float = 1.0

    def __post_init__(self) -> None:
        self.floor = float(min(self.floor, self.ceiling))
        self.ceiling = float(max(self.floor, self.ceiling))

        self.base_value = float(np.clip(self.base_value, self.floor, self.ceiling))
        self.ou_theta = max(0.0, float(self.ou_theta))
        self.ou_sigma = max(0.0, float(self.ou_sigma))
        self.spike_prob = float(np.clip(self.spike_prob, 0.0, 1.0))

        # Normalize spike bounds ordering
        low, high = self.spike_mag
        low_f = float(low)
        high_f = float(high)
        self.spike_mag = (min(low_f, high_f), max(low_f, high_f))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NoisySignificance:
    """
    Ornstein-Uhlenbeck-like noisy proxy generator with rare spikes.

    Use this as an external simulation input (e.g. human context / load / significance proxy),
    not as a replacement for the adapter's internal anchor state.
    """

    def __init__(self, config: NoisySignificanceConfig | None = None, seed: int = 42):
        self.config = config or NoisySignificanceConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.mu = float(self.config.base_value)       # long-term mean
        self.current = float(self.config.base_value)  # current state
        self.step_count = 0

    def reset(self, value: float | None = None) -> None:
        """
        Reset generator state to base_value or a provided clipped value.
        """
        if value is None:
            self.current = float(self.config.base_value)
        else:
            self.current = float(np.clip(value, self.config.floor, self.config.ceiling))
        self.mu = float(self.config.base_value)
        self.step_count = 0
        self.rng = np.random.default_rng(self.seed)

    def set_mean(self, mu: float) -> None:
        """
        Update long-term mean target (clipped to bounds).
        Useful for regime changes in long-horizon simulations.
        """
        self.mu = float(np.clip(mu, self.config.floor, self.config.ceiling))

    def step(self, dt: float = 1.0) -> float:
        """
        Advance one step and return the clipped proxy value in [floor, ceiling].
        """
        dt = max(1e-9, float(dt))

        # OU-like dynamics
        drift = self.config.ou_theta * (self.mu - self.current) * dt
        diffusion = self.config.ou_sigma * np.sqrt(dt) * float(self.rng.normal())
        self.current = float(self.current + drift + diffusion)

        # Rare spikes (stress/peak events)
        if float(self.rng.random()) < self.config.spike_prob:
            mag = float(self.rng.uniform(*self.config.spike_mag))
            self.current = float(self.current + mag)

        self.current = float(np.clip(self.current, self.config.floor, self.config.ceiling))
        self.step_count += 1
        return self.current

    def sample(self, n: int, dt: float = 1.0) -> np.ndarray:
        """
        Generate n samples as a numpy array.
        """
        n = max(0, int(n))
        if n == 0:
            return np.array([], dtype=float)
        out = np.empty(n, dtype=float)
        for i in range(n):
            out[i] = self.step(dt=dt)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_count": int(self.step_count),
            "current": float(self.current),
            "mu": float(self.mu),
            "config": self.config.to_dict(),
        }


if __name__ == "__main__":
    cfg = NoisySignificanceConfig()
    gen = NoisySignificance(cfg, seed=42)
    vals = gen.sample(10)
    print("NoisySignificance quick demo:")
    print([round(float(v), 4) for v in vals])
    print("State:", gen.state_dict())
