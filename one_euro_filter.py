"""
One Euro Filter — a low-pass filter for real-time noisy signal streams.

An exponential-smoothing filter whose cut-off frequency adapts to signal speed,
giving both low jitter at rest **and** low lag during fast movements.

Reference:
    Géry Casiez, Nicolas Roussel, Daniel Vogel.
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input
    in Interactive Systems." CHI 2012.
    https://hal.inria.fr/hal-00670496/document
"""

import numpy as np


def smoothing_factor(t_e: float, cutoff: np.ndarray) -> np.ndarray:
    """
    Compute the exponential smoothing factor α from the time step
    and cut-off frequency.

    α = r / (r + 1)  where  r = 2π · fc · Δt
    """
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(
    a: np.ndarray,
    x: np.ndarray,
    x_prev: np.ndarray,
) -> np.ndarray:
    """
    Simple exponential smoothing:  ŷ = α·x + (1−α)·x_prev
    """
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """
    Vectorised 1€ Filter that works on scalars *and* NumPy arrays.

    Parameters
    ----------
    t0 : float
        Timestamp (seconds or frame index) of the first sample.
    x0 : float | np.ndarray
        First observed value(s).
    dx0 : float
        Initial derivative estimate (default 0).
    min_cutoff : float
        Minimum cut-off frequency.  Lower → more smoothing at rest.
    beta : float
        Speed coefficient.  Higher → less lag during fast motion.
    d_cutoff : float
        Cut-off frequency for the derivative filter.
    """

    def __init__(
        self,
        t0: float,
        x0,
        dx0: float = 0.0,
        min_cutoff: float = 0.004,
        beta: float = 0.7,
        d_cutoff: float = 1.0,
    ):
        x0 = np.asarray(x0, dtype=np.float64)
        self.min_cutoff = np.ones_like(x0) * min_cutoff
        self.beta = np.ones_like(x0) * beta
        self.d_cutoff = np.ones_like(x0) * d_cutoff

        self.x_prev = x0.copy()
        self.dx_prev = np.ones_like(x0) * dx0
        self.t_prev = float(t0)

    def __call__(self, t: float, x) -> np.ndarray:
        """Return the filtered value for the new sample *x* at time *t*."""
        x = np.asarray(x, dtype=np.float64)
        t_e = t - self.t_prev

        if t_e <= 0:
            # Avoid division by zero; just return x unfiltered
            return x

        # Filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # Adaptive cut-off frequency driven by signal speed.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Filtered signal.
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Store state for next call.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
