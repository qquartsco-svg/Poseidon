"""
Marine Extended Kalman Filter (EKF).

4-state vector:  x = [x_m,  y_m,  psi_rad,  u_ms]ᵀ

Prediction step (nonlinear kinematics):
  x' = x + u·cos(ψ)·dt
  y' = y + u·sin(ψ)·dt
  ψ' = ψ + r·dt
  u' = u + (F_x / m_eff)·dt     (simplified thrust acceleration)

Linearised Jacobian F (4×4):
  F = I + dt · [[0, 0, -u·sin(ψ), cos(ψ)],
                [0, 0,  u·cos(ψ), sin(ψ)],
                [0, 0,  0,        0      ],
                [0, 0,  0,        0      ]]

Measurement updates:
  update_gps(z_xy)      — H = [[1,0,0,0],[0,1,0,0]]
  update_heading(z_psi) — H = [[0,0,1,0]]
  update_speed(z_u)     — H = [[0,0,0,1]]

All matrix arithmetic uses pure Python (lists of lists) — no numpy.

Reference:
  Bar-Shalom, Li, Kirubarajan (2001) "Estimation with Applications to
  Tracking and Navigation", Ch. 5 — Extended Kalman Filter
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Pure-Python matrix helpers
# ---------------------------------------------------------------------------

Matrix = List[List[float]]


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def _eye(n: int) -> Matrix:
    m = _zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def _mat_add(A: Matrix, B: Matrix) -> Matrix:
    """Element-wise addition A + B."""
    rows, cols = len(A), len(A[0])
    C = _zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] + B[i][j]
    return C


def _mat_sub(A: Matrix, B: Matrix) -> Matrix:
    """Element-wise subtraction A - B."""
    rows, cols = len(A), len(A[0])
    C = _zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] - B[i][j]
    return C


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication A × B."""
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    C = _zeros(rows_A, cols_B)
    for i in range(rows_A):
        for k in range(cols_A):
            if A[i][k] == 0.0:
                continue
            for j in range(cols_B):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _mat_transpose(A: Matrix) -> Matrix:
    """Transpose of A."""
    rows, cols = len(A), len(A[0])
    T = _zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            T[j][i] = A[i][j]
    return T


def _mat_scalar(A: Matrix, s: float) -> Matrix:
    """Multiply every element by scalar s."""
    rows, cols = len(A), len(A[0])
    C = _zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] * s
    return C


def _mat_inv2(A: Matrix) -> Matrix:
    """Invert a 2×2 matrix."""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-12:
        det = 1e-12
    inv_det = 1.0 / det
    return [
        [ A[1][1] * inv_det, -A[0][1] * inv_det],
        [-A[1][0] * inv_det,  A[0][0] * inv_det],
    ]


def _mat_inv1(A: Matrix) -> Matrix:
    """Invert a 1×1 matrix."""
    v = A[0][0] if abs(A[0][0]) > 1e-12 else 1e-12
    return [[1.0 / v]]


def _mat_inv(A: Matrix) -> Matrix:
    """Dispatch matrix inversion by size (1×1 or 2×2)."""
    n = len(A)
    if n == 1:
        return _mat_inv1(A)
    if n == 2:
        return _mat_inv2(A)
    raise ValueError(f"_mat_inv only supports 1×1 and 2×2; got {n}×{n}")


def _vec_to_col(v: List[float]) -> Matrix:
    """Convert flat list to n×1 column matrix."""
    return [[x] for x in v]


def _col_to_vec(M: Matrix) -> List[float]:
    """Convert n×1 column matrix to flat list."""
    return [row[0] for row in M]


def _normalize_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass
class MarineEKFNoise:
    """Process and measurement noise parameters.

    R_gps     — GPS position measurement variance (m²)
    R_heading — Heading (compass/gyro) measurement variance (rad²)
    R_speed   — Speed (log/GPS-SOG) measurement variance ((m/s)²)
    Q_diag    — Process noise diagonal [x, y, ψ, u]
    """

    R_gps: float = 2.0       # m²  per axis
    R_heading: float = 0.005  # rad²
    R_speed: float = 0.1      # (m/s)²
    Q_diag: tuple = (0.1, 0.1, 0.01, 0.05)


# ---------------------------------------------------------------------------
# EKF state snapshot (immutable output)
# ---------------------------------------------------------------------------

@dataclass
class StateEstimate:
    """EKF posterior state estimate.

    x_m, y_m   — position (m)
    psi_rad     — heading (rad)
    u_ms        — surge speed (m/s)
    P           — 4×4 covariance matrix
    t_s         — timestamp (s)
    """

    x_m: float = 0.0
    y_m: float = 0.0
    psi_rad: float = 0.0
    u_ms: float = 0.0
    P: Matrix = field(default_factory=lambda: _eye(4))
    t_s: float = 0.0


# ---------------------------------------------------------------------------
# EKF implementation
# ---------------------------------------------------------------------------

class MarineEKF:
    """4-state EKF for vessel position, heading and surge speed.

    State vector: x = [x_m, y_m, psi_rad, u_ms]ᵀ

    Typical usage::

        ekf = MarineEKF()
        ekf.predict(state, dt_s=0.1)
        ekf.update_gps([meas_x, meas_y])
        ekf.update_heading(meas_psi)
        ekf.update_speed(meas_u)
        est = ekf.estimate()
    """

    # State indices
    _IX = 0
    _IY = 1
    _IPSI = 2
    _IU = 3
    _N = 4

    def __init__(
        self,
        noise: Optional[MarineEKFNoise] = None,
        initial_state: Optional[List[float]] = None,
        m_eff: float = 5500.0,  # effective surge mass (kg)
    ) -> None:
        self._noise = noise or MarineEKFNoise()
        self._m_eff = m_eff

        # State mean
        if initial_state is not None:
            self._x: List[float] = list(initial_state)
        else:
            self._x = [0.0, 0.0, 0.0, 0.0]

        # Covariance (start somewhat uncertain)
        self._P: Matrix = _eye(self._N)
        for i in range(self._N):
            self._P[i][i] = 1.0

        self._t_s: float = 0.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        r_rads: float = 0.0,
        thrust_n: float = 0.0,
        dt_s: float = 0.1,
    ) -> None:
        """Propagate state and covariance forward by dt_s.

        Nonlinear prediction:
          x' = x + u·cos(ψ)·dt
          y' = y + u·sin(ψ)·dt
          ψ' = ψ + r·dt
          u' = u + (F_x / m_eff)·dt

        Jacobian F is evaluated at the current state.
        """
        x, y, psi, u = self._x

        # --- Nonlinear propagation ---
        a_u = thrust_n / self._m_eff
        x_new = x + u * math.cos(psi) * dt_s
        y_new = y + u * math.sin(psi) * dt_s
        psi_new = _normalize_angle(psi + r_rads * dt_s)
        u_new = u + a_u * dt_s

        self._x = [x_new, y_new, psi_new, u_new]

        # --- Jacobian (linearisation around current state) ---
        F = _eye(self._N)
        # ∂x'/∂ψ = -u·sin(ψ)·dt
        F[self._IX][self._IPSI] = -u * math.sin(psi) * dt_s
        # ∂x'/∂u =  cos(ψ)·dt
        F[self._IX][self._IU] = math.cos(psi) * dt_s
        # ∂y'/∂ψ =  u·cos(ψ)·dt
        F[self._IY][self._IPSI] = u * math.cos(psi) * dt_s
        # ∂y'/∂u =  sin(ψ)·dt
        F[self._IY][self._IU] = math.sin(psi) * dt_s

        # --- Process noise ---
        Q = _zeros(self._N, self._N)
        for i, q in enumerate(self._noise.Q_diag):
            Q[i][i] = q * dt_s

        # --- Covariance propagation: P' = F·P·Fᵀ + Q ---
        FP = _mat_mul(F, self._P)
        Ft = _mat_transpose(F)
        FPFt = _mat_mul(FP, Ft)
        self._P = _mat_add(FPFt, Q)
        self._t_s += dt_s

    # ------------------------------------------------------------------
    # Measurement updates
    # ------------------------------------------------------------------

    def update_gps(self, z_xy: List[float]) -> None:
        """Incorporate a GPS position fix.

        Measurement model: z = H·x + noise,  H = [[1,0,0,0],[0,1,0,0]]
        Measurement noise: R = diag(R_gps, R_gps)
        """
        H: Matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        R: Matrix = [
            [self._noise.R_gps, 0.0],
            [0.0, self._noise.R_gps],
        ]
        z_pred = [self._x[self._IX], self._x[self._IY]]
        innov = [z_xy[0] - z_pred[0], z_xy[1] - z_pred[1]]
        self._update(H, R, innov)

    def update_heading(self, z_psi: float) -> None:
        """Incorporate a compass / gyro heading measurement.

        Measurement model: z = H·x + noise,  H = [[0,0,1,0]]
        """
        H: Matrix = [[0.0, 0.0, 1.0, 0.0]]
        R: Matrix = [[self._noise.R_heading]]
        innov = [_normalize_angle(z_psi - self._x[self._IPSI])]
        self._update(H, R, innov)

    def update_speed(self, z_u: float) -> None:
        """Incorporate a speed-over-ground / speed-log measurement.

        Measurement model: z = H·x + noise,  H = [[0,0,0,1]]
        """
        H: Matrix = [[0.0, 0.0, 0.0, 1.0]]
        R: Matrix = [[self._noise.R_speed]]
        innov = [z_u - self._x[self._IU]]
        self._update(H, R, innov)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def estimate(self) -> StateEstimate:
        """Return current posterior estimate as an immutable snapshot."""
        import copy
        return StateEstimate(
            x_m=self._x[self._IX],
            y_m=self._x[self._IY],
            psi_rad=self._x[self._IPSI],
            u_ms=self._x[self._IU],
            P=copy.deepcopy(self._P),
            t_s=self._t_s,
        )

    def set_state(self, x_m: float, y_m: float, psi_rad: float, u_ms: float) -> None:
        """Hard-set the filter state (e.g. on initialisation from a fix)."""
        self._x = [x_m, y_m, psi_rad, u_ms]

    # ------------------------------------------------------------------
    # Internal EKF update kernel
    # ------------------------------------------------------------------

    def _update(self, H: Matrix, R: Matrix, innov: List[float]) -> None:
        """Generic EKF measurement update.

        K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
        x = x + K·innov
        P = (I - K·H)·P
        """
        Ht = _mat_transpose(H)
        PHt = _mat_mul(self._P, Ht)
        HPHt = _mat_mul(H, PHt)
        S = _mat_add(HPHt, R)
        S_inv = _mat_inv(S)
        K = _mat_mul(PHt, S_inv)

        # State update
        innov_col = _vec_to_col(innov)
        dx_col = _mat_mul(K, innov_col)
        dx = _col_to_vec(dx_col)
        for i in range(self._N):
            self._x[i] += dx[i]
        self._x[self._IPSI] = _normalize_angle(self._x[self._IPSI])

        # Covariance update: P = (I - K·H)·P
        KH = _mat_mul(K, H)
        I_KH = _mat_sub(_eye(self._N), KH)
        self._P = _mat_mul(I_KH, self._P)
