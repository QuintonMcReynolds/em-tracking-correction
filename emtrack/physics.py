"""Forward model of an AC electromagnetic tracker with eddy-current / ferromagnetic distortion.

Frames and units
  world  = cassette frame, millimetres. The field generator (3 orthogonal transmitter coils) sits at the origin.
  body   = handheld X-ray emitter frame; the beam points along -z_body.
  A 3-axis receiver with orientation R at position p measures S (3x3): column j is the field of transmitter coil j
  expressed in the receiver's coil axes, S[:, j] = R^T B_j(p).

Field model (normalised; constants absorbed): a dipole m at the origin produces
  B(r) = (3 r_hat (m . r_hat) - m) / |r|^3, with r in metres.
Metal object k at q_k with polarisability tensor A_k (m^3) acquires an induced moment m_k = A_k B_j(q_k) for each
transmitter coil j and re-radiates a secondary dipole field. First-order (Born) model: no metal-metal coupling.
  conductive sphere, radius a  : A = -(a^3 / 2) I          (eddy currents oppose the field)
  conductive plate, normal n   : A = -c b^3 n n^T           (thin disc/table top, responds to the normal component)
  ferromagnetic rod, axis u    : A = +c L^3 u u^T           (steel rails, tools; field-enhancing)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MM = 1e-3  # mm -> m


def dipole_field(r_mm: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Field at points r (..., 3) [mm] from dipole moment(s) m (..., 3) located at the origin."""
    r = r_mm * MM
    d = np.linalg.norm(r, axis=-1, keepdims=True)
    rh = r / d
    return (3 * rh * np.sum(m * rh, -1, keepdims=True) - m) / d**3


TX = np.eye(3)  # the three transmitter coil moments (columns)


@dataclass
class Metal:
    kind: str                  # sphere | plate | rod
    pos: np.ndarray            # (3,) mm
    A: np.ndarray              # (3,3) m^3, polarisability tensor
    size_mm: float

    def to_dict(self):
        return {"kind": self.kind, "pos": self.pos.round(1).tolist(), "size_mm": round(self.size_mm, 1),
                "A": self.A.round(8).tolist()}


@dataclass
class Environment:
    env_id: str
    family: str
    metals: list[Metal] = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))  # in cassette frame

    def to_dict(self):
        return {"env_id": self.env_id, "family": self.family, "gravity": self.gravity.round(5).tolist(),
                "metals": [m.to_dict() for m in self.metals]}


def primary(p_mm: np.ndarray) -> np.ndarray:
    """(N,3) positions -> (N,3,3): column j = field of transmitter coil j at p (world axes)."""
    return np.stack([dipole_field(p_mm, np.broadcast_to(TX[:, j], p_mm.shape)) for j in range(3)], -1)


def secondary(p_mm: np.ndarray, env: Environment) -> np.ndarray:
    """(N,3,3) field from all metal objects at positions p, same layout as primary()."""
    out = np.zeros(p_mm.shape[:-1] + (3, 3))
    for mt in env.metals:
        Bq = primary(mt.pos[None])[0]                        # (3,3) primary field at the metal, per coil
        m_ind = mt.A @ Bq                                    # (3,3) induced moments, column per coil
        rel = p_mm - mt.pos
        for j in range(3):
            out[..., :, j] += dipole_field(rel, np.broadcast_to(m_ind[:, j], rel.shape))
    return out


def measure(p_mm: np.ndarray, R: np.ndarray, env: Environment | None, rng: np.random.Generator | None = None,
            rel_noise: float = 2e-3) -> np.ndarray:
    """Receiver reading S = R^T (B_primary + B_secondary) + noise. p (N,3), R (N,3,3) -> (N,3,3)."""
    B = primary(p_mm)
    if env is not None and env.metals:
        B = B + secondary(p_mm, env)
    S = np.einsum("nji,njk->nik", R, B)
    if rng is not None and rel_noise > 0:
        scale = np.linalg.norm(S, axis=(1, 2), keepdims=True) / 3
        S = S + rng.normal(0, rel_noise, S.shape) * scale
    return S
