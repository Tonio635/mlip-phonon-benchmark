"""Saving and reloading results using a fixed schema.

Each calculation produces a JSON file in data/results/<model>/<material>.json.
The schema is identical for both models and the DFT reference: the reporting
notebook reads them all in the same way, so the comparison reduces to a subtraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.paths import RESULTS


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def result_path(model: str, material: str) -> Path:
    d = RESULTS / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{material}.json"


def save_result(model: str, material: str, payload: dict) -> Path:
    p = result_path(model, material)
    p.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return p


def load_result(model: str, material: str) -> dict:
    return json.loads(result_path(model, material).read_text(encoding="utf-8"))


def load_all(models=("mace", "chgnet", "m3gnet", "dft")) -> dict:
    """{model: {material: payload}} for everything that has already been computed."""
    out = {}
    for m in models:
        d = RESULTS / m
        if not d.is_dir():
            continue
        out[m] = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted(d.glob("*.json"))}
    return out


def make_payload(
    material: str,
    model: str,
    model_version: str,
    dtype: str,
    ph,
    atoms,
    disp,
    fmax,
    tp,
    bands,
    runtime_s=None,
    notes=None,
) -> dict:

    import numpy as np

    from src.phonons import (
        at_temperature,
        gamma_frequencies,
        has_imaginary,
        max_frequency,
    )

    return dict(
        material=material,
        model=model,
        model_version=model_version,
        dtype=dtype,

        supercell_matrix=np.asarray(
            ph.supercell_matrix,
            dtype=int,
        ).tolist(),

        primitive_matrix=np.asarray(
            ph.primitive_matrix,
            dtype=float,
        ).tolist(),

        displacement=disp,
        fmax=fmax,

        n_atoms_supercell=len(ph.supercell),
        n_displacements=len(ph.forces),

        lattice=(
            atoms.cell.cellpar().tolist()
            if atoms is not None
            else None
        ),

        volume=(
            float(atoms.get_volume())
            if atoms is not None
            else None
        ),

        gamma_frequencies_THz=gamma_frequencies(ph).tolist(),

        omega_max_THz=max_frequency(ph),

        has_imaginary=has_imaginary(ph),

        thermal=dict(
            T=tp["T"],
            F=tp["F"],
            S=tp["S"],
            C_v=tp["C_v"],
        ),

        thermal_300K=at_temperature(tp, 300.0),

        bands=dict(
            qpoints=bands["qpoints"],
            distances=bands["distances"],
            frequencies=bands["frequencies"],
            labels=bands["labels"],
            path_connections=bands["path_connections"],
        ),

        runtime_s=runtime_s,
        notes=notes,
    )