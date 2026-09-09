"""DFT phonon references from the Alexandria phonon benchmark.

The benchmark of Loew et al. provides two DFT datasets:

- PBE: main reference used to benchmark the uMLIPs.
- PBEsol: original MDR reference, retained as a secondary comparison
  to estimate the sensitivity to the exchange-correlation functional.

Files are provided directly by Materials Project ID:
    mp-149.yaml.bz2
    mp-1265.yaml.bz2
    ...

By default PBE is always used.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import phonopy

from src.paths import REFERENCE


BASE_URL = "https://alexandria.icams.rub.de/data/phonon_benchmark"

ALEXANDRIA = REFERENCE / "alexandria"

VALID_FUNCTIONALS = {"pbe", "pbesol"}


def _validate_functional(functional: str) -> str:
    functional = functional.lower()

    if functional not in VALID_FUNCTIONALS:
        raise ValueError(
            f"Unknown functional {functional!r}. "
            f"Expected one of {sorted(VALID_FUNCTIONALS)}."
        )

    return functional


def reference_path(
    mp_id: int,
    functional: str = "pbe",
) -> Path:
    """Local path of an Alexandria phonon reference."""

    functional = _validate_functional(functional)

    return (
        ALEXANDRIA
        / functional
        / f"mp-{int(mp_id)}.yaml.bz2"
    )


def download_reference(
    mp_id: int,
    functional: str = "pbe",
    force: bool = False,
) -> Path:
    """Download one Alexandria phonon reference."""

    functional = _validate_functional(functional)

    path = reference_path(mp_id, functional)

    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    url = (
        f"{BASE_URL}/{functional}/"
        f"mp-{int(mp_id)}.yaml.bz2"
    )

    try:
        urllib.request.urlretrieve(url, path)

    except urllib.error.HTTPError as e:
        if path.exists():
            path.unlink()

        if e.code == 404:
            raise KeyError(
                f"mp-{mp_id} not present in the Alexandria "
                f"{functional.upper()} phonon dataset."
            ) from e

        raise

    except Exception:
        if path.exists():
            path.unlink()
        raise

    return path


def load_reference(
    mp_id: int,
    functional: str = "pbe",
    use_nac: bool = False,
    expect: str | None = None,
):
    """Load a DFT reference as a Phonopy object.

    Parameters
    ----------
    mp_id
        Materials Project numerical ID.

    functional
        "pbe" (default) or "pbesol".

        PBE is the main benchmark reference used by Loew et al.
        PBEsol is retained only as a secondary functional comparison.

    use_nac
        Whether to retain the non-analytical correction (LO-TO splitting).
        False by default because the MLIPs do not provide Born effective
        charges, so NAC must also be disabled in the DFT reference for a
        consistent comparison.

    expect
        Optional element symbol used as a simple sanity check.
    """

    path = download_reference(
        mp_id,
        functional=functional,
    )

    ph = phonopy.load(
        str(path),
        produce_fc=True,
        is_nac=use_nac,
    )

    if expect is not None:
        symbols = set(ph.unitcell.symbols)

        if expect not in symbols:
            raise ValueError(
                f"mp-{mp_id} contains {sorted(symbols)}, "
                f"not expected element {expect!r}."
            )

    return ph