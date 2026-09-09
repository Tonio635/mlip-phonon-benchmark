"""Benchmark materials.
"""

from ase.build import bulk

MATERIALS = {
    "Si": dict(
        mp_id=149, formula="Si", structure="diamond",
        params=dict(a=5.431), spacegroup="Fd-3m", polar=False,
    ),
    "SiC": dict(
        mp_id=8062, formula="SiC", structure="zincblende",
        params=dict(a=4.358), spacegroup="F-43m", polar=True,
    ),
    "AlP": dict(
        mp_id=1550, formula="AlP", structure="zincblende",
        params=dict(a=5.463), spacegroup="F-43m", polar=True,
    ),
    "ZnS": dict(
        mp_id=10695, formula="ZnS", structure="zincblende",
        params=dict(a=5.409), spacegroup="F-43m", polar=True,
    ),
    "MgO": dict(
        mp_id=1265, formula="MgO", structure="rocksalt",
        params=dict(a=4.212), spacegroup="Fm-3m", polar=True,
    ),
    "NaCl": dict(
        mp_id=22862, formula="NaCl", structure="rocksalt",
        params=dict(a=5.640), spacegroup="Fm-3m", polar=True,
    ),
    "AlN": dict(
        mp_id=661, formula="AlN", structure="wurtzite",
        params=dict(a=3.111, c=4.981), spacegroup="P6_3mc", polar=True,
    ),
    "GaN": dict(
        mp_id=804, formula="GaN", structure="wurtzite",
        params=dict(a=3.189, c=5.185), spacegroup="P6_3mc", polar=True,
    ),
}

# Minimum subset for pipeline testing without wait
QUICK = ["Si", "MgO"]


def build_experimental(key: str):
    """Returns ase.Atoms object (primitive cell) for the material."""
    m = MATERIALS[key]
    return bulk(m["formula"], m["structure"], **m["params"])


def info(key: str) -> dict:
    return dict(MATERIALS[key], key=key)