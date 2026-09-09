"""Harmonic phonon calculation with any ASE calculator.

All the physics of the project is contained here and is identical for every model:
the run_* notebooks only change the calc object passed to compute_phonons().
This is what makes the comparison between models fair.

Workflow:

1. relax cell and atomic positions  -> locate a minimum of the PES
2. finite displacements (phonopy)   -> force constants
3. dynamical matrix                 -> frequencies
"""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.constraints import FixSymmetry
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

THZ_TO_CM1 = 33.35641
K_TO_THZ = 0.0208366191
GAMMA_ACOUSTIC_TOL_THZ = -50.0 * K_TO_THZ


def ase_to_ph(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def ph_to_ase(pa: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=pa.symbols,
        cell=pa.cell,
        scaled_positions=pa.scaled_positions,
        pbc=True,
    )


def relax(
    atoms: Atoms,
    calc,
    fmax: float = 0.005,
    steps: int = 500,
    logfile=None,
    fix_symmetry: bool = True,
) -> Atoms:
    """Relax the cell and atomic positions while preserving crystal symmetry.

    The initial structure is taken from the PBE reference. During MLIP
    relaxation, FixSymmetry prevents small numerical errors or force
    asymmetries from artificially breaking the space-group symmetry.

    fix_symmetry=True reproduces the protocol used by Loew et al.

    """

    atoms = atoms.copy()
    atoms.calc = calc

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    cell_filter = FrechetCellFilter(atoms)

    FIRE(
        cell_filter,
        logfile=logfile,
    ).run(
        fmax=fmax,
        steps=steps,
    )

    return atoms


def compute_phonons(
    atoms,
    calc,
    supercell_matrix,
    primitive_matrix,
    disp: float = 0.01,
    fmax: float = 0.005,
    do_relax: bool = True,
    symmetrize: bool = True,
    fix_symmetry: bool = True,
    logfile=None,
):
    """Optionally relax the structure, compute the force constants, and return (ph, atoms).

    ph    : Phonopy object with the force constants already computed
    atoms : the relaxed structure actually used

    """
    if do_relax:
        atoms = relax(
            atoms,
            calc,
            fmax=fmax,
            fix_symmetry=fix_symmetry,
            logfile=logfile,
        )
    else:
        atoms = atoms.copy()
        atoms.calc = calc

    ph = Phonopy(
        ase_to_ph(atoms),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix=np.asarray(primitive_matrix, dtype=float),
    )
    ph.generate_displacements(distance=disp, is_diagonal=False)

    forces = []

    for sc in ph.supercells_with_displacements:
        a = ph_to_ase(sc)
        a.calc = calc

        f = a.get_forces()

        # Remove residual translational drift, as in Loew et al.
        f = f - f.mean(axis=0, keepdims=True)

        forces.append(f)

    ph.forces = np.asarray(forces)
    
    ph.produce_force_constants()

    if symmetrize:
        ph.symmetrize_force_constants()

    return ph, atoms


def gamma_frequencies(ph) -> np.ndarray:
    """Gamma-point frequencies in THz, sorted."""
    ph.run_qpoints([[0, 0, 0]])
    return np.sort(ph.qpoints.frequencies[0])

def commensurate_frequencies(ph) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies at q-points commensurate with the supercell.

    These are the q-points actually represented by the finite-displacement
    calculation, before any Fourier interpolation onto dense meshes.

    Returns
    -------
    qpoints : (n_q, 3) ndarray
        Commensurate q-points.
    frequencies : (n_q, n_modes) ndarray
        Phonon frequencies in THz.
    """
    qpoints = np.asarray(
        get_commensurate_points(ph.supercell_matrix),
        dtype=float,
    )

    frequencies = np.asarray(
        [ph.get_frequencies(q) for q in qpoints],
        dtype=float,
    )

    return qpoints, frequencies


def max_frequency(ph) -> float:
    """Maximum frequency over the q-points commensurate with the supercell.

    This is the definition of omega_max used in the benchmark
    by Loew et al.; the interpolated 20x20x20 mesh is not used.

    """
    _, frequencies = commensurate_frequencies(ph)

    return float(np.max(frequencies))


def thermal_properties(ph, mesh=(20, 20, 20), t_min=0, t_max=1000, t_step=10):
    """Harmonic thermodynamic properties on a Gamma-centered mesh.

    Returns a dict containing T (K), F (kJ/mol), S (J/K/mol), and C_v (J/K/mol).
    The mesh must be Gamma-centered; otherwise, phonopy warns that the
    point-group symmetry is not preserved.

    """
    ph.run_mesh(list(mesh), is_gamma_center=True)
    ph.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    tp = ph.thermal_properties
    return dict(
        T=np.asarray(tp.temperatures),
        F=np.asarray(tp.free_energy),
        S=np.asarray(tp.entropy),
        C_v=np.asarray(tp.heat_capacity),
    )


def at_temperature(tp: dict, T: float = 300.0) -> dict:
    """Interpolate F, S, and C_v at a given temperature."""
    return {k: float(np.interp(T, tp["T"], tp[k])) for k in ("F", "S", "C_v")}


def make_band_path(ph_ref, npoints: int = 101) -> dict:
    """Generate the high-symmetry path from the PBE reference.

    The path is determined only once from the PBE DFT structure
    and must then be reused for both DFT and all MLIPs.
    """

    ph_ref.auto_band_structure(npoints=npoints)
    bs = ph_ref.band_structure

    return dict(
        qpoints=[
            np.asarray(q, dtype=float)
            for q in bs.qpoints
        ],
        distances=[
            np.asarray(d, dtype=float)
            for d in bs.distances
        ],
        labels=(
            list(bs.labels)
            if bs.labels is not None
            else None
        ),
        path_connections=list(bs.path_connections),
    )


def band_structure(ph, band_path: dict) -> dict:
    """Compute the dispersion along an externally defined path.

    band_path must be obtained from the PBE reference via
    make_band_path(). The MLIP frequencies are therefore evaluated
    at the same reduced q-points used for the DFT reference.

    The distance axis is taken from the PBE reference, so the overlays
    share exactly the same x-axis.
    """

    bs = ph.run_band_structure(
        band_path["qpoints"],
        path_connections=band_path["path_connections"],
        labels=band_path["labels"],
    )

    return dict(
        qpoints=[
            np.asarray(q, dtype=float)
            for q in bs.qpoints
        ],

        # Important: common PBE-reference x axis
        distances=[
            np.asarray(d, dtype=float)
            for d in band_path["distances"]
        ],

        frequencies=[
            np.asarray(f, dtype=float)
            for f in bs.frequencies
        ],

        labels=band_path["labels"],
        path_connections=band_path["path_connections"],
    )


def has_imaginary(
    ph,
    gamma_acoustic_tol: float = GAMMA_ACOUSTIC_TOL_THZ,
    zero_tol: float = 1e-8,
) -> bool:
    """True if the structure is dynamically unstable according to Loew et al.

    Stability is evaluated exclusively at q-points commensurate
    with the supercell.

    Away from Gamma:
        all frequencies must be real (>= 0, within numerical noise).

    At Gamma:
        the three acoustic modes may be negative down to -50 K
        (~ -1.042 THz). All other modes must be non-negative.

    """

    qpoints, frequencies = commensurate_frequencies(ph)

    for q, freq in zip(qpoints, frequencies):

        is_gamma = np.allclose(
            q,
            [0.0, 0.0, 0.0],
            atol=1e-8,
        )

        freq = np.sort(freq)

        if is_gamma:
            # The three lowest modes are the acoustic branches.
            acoustic = freq[:3]
            optical = freq[3:]

            if np.any(acoustic < gamma_acoustic_tol):
                return True

            if np.any(optical < -zero_tol):
                return True

        else:
            if np.any(freq < -zero_tol):
                return True

    return False


def convergence_test(
    atoms,
    calc,
    primitive_matrix,
    sizes=(2, 3, 4),
    disp=0.01,
    fmax=0.005,
    logfile=None,
) -> dict:
    """Diagnostic test of convergence with respect to supercell size.

    In the finite-displacement method, the supercell size determines
    the maximum distance over which harmonic interactions are represented
    explicitly. Increasing the supercell size therefore makes it possible
    to assess how sensitive the phonon frequencies are to the spatial
    truncation of the force constants.

    The test relaxes the PBE unit cell only once and repeats the phonon
    calculation using isotropic n×n×n supercells for the values specified
    in sizes.

    The primitive_matrix is kept identical to that of the PBE DFT
    reference.

    This test is purely diagnostic: it is NOT used to select a single
    global supercell for the final benchmark. In the MACE, CHGNet, and
    M3GNet runs, the ph_ref.supercell_matrix specific to the
    corresponding PBE reference is always reused, following the protocol
    of Loew et al.

    Parameters
    ----------
    atoms
        ASE Atoms object obtained from the PBE reference unit cell.

    calc
        ASE calculator for the MLIP.

    primitive_matrix
        Primitive matrix of the PBE reference.

    sizes
        Values of n for the isotropic n×n×n supercells to be tested.

    disp
        Finite-displacement amplitude in Å.

    fmax
        Relaxation convergence threshold in eV/Å.

    logfile
        Output file for the ASE optimizer.

    Returns
    -------
    dict
        For each supercell size n, contains the supercell matrix, Gamma-point
        frequencies, omega_max over the commensurate q-points, number of atoms,
        number of displacements, and runtime.

    """

    out = {}

    # Rilassiamo una sola volta la cella iniziale PBE
    relaxed = relax(
        atoms,
        calc,
        fmax=fmax,
        fix_symmetry=True,
        logfile=logfile,
    )

    for n in sizes:
        t0 = time.perf_counter()

        supercell_matrix = np.diag([n, n, n])

        ph, _ = compute_phonons(
            relaxed,
            calc,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            disp=disp,
            do_relax=False,
            logfile=logfile,
        )

        g = gamma_frequencies(ph)

        out[n] = dict(
            supercell_matrix=supercell_matrix.tolist(),

            gamma=g.tolist(),
            omega_max_gamma=float(g.max()),

            omega_max=max_frequency(ph),

            n_atoms_supercell=len(ph.supercell),
            n_displacements=len(ph.forces),

            runtime_s=round(
                time.perf_counter() - t0,
                2,
            ),
        )

    return out


def is_dynamically_stable(ph) -> bool:
    return not has_imaginary(ph)