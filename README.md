# mlip-phonon-benchmark

Benchmark of harmonic phonons calculated with pretrained universal machine-learning interatomic potentials.

The project reproduces the methodology of the following work on a subset of materials:

**A. Loew, D. Sun, H.-C. Wang, S. Botti, M. A. L. Marques,
“Universal machine learning interatomic potentials are ready for phonons”,
npj Computational Materials 11, 178 (2025).**

Exam project for *Molecular Modeling of Materials*.

## Objective

Universal machine-learning interatomic potentials are trained primarily on DFT energies, forces, and stresses. Phonon frequencies, however, depend on the curvature of the potential energy surface, i.e. on the second derivatives of the energy with respect to atomic coordinates.

The main question is therefore:

> Are models trained on energies and forces sufficiently accurate to predict vibrational properties as well?

Three pretrained potentials are compared:

* **MACE-MP-0**
* **CHGNet**
* **M3GNet MP-2021.2.8-EFS**

No training or fine-tuning is performed: all models are used for inference with frozen weights.

## Materials

The reduced benchmark contains eight simple crystals:

| Material |    MP ID | Structure  |
| -------- | -------: | ---------- |
| Si       |   mp-149 | diamond    |
| SiC      |  mp-8062 | zincblende |
| AlP      |  mp-1550 | zincblende |
| ZnS      | mp-10695 | zincblende |
| MgO      |  mp-1265 | rocksalt   |
| NaCl     | mp-22862 | rocksalt   |
| AlN      |   mp-661 | wurtzite   |
| GaN      |   mp-804 | wurtzite   |

## DFT reference

The main reference is the **PBE dataset from the Alexandria phonon benchmark** used by Loew et al.

For each material, the Phonopy files identified by the Materials Project ID are downloaded directly.

The **PBEsol** collection is retained only as a secondary comparison, useful for assessing the sensitivity of phonon frequencies to the choice of exchange-correlation functional.

The comparison hierarchy is therefore:

```text
MLIP  ───────────────→  PBE
                        ↑
                 main reference

PBEsol ──────────────→  PBE
                        ↑
                functional effect
```

## Calculation protocol

The pipeline follows the protocol of the original benchmark as closely as possible.

### 1. Initial structure

Each MLIP calculation starts directly from the **PBE unit cell of the DFT reference**.

The hard-coded experimental lattice parameters in `materials.py` are not used for production runs.

### 2. Relaxation

The structure is relaxed with ASE using:

* `FIRE`
* `FrechetCellFilter`
* `FixSymmetry`
* `fmax = 0.005 eV/Å`

`FixSymmetry` prevents small numerical errors in the ML potential from artificially breaking the initial space group.

### 3. Supercell

For each material, the following are reused exactly:

```python
ph_ref.supercell_matrix
ph_ref.primitive_matrix
```

from the corresponding PBE reference.

The supercell is therefore not chosen globally as `2x2x2`: different materials may use different matrices.

### 4. Finite displacements

The harmonic force constants are obtained with Phonopy using:

```text
displacement = 0.01 Å
is_diagonal = False
```

For each displaced supercell:

1. the MLIP calculates the forces;
2. the residual mean force is subtracted, removing translational drift;
3. the force constants are constructed;
4. the force constants are symmetrized.

### 5. Maximum frequency and dynamical stability

The maximum frequency `omega_max` is not searched for on a dense interpolated mesh.

As in the original benchmark, it is evaluated exclusively at the **q-points commensurate with the supercell**.

Dynamical stability is evaluated at the same q-points.

At Gamma, a small negative frequency is allowed only for the three acoustic modes, down to the `-50 K` tolerance used in the paper.

### 6. Thermodynamic properties

The vibrational properties are instead calculated on a:

```text
20 x 20 x 20
```

mesh and include:

* vibrational Helmholtz free energy;
* vibrational entropy;
* constant-volume heat capacity `C_v`.

### 7. Phonon dispersions

The high-symmetry path is generated **only once from the PBE reference**.

The same q-points are then used for:

```text
PBE
MACE
CHGNet
M3GNet
```

so that the curves can be overlaid directly.

The PBE path distances are used as the common axis in graphical comparisons.

### 8. LO-TO correction

For polar materials, the DFT reference may contain Born effective charges and the non-analytical correction responsible for LO-TO splitting.

The three MLIPs do not provide these quantities.

To maintain a consistent comparison, the reference is therefore loaded with:

```python
use_nac=False
```

## Project structure

```text
src/
    paths.py
        standard project paths

    materials.py
        metadata for the eight materials and MP IDs

    reference.py
        download and loading of the Alexandria PBE/PBEsol references

    phonons.py
        relaxation, finite displacements, force constants,
        commensurate q-points, thermodynamics, and dispersions

    io_utils.py
        serialization of results to JSON

notebooks/
    00_method_validation.ipynb
        methodological checks and sanity checks

    01_run_mace.ipynb
        MACE result production

    02_run_chgnet.ipynb
        CHGNet result production

    03_run_m3gnet.ipynb
        M3GNet result production

    04_report.ipynb
        DFT/MLIP comparison, figures, and final discussion

data/
    reference/
        alexandria/
            pbe/
            pbesol/

    results/
        mace/
        chgnet/
        m3gnet/

figures/
    figures exported by the report

requirements/
    mace.txt
    chgnet.txt
    m3gnet.txt
```

## Python environments

The three models are run in separate virtual environments.

This avoids conflicts between dependencies and numerical dtypes, particularly between MACE, PyTorch, and the legacy TensorFlow stack required by M3GNet.

Example:

```bash
python -m venv .venv-mace
pip install -r requirements/mace.txt
```

Separate environments are created analogously for CHGNet and M3GNet.

### M3GNet

To reproduce the model used in the paper, the original TensorFlow implementation is used:

```text
m3gnet == 0.2.4
TensorFlow == 2.15.1
Keras == 2.15.0
model = MP-2021.2.8-EFS
```

Small compatibility patches are included to run the legacy package with recent ASE/Pymatgen versions on Windows.

Correct checkpoint loading was validated against the official Mo benchmark:

```text
a_relaxed ≈ 3.169 Å
E ≈ -10.859 eV/atom
```

## Output

Each run produces a file:

```text
data/results/<model>/<material>.json
```

containing, among other things:

* model and version;
* numerical precision;
* `supercell_matrix`;
* `primitive_matrix`;
* relaxed lattice;
* number of displaced supercells;
* Gamma-point frequencies;
* `omega_max`;
* dynamical stability;
* thermodynamic properties;
* q-points and dispersion frequencies;
* runtime.

This allows the reporting notebook to be run without recomputing the models.

## Note on supercell convergence

`00_method_validation.ipynb` includes a diagnostic `2x2x2`, `3x3x3`, `4x4x4` test on Si.

This test is used to illustrate the effect of spatial truncation of the force constants in the finite-displacement method.

It is not used, however, to select a single global supercell: in the final runs, the `supercell_matrix` specific to the PBE reference is always used, in accordance with the original benchmark.

## License

See `LICENSE`.
