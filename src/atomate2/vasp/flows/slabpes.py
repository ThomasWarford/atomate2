"""
Module defining SurfPES flows.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, Flow, job, OnMissing
from pymatgen.io.vasp.sets import SlabPESStaticSet
import numpy as np
import h5py

from pathlib import Path
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.utils.path import strip_hostname
from pymatgen.io.vasp import Outcar, Vasprun, Chgcar, Locpot
from monty.os.path import zpath
from monty.serialization import loadfn
from ase.io import read, write
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import covalent_radii

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker
from atomate2.common.jobs.utils import remove_workflow_files
from atomate2.common.utils import _recursive_get_dir_names


if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from collections.abc import Sequence

# functions that can be used outside slabpes, but need to be here for atomate to be happy
def get_dipol(structure):
    weights = [site.species.weight for site in structure]
    center_of_mass = np.average(structure.cart_coords, weights=weights, axis=0).tolist()
    return center_of_mass

def estimate_dipole(structure_with_charge):
    dipol = get_dipol(structure_with_charge)

    # get positions - dipol
    positions_minus_dipol = structure_with_charge.cart_coords - dipol
    # get charges
    charges = structure_with_charge.site_properties['charge']
    charges = np.array(charges)[:, None] # for broadcasting
    if (charges is not None) and not (None in charges):
        return (charges * positions_minus_dipol).sum(axis=0)
    else:
        return None

def get_structure_with_charge(atoms, mpid_to_average_charges_path):
    mpid_to_average_charge = loadfn(mpid_to_average_charges_path)
    average_charges = mpid_to_average_charge[atoms.info['bulk_mpid']]
    charges = list(map(average_charges.get, atoms.get_chemical_symbols()))

    structure_with_charge = AseAtomsAdaptor.get_structure(atoms)
    structure_with_charge.add_site_property('charge', charges)
    return structure_with_charge
    
def estimate_dipole_from_oxidation_states(atoms):
    try:
        struct = get_structure_with_charge(atoms, '/home/s5f/twarf.s5f/.dft/SurfPES/mp_data/mpid_to_average_oxidation_states.json.gz') # TODO: remove this hack
        return estimate_dipole(struct)
    except KeyError:
        return None

def bulk_mpid_to_bulk_bandgap(bulk_mpid):
    try:
        with MPRester() as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[bulk_mpid],
                fields=["material_id", "band_gap"]
            )
        return docs[0].band_gap 
    except Exception as e:
        print("Error getting bulk bandgap:")
        print(e)
        return None
    
def add_tags_from_input_dct(input_dct, atoms):
    try:
        tags = np.array(input_dct['job'].function_args[0].site_properties['tags'])
        atoms_copy = atoms.copy()
        atoms_copy.set_tags(tags)

        atoms_copy.info['num_adsorbate_atoms'] = sum(tags == 2)

        adsorbate_forces = atoms_copy.arrays['ref_forces'][tags == 2]
        net_adsorbate_force = adsorbate_forces.sum(axis=0)
        atoms_copy.info['net_adsorbate_force'] = net_adsorbate_force
        

        return atoms_copy
    except:
        print('Failed to add tags')
        return atoms

def get_volumetric_data_properties(workdir):
    def get_grad(data, lattice_vectors):
        grad = np.stack(np.gradient(data, *(1/n for n in data.shape)),axis=0)
        return np.einsum('ij, j...->i...', np.linalg.inv(lattice_vectors), grad)

    def get_smoothness_score(data, lattice_vectors):
        grad = get_grad(data, lattice_vectors)
        grad_squared_integral = np.abs(np.linalg.det(lattice_vectors)) * (grad**2).mean()
        return grad_squared_integral
    
    properties = {}
    
    charge_data = {}

    # add AECCARs
    for i in range(3):
        aeccar = Chgcar.from_file(zpath(workdir/f'AECCAR{i}'))
        charge = aeccar.data['total'] # work with total charge
        charge = np.swapaxes(charge, 0, 2) # make z axis first
        charge_data[f'aeccar{i}'] = charge

    with h5py.File(workdir/"vaspwave.h5", "r") as f: 
        chgcar = np.array(f['charge']['charge'])
        charge_data['chgcar'] = chgcar[0]
        grid = np.array(f['charge']['grid']) # NOTE: grid != chgcar.shape!! Weird from vasp

        charge_data['spin_density'] = chgcar[1]

        lattice_vectors =  np.array(f['structure']['positions']['lattice_vectors'])
    cell_volume = np.abs(np.linalg.det(lattice_vectors))

    past_shape = None
    for label, data in charge_data.items():
        if past_shape:
            np.testing.assert_equal(data.shape, past_shape)
        past_shape = data.shape

        properties[f'{label}_sum'] = data.sum() / np.prod(grid)
        properties[f'{label}_smoothness'] = get_smoothness_score(data/(np.prod(grid) * cell_volume), lattice_vectors)


    properties['aeccar2_minus_aeccar1_sum'] = properties['aeccar2_sum'] - properties['aeccar1_sum']
    properties['chgcar_minus_aeccar1_sum'] = properties['chgcar_sum'] - properties['aeccar1_sum']
    properties['aeccar2_minus_aeccar1_smoothness'] = get_smoothness_score((charge_data['aeccar2']-charge_data['aeccar1'])/(np.prod(grid) * cell_volume), lattice_vectors)
    properties['chgcar_minus_aeccar1_smoothness'] = get_smoothness_score((charge_data['chgcar']-charge_data['aeccar1'])/(np.prod(grid) * cell_volume), lattice_vectors)

    # read vaspout
    with h5py.File(workdir/"vaspout.h5", "r") as f:
        num_valence_electrons = np.array(f['results']["electron_eigenvalues"]["nelectrons"])

        hartree = np.array(f['results']["potential"]["hartree"])[0]
        ionic = np.array(f['results']["potential"]["ionic"])[0]
        xc = np.array(f['results']["potential"]["xc"])
        v_xc = xc[0]
        b_xc = xc[1]
        total = np.array(f['results']["potential"]["total"])
        v_total = total[0]
        b_total = total[1]

        np.testing.assert_equal(hartree.shape, ionic.shape)
        np.testing.assert_equal(hartree.shape, v_xc.shape)
        np.testing.assert_equal(hartree.shape, b_xc.shape)
        np.testing.assert_equal(hartree.shape, v_total.shape)
        np.testing.assert_equal(hartree.shape, b_total.shape)

    # add potential properties
    potential_smoothness_targets = {
        'v_hartree': hartree,
        'v_ionic': ionic,
        'v_xc': v_xc,
        'b_xc': b_xc,
        'v_total': v_total,
        'b_total': b_total,
    }
    for label, data in potential_smoothness_targets.items():
        properties[f'{label}_smoothness'] = get_smoothness_score(data, lattice_vectors)


    # add other properties
    properties['grid'] = grid
    properties['num_valence_electrons'] = num_valence_electrons

    return properties
    
def closeness_metrics(atoms, k=10, radii_table=covalent_radii):
    # scaling factors
    r = radii_table[atoms.get_atomic_numbers()]              # (N,)
    R = r[:, None] + r[None, :] # (N, N)

    # get per-atom nn distance
    D = atoms.get_all_distances(mic=True)  # (N, N)
    # Replace diagonal zeros with +inf so they aren't selected
    np.fill_diagonal(D, np.inf)
    
    # do scaling
    scaled_D = D / R

    scaled_per_atom_nn = scaled_D.min(axis=0)
    bottom_k_vals = np.partition(scaled_per_atom_nn, k)[:k]
    
    print(len(atoms))

    return {
        'min_scaled_distance': scaled_D.min(),
        f'avg_bottomk{k}_nn_distance': bottom_k_vals.mean(),
    }

@job
def post_process_slabpes(workdir_names, output_dir, uuids=None, process_volumetric=True):
    dataset_dir = Path(output_dir)
    for i, workdir in enumerate(workdir_names): # TODO: parallelize to allow running on short queue
        if workdir is None: continue
        workdir = Path(strip_hostname(workdir))
        vasprun = Vasprun(zpath(workdir/'vasprun.xml'), parse_potcar_file=False)
        if not vasprun.converged_electronic:
            continue
        outcar = Outcar(zpath(workdir/'OUTCAR'))
        outcar.read_vacuum_potentials() # put vacuum_potential_upper, vacuum_potential_lower in outcar.data, if found
        
        id = vasprun.incar['SYSTEM']
        input_dct = loadfn(f'{workdir}/jfremote_in.json')
        bulk_mpid = input_dct['job'].metadata.get('bulk_mpid', None)
        xc_functional = 'R2SCAN' if 'METAGGA' in vasprun.incar else 'PBE'
        has_dipole_correction = vasprun.incar.get('LDIPOL', False)

        # get energy, stress, forces, dipole using ASE
        atoms = read(zpath(workdir/'vasprun.xml'), -1)
        atoms.info['ref_energy'] = atoms.get_total_energy()
        atoms.info['ref_stress'] = atoms.get_stress(voigt=True)
        atoms.arrays['ref_forces'] = atoms.get_forces()
        if has_dipole_correction: atoms.info['ref_dipole'] = atoms.get_dipole_moment()
        atoms.calc = None


        # Determine if custodian applied any corrections (i.e., job was restarted)
        try:
            entries = loadfn(zpath(workdir / "custodian.json"))
            corrections = [c for e in entries for c in e.get("corrections", [])]
            restart_count = len(corrections)
        except Exception:
            restart_count = -1

        num_scf = len(vasprun.ionic_steps[-1]['electronic_steps'])

        vacuum_potential_upper = outcar.data.get("vacuum_potential_upper", None)
        vacuum_potential_lower = outcar.data.get("vacuum_potential_lower", None)
        try: drift = np.array(outcar.drift[-1])
        except: drift = None

        efermi = vasprun.efermi
        try: efermi_pmg = vasprun.calculate_efermi()
        except: efermi_pmg = None
        efield = vasprun.incar.get('EFIELD', None)

        try: band_gap = vasprun.get_band_structure(efermi="smart").get_band_gap()
        except: band_gap = None
        
        # get materials project properties
        bulk_band_gap = None
        if bulk_mpid is not None:
            atoms.info['bulk_mpid'] = bulk_mpid
            bulk_band_gap = bulk_mpid_to_bulk_bandgap(bulk_mpid)
            dipole_estimate_oxidation_state = estimate_dipole_from_oxidation_states(atoms)
        if bulk_band_gap is not None: atoms.info['bulk_band_gap'] = bulk_band_gap
        if dipole_estimate_oxidation_state is not None: atoms.info['dipole_estimate_oxidation_state'] = dipole_estimate_oxidation_state
        
        # 
        atoms.info['restart_count'] = restart_count
        atoms.info['num_scf'] = num_scf
        atoms.info['efermi'] = efermi
        if efermi_pmg is not None: atoms.info['efermi_pmg'] = efermi_pmg
        if band_gap is not None: atoms.info['band_gap'] = band_gap
        if efield is not None: atoms.info['efield'] = efield
        if vacuum_potential_upper is not None: atoms.info['vacuum_potential_upper'] = vacuum_potential_upper
        if vacuum_potential_lower is not None: atoms.info['vacuum_potential_lower'] = vacuum_potential_lower
        if drift is not None: atoms.info['drift'] = drift
        atoms.info['final_electronic_step'] = vasprun.ionic_steps[-1]['electronic_steps'][-1]
        atoms.info['time'] = outcar.run_stats['Elapsed time (sec)']
        
        # for tracking purposes
        atoms.info['vasp_dir_name'] = str(workdir)
        if uuids:
            atoms.info['uuid'] = uuids[i]

        # adsorbate tags + net adsorbate force
        atoms = add_tags_from_input_dct(input_dct, atoms) # TODO: test
        # closeness metrics
        atoms.info |= closeness_metrics(atoms)
        if process_volumetric:
            atoms.info |= get_volumetric_data_properties(workdir)

        # save
        functional_dipole_label = xc_functional
        if has_dipole_correction: functional_dipole_label += "_dipole"
        out_dir = (dataset_dir / functional_dipole_label / id); out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / "labels.xyz.gz", atoms, append=True)


@dataclass
class SlabPesStaticFlowMaker(Maker):
    """SlabPes flow doing a GGA static followed by meta-GGA static, then optional
    electric-field statics.

    The WAVECAR from the previous calculation is always used to accelerate the next calculation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static1 : .BaseVaspMaker or None
        Maker to generate the first VASP static.
    static2 : .BaseVaspMaker or None
        Maker to generate the second VASP static.
    efield_values : list[float]
        Electric field strengths to run (VASP EFIELD; units eV/Å) after static2.
    chain_fields : bool
        If True, each field job uses the previous field job as prev_dir; otherwise,
        all field jobs start from static2.
    """

    name: str = "SurfPES static flow"
    static1: Maker | None = field(
        default_factory=lambda: MatPesGGAStaticMaker(
            name=f"SlabPES GGA static",
            input_set_generator=SlabPESStaticSet(xc_functional='PBE'),
        )
    )
    static2: Maker | None = field(
        default_factory=lambda: MatPesMetaGGAStaticMaker(
            name=f"SlabPES meta-GGA static",
            input_set_generator = SlabPESStaticSet(xc_functional='R2SCAN'),
            # start from pre-conditioned WAVECAR from static1 to speed up convergence
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )
    )

    # Electric-field sweep configuration (run after static2).
    efield_values: list[float] = field(default_factory=list)  # e.g. [0.0, 0.05, 0.10]
    chain_field_calcs: bool = True

    clean_files: Sequence[str] | None = (
        "WAVECAR", "POTCAR", "XDATCAR", "REPORT", "CHG", # WAVECAR, POTCAR are big; REPORT, XDATCAR concern molecular dynamics; all CHG info is stored in CHGCAR
                   "POTCAR.orig", "POSCAR.orig" # not used in new calculations
        )

    def __post_init__(self) -> None:
        """Validate flow."""
        if self.static1 is None and self.static2 is None and not self.efield_values:
            raise ValueError("Must provide at least one job: static1, static2, efield_values")

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a flow with SlabPES statics and optional electric-field sweep."""
        jobs = []
        output = {}

        # 1) static1
        if self.static1 is not None:
            static1 = self.static1.make(structure, prev_dir=prev_dir)
            jobs.append(static1)
            output["static1"] = static1.output

        prev_dir_after_s1 = static1.output.dir_name if self.static1 is not None else prev_dir

        # 2) static2
        if self.static2 is not None:
            static2 = self.static2.make(structure, prev_dir=prev_dir_after_s1)
            jobs.append(static2)
            output["static2"] = static2.output
            prev_after_s2 = static2.output.dir_name
        else:
            prev_after_s2 = prev_dir_after_s1

        # 3) Electric-field sweep after static2
        efield_jobs = []
        if self.efield_values:
            prev_for_field = prev_after_s2
            output["efield_statics"] = {}
            for idx, efield in enumerate(self.efield_values):
                field_set = SlabPESStaticSet(
                    # Ensure dipole correction and field are set
                    xc_functional="R2SCAN",
                    auto_dipole=True,
                    user_incar_settings={"EFIELD": efield}
                )
                field_maker = MatPesMetaGGAStaticMaker(
                    name=f"SlabPES meta-GGA static, E={efield}",
                    input_set_generator=field_set,
                    copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)},
                )
                field_job = field_maker.make(structure, prev_dir=prev_for_field)
                jobs.append(field_job)
                efield_jobs.append(field_job)
                if self.chain_field_calcs:
                    prev_for_field = field_job.output.dir_name
                output["efield_statics"][efield] = field_job.output

        vasp_directories: list[str] = []
        _recursive_get_dir_names(jobs, vasp_directories)

        self.clean_files = self.clean_files or []
        if len(self.clean_files) > 0:
            cleanup = remove_workflow_files(
                directories=vasp_directories,
                file_names=self.clean_files,
                allow_zpath=True,
            )
            cleanup.config.on_missing_references = OnMissing.NONE
            jobs += [cleanup]

        return Flow(jobs=jobs, output=output, name=self.name)
