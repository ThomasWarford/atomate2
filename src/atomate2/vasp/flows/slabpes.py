"""
Module defining SurfPES flows.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, Flow, job, OnMissing
from pymatgen.io.vasp.sets import SlabPESStaticSet
import numpy as np

from pathlib import Path
from atomate2.vasp.powerups import update_user_incar_settings
from atomate2.utils.path import strip_hostname
from pymatgen.io.vasp import Outcar, Vasprun, Chgcar, Locpot
from monty.os.path import zpath
from monty.serialization import loadfn
from ase.io import read, write
from mp_api.client import MPRester

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker
from atomate2.common.jobs.utils import remove_workflow_files
from atomate2.common.utils import _recursive_get_dir_names


if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from collections.abc import Sequence

def bulk_mpid_to_bulk_bandgap(bulk_mpid):
    try:
        with MPRester() as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[bulk_mpid],
                fields=["material_id", "band_gap"]
            )
        return docs[0].band_gap 
    except:
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

@job
def post_process_slabpes(workdir_names, output_dir, uuids=None, process_volumetric=False):
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
        try: dipole = atoms.get_dipole_moment()
        except: dipole = None
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
        if bulk_band_gap is not None: atoms.info['bulk_band_gap'] = bulk_band_gap
        
        # 
        if dipole is not None: atoms.info['dipole'] = dipole
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

        # save
        functional_dipole_label = xc_functional
        if has_dipole_correction: functional_dipole_label += "_dipole"
        out_dir = (dataset_dir / functional_dipole_label / id); out_dir.mkdir(parents=True, exist_ok=True)
        write(out_dir / "labels.xyz.gz", atoms, append=True)

        if process_volumetric:
            # chgcar stuff
            c_stem = 'charge'
            if efield is not None: c_stem += f'_e{efield}'
            Chgcar.from_file(zpath(workdir/'CHGCAR')).to_cube(out_dir / f'{c_stem}.cube.gz')

            # locpot stuff
            l_stem = 'locpot'
            if efield is not None: l_stem += f'_e{efield}'
            Locpot.from_file(zpath(workdir/'LOCPOT')).to_cube(out_dir / f'{l_stem}.cube.gz')

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
