"""
Module defining SurfPES flows.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, Flow
from pymatgen.io.vasp.sets import SlabPESStaticSet

from atomate2.vasp.jobs.matpes import MatPesGGAStaticMaker, MatPesMetaGGAStaticMaker
from atomate2.common.jobs.utils import remove_workflow_files
from atomate2.common.utils import _recursive_get_dir_names


if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core import Structure
    from collections.abc import Sequence

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
            input_set_generator=SlabPESStaticSet(),
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

        # add vasp_dir_names to output to aid post-processing
        vasp_directories: list[str] = []
        _recursive_get_dir_names(jobs, vasp_directories)
        output["vasp_dir_names"] = vasp_directories

        self.clean_files = self.clean_files or []
        if len(self.clean_files) > 0:
            cleanup = remove_workflow_files(
                directories=vasp_directories,
                file_names=self.clean_files,
                allow_zpath=True,
            )
            jobs += [cleanup]

        return Flow(jobs=jobs, output=output, name=self.name)
