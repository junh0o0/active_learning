"""Logging for molecular dynamics."""
import weakref
import numpy as np

from typing import IO, Any, Union

from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


class ALMDLogger(IOContext):
    """Class for logging molecular dynamics simulations.

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.

    logfile:       File name or open file, "-" meaning standard output.

    stress=False:  Include stress in log.

    peratom=False: Write energies per atom.

    mode="a":      How the file is opened if logfile is a filename.
    """

    def __init__(
        self,
        dyn: Any,  # not fully annotated so far to avoid a circular import
        atoms: Atoms,
        logfile: Union[IO, str],
        header: bool = True,
        active : bool = True,
        mode: str = "a",
        comm=world,
    ):
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, "get_time") else None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        self.active = active
        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        if self.active:
            self.hdr += "%12s %12s %12s %12s %12s %12s  %6s" % ("E1[eV]", "E2[eV]","E3[eV]","E4[eV]","E_mean[eV]","STD","T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 6 * ("%%12.%df " % (digits,)) + " %6.1f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        e1 = self.dyn.atoms.calc.results["energies"][0]
        e2 = self.dyn.atoms.calc.results["energies"][1]
        e3 = self.dyn.atoms.calc.results["energies"][2]
        e4 = self.dyn.atoms.calc.results["energies"][3]
        em = self.dyn.atoms.calc.results["energy"]
        std = np.sqrt(((e1-em)**2+(e2-em)**2+(e3-em)**2+(e4-em)**2)/4)
        temp = self.atoms.get_temperature()
        global_natoms = self.atoms.get_global_number_of_atoms()
        if self.active is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        dat += (e1,e2,e3,e4,em,std,temp)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()
