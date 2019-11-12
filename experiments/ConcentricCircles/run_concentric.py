import time
import resource
import warnings
import datetime
import itertools
import sys

import dolfin as df
import numpy as np

from pathlib import Path
from scipy import signal
from math import pi

from extension_modules import load_module

from postfields import (
    Field,
    PointField,
)

from typing import (
    Union,
    Tuple,
)

from post import Saver
from coupled_brainmodel import CoupledBrainModel
from coupled_splittingsolver import (
    BidomainSplittingSolver,
    MonodomainSplittingSolver,
)

from postutils import (
    interpolate_ic,
    store_sourcefiles,
    simulation_directory,
    circle_points,
)

from xalbrain.cellmodels import (
    Cressman,
    Noble
)

from coupled_utils import (
    CellTags,
    InterfaceTags,
    CoupledSplittingSolverParameters,
    CoupledODESolverParameters,
    CoupledBidomainParameters,
    CoupledMonodomainParameters,
    get_mesh,
)

from postspec import (
    FieldSpec,
    SaverSpec,
)



def get_brain(i, conductivity) -> CoupledBrainModel:
    time_constant = df.Constant(0)
    mesh, cell_function, interface_function = get_mesh("concentric_meshes", "concentric_circle{i}".format(i=i))

    cell_tags = CellTags(CSF=None, GM=2, WM=1, Kinf=None)
    interface_tags = InterfaceTags(skull=None, CSF_GM=None, GM_WM=None, CSF=None, GM=None, WM=None)

    Chi = 1.26e3      # 1/cm -- Dougherty 2015
    Cm = 1.0          # muF/cm^2 -- Dougherty 2015

    Mi_dict = {
        1: df.Constant(conductivity),        # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
        2: df.Constant(conductivity),       # Dlougherty isotropic WM intracellular conductivity 1.0 [mS/cm]
    }

    lambda_dict = {
        1: df.Constant(2.76),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm] 16.54
        2: df.Constant(2.76),     # Dougherty isotropic CSF conductivity 16.54 [mS/cm] 16.54
    }

    brain = CoupledBrainModel(
        time=time_constant,
        mesh=mesh,
        cell_model=Cressman(),
        cell_function=cell_function,
        cell_tags=cell_tags,
        interface_function=interface_function,
        interface_tags=interface_tags,
        intracellular_conductivity=Mi_dict,
        other_conductivity=lambda_dict,         # Either lmbda or extracellular
        surface_to_volume_factor=Chi,
        membrane_capacitance=Cm
    )
    return brain


def get_solver(brain, Ks, Ku) -> BidomainSplittingSolver:

    odesolver_module = load_module("LatticeODESolver")
    odemap = odesolver_module.ODEMap()
    odemap.add_ode(1, odesolver_module.Cressman(Ks))
    odemap.add_ode(2, odesolver_module.Cressman(Ku))

    parameters = CoupledSplittingSolverParameters()
    ode_parameters = CoupledODESolverParameters(
        valid_cell_tags=(1, 2),
        reload_extension_modules=False,
        parameter_map=odemap
    )

    pde_parameters = CoupledMonodomainParameters(
    # pde_parameters = CoupledBidomainParameters(
        linear_solver_type="direct"
    )

    solver = MonodomainSplittingSolver(
    # solver = BidomainSplittingSolver(
        brain=brain,
        parameters=parameters,
        ode_parameters=ode_parameters,
        pde_parameters=pde_parameters
    )

    vs_prev, *_ = solver.solution_fields()
    vs_prev.assign(brain.cell_model.initial_conditions())
    return solver


def get_saver(
    brain: CoupledBrainModel,
    outpath: Union[str, Path],
    case_index: int
) -> Saver:
    sourcefiles = [
        "coupled_monodomain.py",
        "coupled_brainmodel.py",
        "coupled_odesolver.py",
        "coupled_splittingsolver.py",
        "coupled_utils.py",
        "run_concentric.py",
    ]
    store_sourcefiles(map(Path, sourcefiles), outpath)

    saver_parameters = SaverSpec(casedir=outpath, overwrite_casedir=True)
    saver = Saver(saver_parameters)
    saver.store_mesh(brain.mesh)

    field_spec_checkpoint = FieldSpec(save_as=("xdmf", "hdf5"), stride_timestep=20)
    saver.add_field(Field("v", field_spec_checkpoint))

    field_spec_checkpoint = FieldSpec(save_as=("hdf5"), stride_timestep=20*1000)
    saver.add_field(Field("vs", field_spec_checkpoint))

    points = np.zeros((9, 2))
    points[:, 0] = np.linspace(0.1, 0.9, 9)

    trace_names = []
    for subfield_index in range(7):
        point_field_spec = FieldSpec(stride_timestep=4, sub_field_index=subfield_index)
        name = "trace_sub{}".format(subfield_index)
        trace_names.append(name)
        saver.add_field(PointField(name, point_field_spec, points))
    return saver, trace_names


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    from multiprocessing import Pool

    def run(args):
        conductivity, case_id, Ks, Ku = args
        # Ks, Ku, case_id = args
        T = 1e1
        dt = 0.05

        brain = get_brain(case_id, conductivity)
        solver = get_solver(brain, Ks, Ku)

        identifier = simulation_directory(
            home=Path("."),
            parameters={
                "time": datetime.datetime.now(),
                "case_id": case_id,
                "conductivity": conductivity, 
                "Ks": Ks,
                "Ku": Ku
            },
            directory_name=Path("concentric_circle")
        )

        saver, trace_name_list = get_saver(brain, identifier, case_id)
        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        tick = time.perf_counter()
        for i, solution_struct in enumerate(solver.solve(0, T, dt)):
            print(f"{i} -- {brain.time(0):.3f} -- {solution_struct.vur.vector().norm('l2'):.3e}")

            update_dict = dict()
            if saver.update_this_timestep(field_names=trace_name_list, timestep=i, time=brain.time(0)):
                update_dict.update({n: solution_struct.vs for n in trace_name_list})

            if saver.update_this_timestep(field_names=["v"], timestep=i, time=brain.time(0)):
                update_dict.update({"v": solution_struct.vur})

            if saver.update_this_timestep(field_names=["vs"], timestep=i, time=brain.time(0)):
                update_dict.update({"vs": solution_struct.vs})

            if len(update_dict) != 0:
                saver.update(brain.time, i, update_dict)

        saver.close()
        tock = time.perf_counter()
        max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
        print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
        print("Execution time: {:.2f} s".format(tock - tick))

    run((1, 1, 4, 8))
    # conductivities = [2**(2*n) for n in range(-3, 2)]
    # lengths = list(range(3))

    # Ks = float(sys.argv[1])
    # Ku = float(sys.argv[2])

    # parameter_list = list(itertools.product(conductivities, lengths, [Ks], [Ku]))
    # print(len(parameter_list))
