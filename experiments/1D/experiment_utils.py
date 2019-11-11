"""Utility functions for running experiments."""

import dolfin as df
import numpy as np

from xalbrain import (
    SplittingSolver,
    CardiacModel,
)

from postspec import LoaderSpec

from post import Loader

from pathlib import Path

from typing import Any


def get_mesh(dimension: int, N: int) -> df.Mesh:
    """Create the mesh [0, 1]^d cm."""
    if dimension == 1:
        mesh = df.UnitIntervalMesh(N)
    elif dimension == 2:
        mesh = df.UnitSquareMesh(N, N)         # 1cm time 1cm
    elif dimension == 3:
        mesh = df.UnitCubeMesh(N, N, N)       # 1cm time 1cm
    return mesh


def get_conductivities(value: float = 0.5) -> df.Constant:
    """Create the conductivities."""
    Mi = df.Constant(value)     # mS/cm
    return Mi


def get_solver(brain: CardiacModel, ode_dt: int = 1) -> SplittingSolver:
    ps = SplittingSolver.default_parameters()
    ps["pde_solver"] = "monodomain"
    ps["theta"] = 0.5
    ps["MonodomainSolver"]["linear_solver_type"] = "direct"
    ps["BidomainSolver"]["use_avg_u_constraint"] = False
    ps["CardiacODESolver"]["scheme"] = "RK4"

    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["optimize"] = True

    flags = "-O3 -ffast-math -march=native"
    df.parameters["form_compiler"]["cpp_optimize_flags"] = flags

    df.parameters["form_compiler"]["quadrature_degree"] = 1
    solver = SplittingSolver(brain, ode_timestep=ode_dt, params=ps)
    return solver


def assign_initial_conditions(solver: Any) -> None:
    brain = solver.model
    model = brain.cell_models
    vs_, *_ = solver.solution_fields()
    vs_.assign(model.initial_conditions())


def reload_initial_condition(solver: Any, casedir: Path) -> None:
    loader_spec = LoaderSpec(casedir)
    loader = Loader(loader_spec)
    ic = loader.load_initial_condition("v", timestep_index=-10)
    vs_, *_ = solver.solution_fields()
    vs_.assign(ic)


def get_points(dimension: int, num_points: int) -> np.ndarray:
    _npj = num_points*1j
    if dimension == 1:
        numbers = np.mgrid[0:1:_npj]
        return np.vstack(map(lambda x: x.ravel(), numbers)).reshape(-1, dimension)
    if dimension == 2:
        # numbers = np.mgrid[0:1:_npj, 0:1:_npj]
        my_range = np.arange(10)/10

        foo = np.zeros(shape=(10, 2))
        foo[:, 0] = my_range

        bar = np.zeros(shape=(10, 2))
        bar[:, 0] = my_range
        bar[:, 1] = my_range
        return np.vstack((foo, bar))
    if dimension == 3:
        assert False, "Do something clever here"
        pass


def get_outpath(dim=1) -> Path:
    return Path("{:d}D_out_cressman".format(dim))
