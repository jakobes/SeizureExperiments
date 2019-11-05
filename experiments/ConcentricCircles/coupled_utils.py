import dolfin as df
import numpy as np

from typing import (
    Union,
    NamedTuple,
    List,
    Sequence,
    Iterator,
    Tuple,
    Any,
    Dict,
)

from xalode import VectorInt


class CellTags(NamedTuple):
    CSF: int = 1
    GM: int = 2
    WM: int = 3
    Kinf: int = 4


class InterfaceTags(NamedTuple):
    CSF_GM: int = 4
    GM_WM: int = 5
    skull: int = 6
    CSF: int = 7
    GM: int = 8
    WM: int = 9


class CoupledMonodomainParameters(NamedTuple):
    timestep: df.Constant = df.Constant(1.0)
    theta: df.Constant = df.Constant(0.5)
    linear_solver_type: str = "direct"
    lu_type: str = "default"
    krylov_method: str = "cg"
    krylov_preconditioner: str = "petsc_amg"


class CoupledBidomainParameters(NamedTuple):
    restrict_tags: Any = set()
    timestep: df.Constant = df.Constant(1.0)
    theta: df.Constant = df.Constant(0.5)
    linear_solver_type: str = "direct"
    lu_type: str = "default"
    krylov_method: str = "gmres"   # CG fails
    krylov_preconditioner: str = "petsc_amg"


class CoupledSplittingSolverParameters(NamedTuple):
    theta: df.Constant = df.Constant(0.5)


class CoupledODESolverParameters(NamedTuple):
    parameter_map: "ODEMap"
    valid_cell_tags: Sequence[int]
    timestep: df.Constant = df.Constant(1)
    reload_extension_modules: bool = False
    theta: df.Constant = df.Constant(0.5)


def get_mesh(directory: str, name: str) -> Tuple[df.Mesh, df.MeshFunction, df.MeshFunction]:
    mesh = df.Mesh()
    with df.XDMFFile(f"{directory}/{name}.xdmf") as infile:
        infile.read(mesh)

    mvc = df.MeshValueCollection("size_t", mesh, 2)
    with df.XDMFFile(f"{directory}/{name}_cf.xdmf") as infile:
        infile.read(mvc, "cell_data")
    cell_function = df.MeshFunction("size_t", mesh, mvc)

    try:
        mvc = df.MeshValueCollection("size_t", mesh, 1)
        with df.XDMFFile(f"{directory}/{name}_ff.xdmf") as infile:
            infile.read(mvc, "facet_data")
        interface_function = df.MeshFunction("size_t", mesh, mvc)
    except:
        interface_function = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
        interface_function.set_all(0)
    return mesh, cell_function, interface_function


def create_linear_solver(
        lhs_matrix,
        parameters: CoupledMonodomainParameters
) -> Union[df.LUSolver, df.KrylovSolver]:
    """helper function for creating linear solver."""
    solver_type = parameters.linear_solver_type       # direct or iterative

    if solver_type == "direct":
        solver = df.LUSolver(lhs_matrix, parameters.lu_type)
        solver.parameters["symmetric"] = True

    elif solver_type == "iterative":
        method = parameters.krylov_method
        preconditioner = parameters.krylov_preconditioner

        solver = df.PETScKrylovSolver(method, preconditioner)
        solver.set_operator(lhs_matrix)
        solver.parameters["nonzero_initial_guess"] = True
        solver.ksp().setFromOptions()       # TODO: What is this?
    else:
        raise ValueError(f"Unknown linear_solver_type given: {solver_type}")

    return solver


def masked_dofs(dofmap: df.DofMap, cell_domains_array: np.ndarray, valid_cell_tags: Sequence[int]):

    mask_list: List[int] = []
    for i, ct in enumerate(cell_domains_array):
        cell_dofs = dofmap.cell_dofs(i)
        if ct in valid_cell_tags:
            mask_list += list(cell_dofs)
    # return VectorInt(np.unique(mask_list))
    return VectorInt(mask_list)


def time_stepper(*, t0: float, t1: float, dt: float = None) -> Iterator[Tuple[float, float]]:
    if dt is None:
        dt = t1 - t0
    elif dt > t1 - t0:
        raise ValueError("dt greater than time interval")

    _t0 = t0
    _t = t0 + dt
    while _t < t1:
        yield _t0, _t
        _t += dt
        _t0 += dt

