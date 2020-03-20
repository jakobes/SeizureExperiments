import dolfin as df

from xalode import VectorInt, VectorSizet

from extension_modules import load_module

from typing import (
    Tuple,
    Dict,
    NamedTuple,
    Sequence,
    Iterator,
)

from coupled_utils import (
    time_stepper,
    CoupledODESolverParameters,
)

from xalbrain.cellmodels import CellModel


class CoupledODESolver:
    def __init__(
            self,
            time: df.Constant,
            mesh: df.Mesh,
            cell_model: CellModel,
            parameters: CoupledODESolverParameters,
            cell_function: df.MeshFunction,
    ) -> None:
        """Initialise parameters. NB! Keep I_s for compatibility"""
        # Store input
        self._mesh = mesh
        self._time = time
        self._model = cell_model     # FIXME: For initial conditions and num states

        # Extract some information from cell model
        self._num_states = self._model.num_states()

        self._parameters = parameters
        _cell_function_tags = set(cell_function.array())
        if not set(self._parameters.valid_cell_tags) <= _cell_function_tags:
            msg = "Valid cell tag not found in cell function. Expected {}, for {}."
            raise ValueError(msg.format(set(self._parameters.valid_cell_tags), _cell_function_tags))
        valid_cell_tags = self._parameters.valid_cell_tags

        # Create (vector) function space for potential + states
        self._function_space_VS = df.VectorFunctionSpace(self._mesh, "CG", 1, dim=self._num_states + 1)

        # Initialize solution field
        self.vs_prev = df.Function(self._function_space_VS, name="vs_prev")
        self.vs = df.Function(self._function_space_VS, name="vs")

        self.ode_module = load_module(
            "LatticeODESolver",
            recompile=self._parameters.reload_extension_modules,
            verbose=self._parameters.reload_extension_modules
        )

        self.ode_solver = self.ode_module.LatticeODESolver(
            self._function_space_VS._cpp_object,
            VectorSizet(cell_function.array()),
            self._parameters.parameter_map
        )


    def solution_fields(self) -> Tuple[df.Function, df.Function]:
        """
        Return current solution object.

        Modifying this will modify the solution object of the solver
        and thus provides a way for setting initial conditions for
        instance.
        """
        return self.vs_prev, self.vs

    def step(self, t0: float, t1: float) -> None:
        """Take a step using my much better ode solver."""
        theta = self._parameters.theta
        dt = t1 - t0        # TODO: Is this risky?

        # Set time (propagates to time-dependent variables defined via self.time)
        t = t0 + theta*(t1 - t0)
        self._time.assign(t)

        # FIXME: Is there some theta shenanigans I have missed?
        self.ode_solver.solve(self.vs_prev.vector(), t0, t1, dt)
        self.vs.assign(self.vs_prev)

    def solve(
            self,
            t0: float,
            t1: float,
            dt: float = None,
    ) -> Iterator[Tuple[Tuple[float, float], df.Function]]:
        """
        Solve the problem given by the model on a given time interval
        (t0, t1) with a given timestep dt and return generator for a
        tuple of the interval and the current vs solution.

        *Example of usage*::

          # Create generator
          solutions = solver.solve(0.0, 1.0, 0.1)

          # Iterate over generator (computes solutions as you go)
          for interval, vs in solutions:
            # do something with the solutions

        """
        # Solve on entire interval if no interval is given.
        for interval in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(*interval)

            # Yield solutions
            yield interval, self.vs
            self.vs_prev.assign(self.vs)


class CoupledSingleCellSolver(CoupledODESolver):
    def __init__(
            self,
            cell_model: CellModel,
            time: df.Constant,
            reload_ext_modules: bool = False,
            parameters: df.Parameters = None
    ) -> None:
        """Create solver from given cell model and optional parameters."""
        # Store model
        self.cell_model = cell_model

        # Define carefully chosen dummy mesh
        mesh = df.UnitIntervalMesh(1)

        super().__init__(
            mesh,
            time,
            cell_model,
            reload_ext_modules=reload_ext_modules,
            parameters=parameters
        )
