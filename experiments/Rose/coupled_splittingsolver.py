import dolfin as df

from coupled_utils import (
    time_stepper,
    CoupledSplittingSolverParameters,
    CoupledMonodomainParameters,
    CoupledODESolverParameters,
    CoupledBidomainParameters,
)

from typing import (
    Iterator,
    Tuple,
)

from abc import (
    ABC,
    abstractmethod
)

from coupled_odesolver import CoupledODESolver

from coupled_monodomain import (
    CoupledMonodomainSolver
)

from coupled_bidomain import CoupledBidomainSolver

from coupled_brainmodel import CoupledBrainModel

from collections import namedtuple


SolutionStruct = namedtuple("SolutionStruct", ("t0", "t1", "vs_prev", "vs", "vur"))


class CoupledSplittingSolver(ABC):
    def __init__(
        self,
        *,
        brain: CoupledBrainModel,
        parameters: CoupledSplittingSolverParameters,
    ) -> None:
        """Create solver from given Model and (optional) parameters."""
        self._brain = brain
        self._parameters = parameters

        # Create ODE solver and extract solution fields
        self.ode_solver = self.create_ode_solver()
        self.vs_prev, self.vs = self.ode_solver.solution_fields()
        self.VS = self.vs.function_space()

        # Create PDE solver and extract solution fields
        self.pde_solver = self.create_pde_solver()
        self.v_prev, self.vur = self.pde_solver.solution_fields()

        # # Create function assigner for merging v from self.vur into self.vs[0]
        if self.vur.function_space().num_sub_spaces() > 1:
            V = self.vur.function_space().sub(0)
        else:
            V = self.vur.function_space()
        self.merger = df.FunctionAssigner(self.VS.sub(0), V)

    @abstractmethod
    def create_ode_solver(self):
        pass

    @abstractmethod
    def create_pde_solver(self):
        pass

    @abstractmethod
    def merge(self, solution: df.Function) -> None:
        pass

    def solve(self, t0: float, t1: float, dt: float) -> Iterator[SolutionStruct]:
        """
        Solve the problem given by the model on a time interval with a given time step.
        Return a generator for a tuple of the time step and the solution fields.

        *Arguments*
          interval (:py:class:`tuple`)
            The time interval for the solve given by (t0, t1)
          dt (int, list of tuples of floats)
            The timestep for the solve. A list of tuples of floats can
            also be passed. Each tuple should contain two floats where the
            first includes the start time and the second the dt.

        *Returns*
          (timestep, solution_fields) via (:py:class:`genexpr`)

        *Example of usage*::

          # Create generator
          dts = [(0., 0.1), (1.0, 0.05), (2.0, 0.1)]
          solutions = solver.solve((0.0, 1.0), dts)

          # Iterate over generator (computes solutions as you go)
          for ((t0, t1), (vs_, vs, vur)) in solutions:
            # do something with the solutions

        """
        # Create timestepper
        for interval in time_stepper(t0=t0, t1=t1, dt=dt):
            self.step(*interval)       # Takes only one step

            # Yield solutions
            yield SolutionStruct(*interval, *self.solution_fields())
            # yield interval, self.solution_fields()

            # Update previous solution
            self.vs_prev.assign(self.vs)

    def step(self, t0: float, t1: float) -> None:
        """
        Solve the pde for one time step.

        Invariants:
            Given self._vs in a correct state at t0, provide v and s (in self.vs) and u
            (in self.vur) in a correct state at t1.
            (Note that self.vur[0] == self.vs[0] only if theta = 1.0.)
        """
        theta = self._parameters.theta

        # Extract time domain
        _dt = t1 - t0
        t = t0 + theta*_dt

        # Compute tentative membrane potential and state (vs_star)
        # df.begin(df.PROGRESS, "Tentative ODE step")
        # Assumes that its vs_ is in the correct state, gives its vs
        # in the current state
        # self.ode_solver.step((t0, t))
        self.ode_solver.step(t0, t)

        # TODO: This should be sued when using fenics ode solver!
        # self.vs_.assign(self.vs)

        # Compute tentative potentials vu = (v, u)
        # Assumes that its vs_ is in the correct state, gives vur in
        # the current state
        self.pde_solver.step(t0, t1)

        # If first order splitting, we need to ensure that self.vs is
        # up to date, but otherwise we are done.
        if theta == 1.0:
            # Assumes that the v part of its vur and the s part of its
            # vs are in the correct state, provides input argument(in
            # this case self.vs) in its correct state
            self.merge(self.vs)
            return

        # Otherwise, we do another ode_step:

        # Assumes that the v part of its vur and the s part of its vs
        # are in the correct state, provides input argument (in this
        # case self.vs_) in its correct state
        self.merge(self.vs_prev)    # self.vs_.sub(0) <- self.vur.sub(0)
        # Assumes that its vs_ is in the correct state, provides vs in the correct state

        # self.ode_solver.step((t0, t))
        self.ode_solver.step(t0, t)

    def solution_fields(self) -> Tuple[df.Function, df.Function, df.Function]:
        """
        vs is the current solution to the ode.
        vs_prev is the previous solution to the ode.
        vur is the solution to the pde.
        """
        return self.vs_prev, self.vs, self.vur


class MonodomainSplittingSolver(CoupledSplittingSolver):
    def __init__(
        self,
        *,
        brain: CoupledBrainModel,
        parameters: CoupledSplittingSolverParameters,
        ode_parameters: CoupledODESolverParameters,
        pde_parameters: CoupledMonodomainParameters
    ) -> None:
        self._pde_parameters = pde_parameters
        self._ode_parameters = ode_parameters
        super().__init__(brain=brain, parameters=parameters)

    def create_ode_solver(self) -> CoupledODESolver:
        """The idea is to subplacc this and implement another version of this function."""
        solver = CoupledODESolver(
            time=self._brain.time,
            mesh=self._brain.mesh,
            cell_model=self._brain.cell_model,
            cell_function=self._brain.cell_function,
            parameters=self._ode_parameters,
        )
        return solver

    def create_pde_solver(self) -> CoupledMonodomainSolver:
        """The idea is to subplacc this and implement another version of this function."""

        solver = CoupledMonodomainSolver(
            self._brain.time,
            self._brain.mesh,
            self._brain.intracellular_conductivity,
            self._brain.extracellular_conductivity,
            self._brain.cell_function,
            self._brain.cell_tags,
            self._brain.interface_function,
            self._brain.interface_tags,
            self._pde_parameters,
            self._brain.neumann_boundary_condition,
            v_prev=self.vs[0]
        )
        return solver

    def merge(self, solution: df.Function) -> None:
        """
        Combine solutions from the PDE and the ODE to form a single mixed function.

        `solution` holds the solution from the PDEsolver.
        """
        v = self.vur
        self.merger.assign(solution.sub(0), v)


class BidomainSplittingSolver(CoupledSplittingSolver):
    def __init__(
        self,
        *,
        brain: CoupledBrainModel,
        parameters: CoupledSplittingSolverParameters,
        ode_parameters: CoupledODESolverParameters,
        pde_parameters: CoupledBidomainParameters
    ) -> None:
        self._pde_parameters = pde_parameters
        self._ode_parameters = ode_parameters
        super().__init__(brain=brain, parameters=parameters)

    def create_ode_solver(self) -> CoupledODESolver:
        """The idea is to subplacc this and implement another version of this function."""
        solver = CoupledODESolver(
            time=self._brain.time,
            mesh=self._brain.mesh,
            cell_model=self._brain.cell_model,
            cell_function=self._brain.cell_function,
            parameters=self._ode_parameters,
        )
        return solver

    def create_pde_solver(self) -> CoupledBidomainSolver:
        """The idea is to subplacc this and implement another version of this function."""

        solver = CoupledBidomainSolver(
            self._brain.time,
            self._brain.mesh,
            self._brain.intracellular_conductivity,
            self._brain.extracellular_conductivity,
            self._brain.cell_function,
            self._brain.cell_tags,
            self._brain.interface_function,
            self._brain.interface_tags,
            self._pde_parameters,
            self._brain.neumann_boundary_condition,
            v_prev=self.vs[0],
            surface_to_volume_factor=self._brain.surface_to_volume_factor,
            membrane_capacitance=self._brain.membrane_capacitance
        )
        return solver

    def merge(self, solution: df.Function) -> None:
        """
        Combine solutions from the PDE and the ODE to form a single mixed function.

        `solution` holds the solution from the PDEsolver.
        """
        v = self.vur.sub(0)
        self.merger.assign(solution.sub(0), v)
