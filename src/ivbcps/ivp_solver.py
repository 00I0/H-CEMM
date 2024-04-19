from abc import ABC, abstractmethod
from typing import Tuple, MutableSequence

import numpy as np
from pde import ScalarField, CallbackTracker, CartesianGrid, Boundaries, PeriodicityError

from ivbcps.diffusion_PDEs import DiffusionPDEBase
from ivbcps.ivbcp import SymmetricIVBCPBase


class IVPSolverBase(ABC):
    """
    A base class for solvers of initial value problems (IVPs) of diffusion partial differential equations (PDEs).

    Methods:
        solve(t_range, dt, collection_interval, plot_kind, report_progress): Solve the PDE over a given time range.

    Attributes:
        _pde (DiffusionPDEBase): The diffusion PDE to be solved.
        _initial_condition (np.ndarray): The initial condition for the PDE simulation.

    Properties:
        pde (DiffusionPDEBase): Property for getting and setting the diffusion PDE.
    """

    def __init__(
            self,
            pde: DiffusionPDEBase,
            initial_condition: np.ndarray,
    ):
        """
        Initialize a IVPPDESolverBase. Since it's an abstract class this method should not be called directly.

        Args:
            pde (DiffusionPDEBase): The diffusion PDE to solve.
            initial_condition (np.ndarray): The initial condition for the PDE, a one dimensional np.ndarray with the
                values along a radius, the first `inner_radius` number of values will be treated as if they were nans.
        """
        self._pde = pde
        self._initial_condition = initial_condition

    @property
    def pde(self):
        return self._pde

    @abstractmethod
    def solve(
            self,
            t_range: float | Tuple[float, float],
            dt: float = 1e-4,
            collection_interval: float | None = None,
            report_progress: bool = True,
            scheme: str = 'rk45'
    ) -> list[np.ndarray]:
        """
        Abstract method to solve the PDE over a given time range. Must be implemented by subclasses.

        Args:
            t_range (float | Tuple[float, float]): The time range for the simulation, either as a single float
                indicating the end time or a tuple of (start, end) times.
            dt (float): The time step for the simulation.
            collection_interval (Union[float, None]): Interval for collecting simulation states, if None, states are
                only collected at the end.
            report_progress (bool): Flag to indicate whether progress should be reported during the simulation.
            scheme (str): Numerical integration scheme to use. (could be: 'rk45' or 'euler')

        Returns:
            list: A list of np.ndarray objects representing the state of the system at each collected interval.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()


class SymmetricIVPSolver(IVPSolverBase):
    """
    A solver class specialized for symmetric initial value problems (IVPs) of diffusion PDEs, implementing radial
    symmetry handling.

    Inherits from IVPSolverBase and implements specific numerical strategies for symmetric PDEs.
    """

    def __init__(
            self,
            symmetric_ivp: SymmetricIVBCPBase,
            is_padding_with_nans: bool = True
    ):
        """
        Args:
            symmetric_ivp (SymmetricIVBCPBase): The well specified IVBCP to solve
            is_padding_with_nans (bool): Whether to pad the solution with NaNs from zero up to the initial condition.
        """
        super().__init__(symmetric_ivp.pde, symmetric_ivp.initial_condition)
        if self._initial_condition.ndim != 1:
            raise ValueError(f'`initial_condition` must be 1D, but it was: {self._initial_condition.ndim}')

        self._padding = len(self._initial_condition)
        self._spatial_size = symmetric_ivp.spatial_size
        self._inner_radius = symmetric_ivp.inner_radius
        self._is_padding_with_nans = is_padding_with_nans

        self._periodicity = None
        for periodicity in [(False, False), (False, True), (True, False), (True, True)]:
            try:
                grid = CartesianGrid(
                    bounds=[(0, 1), (0, self._spatial_size)],
                    shape=(1, len(self._initial_condition)),
                    periodic=periodicity
                )
                Boundaries.from_data(grid, self.pde.bc)
                self._periodicity = periodicity
                break
            except PeriodicityError:
                continue

        if self._periodicity is None:
            raise ValueError('Invalid boundary conditions.')

    def _create_collect_callback(self, container: MutableSequence[np.ndarray]):
        if self._is_padding_with_nans:
            inner_radius_np_nan = np.empty(self._inner_radius) * np.nan

            def callback(state: ScalarField, _time: float):
                state = state.data[0, :len(self._initial_condition)]
                container.append(np.concatenate((inner_radius_np_nan, state.copy())))

            return callback

        def callback(state: ScalarField, _time: float):
            state = state.data[0, :len(self._initial_condition)]
            container.append(state.copy())

        return callback

    def solve(
            self,
            t_range: float | Tuple[float, float],
            dt: float | None = 1e-4,
            collection_interval: float | None = None,
            report_progress: bool = True,
            scheme: str = 'rk45'
    ) -> list[np.ndarray]:
        """
        Implements the solve method for symmetric IVPs, solving the diffusion PDE over a specified time range using the
         defined scheme.

        Args:
            t_range (Union[float, Tuple[float, float]]): The time range for the simulation.
            dt (float | None): The time step for the simulation. If None the `self.pde.calculate_delta_t` will be used.
            collection_interval (Union[float, None]): Optional interval for collecting simulation states.
            report_progress (bool): Whether to report progress during the simulation.
            scheme (str): The numerical integration scheme to use (e.g.: 'rk45', 'euler').

        Returns:
            list: Collected states of the system as np.ndarray instances at each interval.

        """

        initial_condition = np.pad(
            self._initial_condition,
            (0, self._padding),
            mode='constant',
            constant_values=self._initial_condition[-1]
        )
        grid = CartesianGrid(
            bounds=[(0, 1), (0, self._spatial_size)],
            shape=(1, len(initial_condition)),
            periodic=self._periodicity
        )
        state = ScalarField(grid, data=initial_condition)

        trackers = []
        if report_progress:
            trackers.append('progress')

        state_container = []
        if collection_interval is not None:
            collecting_tracker = CallbackTracker(
                self._create_collect_callback(container=state_container),
                interval=collection_interval
            )

            trackers.append(collecting_tracker)

        pde = self._pde
        dt = dt or min(pde.calculate_delta_t(self._spatial_size / len(initial_condition)), 0.95 * collection_interval)
        if dt <= 0:
            raise ValueError(f'dt must be greater than 0 but it was: {dt}')
        
        end_state = pde.solve(state, t_range=t_range, dt=dt, tracker=trackers, adaptive=False, scheme=scheme)

        if self._is_padding_with_nans:
            state_container.append(np.concatenate((
                (np.empty(self._inner_radius) * np.nan),
                end_state.data[0, :len(self._initial_condition) - self._inner_radius].copy()
            )))
        else:
            state_container.append(end_state.data.copy())

        return state_container


class CartesianIVPSolver(IVPSolverBase):
    """
    A class for solving 2D initial value problems (IVPs) of diffusion partial differential equations (PDEs).

    This class is specifically designed to handle IVPs over a 2D cartesian grid.
    """

    def __init__(
            self,
            pde: DiffusionPDEBase,
            initial_condition: np.ndarray,
            spatial_height: float = 10.0,
            spatial_width: float = 10.0,
    ):
        """
        Args:
            pde (DiffusionPDEBase): The diffusion PDE to solve.
            initial_condition (np.ndarray): The initial condition for the PDE, a two-dimensional np.ndarray.
            spatial_height (float): The spatial height of the initial condition
                (The physical distance along the 0th axis).
            spatial_width (float): The spatial width of the initial condition
                (The physical distance along the 1st axis).
        """
        super().__init__(pde, initial_condition)
        self._spatial_height = spatial_height
        self._spatial_width = spatial_width

        self._periodicity = None
        for periodicity in [(False, False), (False, True), (True, False), (True, True)]:
            try:
                grid = CartesianGrid(
                    bounds=[(0, self._spatial_height), (0, self._spatial_width)],
                    shape=self._initial_condition.shape,
                    periodic=periodicity
                )
                Boundaries.from_data(grid, self.pde.bc)
                self._periodicity = periodicity
                break
            except PeriodicityError:
                continue

        if self._periodicity is None:
            raise ValueError('Invalid boundary condition in PDE.')

    @staticmethod
    def _create_collect_callback(container: MutableSequence[np.ndarray]):
        def callback(state: ScalarField, _time: float):
            container.append(state.data.copy())

        return callback

    def solve(
            self,
            t_range: float | Tuple[float, float],
            dt: float = 0.0001,
            collection_interval: float | None = None,
            report_progress: bool = True,
            scheme: str = 'rk45'
    ):
        """
        Solve the diffusion PDE over a given time range.

        Args:
            t_range (float | Tuple[float, float]): The time range for the simulation.
            dt (float): The time step for the simulation.
            collection_interval (float | None): Interval for collecting simulation states (optional).
            report_progress (bool): Whether to report progress during the simulation.
            scheme (str): The numerical integration scheme to use (e.g.: 'rk45', 'euler').

        Returns:
            A python list with the collected states (as np.ndarray). The last entry will always be the state
             returned by BasePDE's sole method.

        Raises:
            ValueError: If plot_kind is not in ('none', 'image'), or if a movie is not provided when needed.
        """

        grid = CartesianGrid(
            bounds=[(0, self._spatial_height), (0, self._spatial_width)],
            shape=self._initial_condition.shape,
            periodic=self._periodicity
        )
        state = ScalarField(
            grid,
            data=self._initial_condition
        )

        trackers = []

        if report_progress:
            trackers.append('progress')

        state_container = []
        if collection_interval is not None:
            collecting_tracker = CallbackTracker(
                CartesianIVPSolver._create_collect_callback(container=state_container),
                interval=collection_interval
            )

            trackers.append(collecting_tracker)

        pde = self.pde
        end_state = pde.solve(state, t_range=t_range, dt=dt, tracker=trackers, scheme=scheme)
        state_container.append(end_state.data.copy())

        return state_container
