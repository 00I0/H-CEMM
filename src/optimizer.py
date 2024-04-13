import math
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from lmfit import Parameters, Minimizer

from ivbcp import SymmetricIVBCPBase
from src.ivp_solver import SymmetricIVPSolver


class OptimizerBase(ABC):
    """
    Base class for optimizers of symmetric initial value boundary condition problems (IVPBCs).

    This abstract base class defines the common properties and methods for optimizers of symmetric IVPBCs.
    Subclasses should implement the `optimize` method to provide specific optimization algorithms.

    Args:
        ivp (SymmetricIVBCPBase): The symmetric IVPBC to be optimized.

    Properties:
        optimal_solution (np.ndarray): The optimal solution obtained after optimization.
        optimal_mse (float): The optimal mean squared error achieved during optimization.
        optimal_rsqrd (float): The optimal R-squared value achieved during optimization.
        optimal_parameters (Parameters): The optimal parameters obtained during optimization.
        number_of_iterations (int): The number of iterations performed during optimization.
        time_required (int): The time required for optimization, in seconds.
        stats (list): A list of dictionaries containing optimization statistics for each resolution.
    """

    def __init__(
            self,
            ivp: SymmetricIVBCPBase
    ):
        self._ivp_solver = SymmetricIVPSolver(ivp)
        self._ivp = ivp

        self._optimal_solution: Optional[np.ndarray] = None
        self._optimal_mse: float = -1
        self._optimal_rsqrd: Optional[float] = None
        self._optimal_parameters: Optional[Parameters] = None
        self._number_of_iterations: int = -1
        self._time_required: int = -1
        self._message = ''
        self._stats = []

    @property
    def optimal_solution(self) -> np.ndarray:
        if self._optimal_solution is None:
            raise AttributeError('The method `optimize` should be called first.')
        return self._optimal_solution

    @property
    def optimal_mse(self) -> float:
        if self._optimal_mse < 0:
            raise AttributeError('The method `optimize` should be called first.')
        return self._optimal_mse

    @property
    def optimal_rsqrd(self) -> float:
        if self._optimal_rsqrd is None:
            raise AttributeError('The method `optimize` should be called first.')
        return self._optimal_rsqrd

    @property
    def stats(self) -> list:
        """
        A list of dictionaries containing optimization statistics for each resolution.

        Returns:
            list: A list of dictionaries containing optimization
        """
        return self._stats

    @property
    def optimal_parameters(self) -> Parameters:
        if self._optimal_parameters is None:
            raise AttributeError('The method `optimize` should be called first.')
        return self._optimal_parameters

    @property
    def number_of_iterations(self) -> int:
        if self._number_of_iterations < 0:
            raise AttributeError('The method `optimize` should be called first.')
        return self._number_of_iterations

    @property
    def time_required(self) -> int:
        if self._time_required < 0:
            raise AttributeError('The method `optimize` should be called first.')
        return self._time_required

    @abstractmethod
    def optimize(
            self,
            parameters: Parameters,
            max_iter: int = 1_000,
            tol: float = 1e-4,
            report_progress: bool = False,
            method: str | None = 'SLSQP',
            dt: float = 1e-4,
    ):
        """
        Optimize the specified parameters of the IVBCP using optimization settings.

        Args:
            parameters (Parameters): The initial parameters for the optimization.
            max_iter (int): The maximum number of iterations for the optimization (default: 1000).
            tol (float): The tolerance for the optimization (default: 1e-4).
            report_progress (bool): Whether to report the progress during optimization (default: False).
            method (str | None): The optimization method to use (default: 'SLSQP').
            dt (float): The time step for the numerical solver (default: 1e-4).

        Raises:
            NotImplementedError: This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError


class StaticMeshResolutionOptimizer(OptimizerBase):
    """
    Optimizer for symmetric IVPBCs with a static mesh resolution.

    This class implements the `optimize` method from the `OptimizerBase` class and performs optimization of the
    IVBCP parameters for a fixed mesh resolution.

    """

    def optimize(
            self,
            parameters: Parameters,
            max_iter: int = 1_000,
            tol: float = 1e-4,
            report_progress: bool = False,
            method: str | None = 'SLSQP',
            dt: float = 1e-4,
            scheme: str = 'rk45'
    ) -> Parameters:
        start_time = time.time()
        expected_values = self._ivp.expected_values
        t_range = self._ivp.frames * self._ivp.sec_per_frame

        def target_function(params: Parameters):
            self._ivp_solver.pde.parameters = params

            sol = self._ivp_solver.solve(
                collection_interval=t_range / (len(expected_values) - 1),
                t_range=t_range,
                report_progress=False,
                dt=dt,
                scheme=scheme
            )[:-1]

            return expected_values - np.array(sol)[:, self._ivp.inner_radius:]

        optimal_params: Optional[Parameters] = None
        optimal_mse = math.inf

        def iter_cb(params, iter_number, resid):
            mse = (resid ** 2).sum()
            nonlocal optimal_mse

            self._number_of_iterations = iter_number

            if mse < optimal_mse:
                optimal_mse = mse

                nonlocal optimal_params
                optimal_params = params.copy()

            if report_progress:
                print(
                    f'{iter_number :4d}: '
                    f'{str([f"{param.name} {param.value:4.4f}" for param in params.values()]):120s} '
                    f'MSE: {mse:6.4f}'
                )

        # noinspection PyTypeChecker
        minimizer = Minimizer(target_function, parameters.copy(), iter_cb=iter_cb, max_nfev=max_iter)

        if method in ('SLSQP', 'COBYLA', 'Powell'):
            minimizer.minimize(method=method, tol=tol)

        elif method in ('leastsq',):
            minimizer.minimize(method=method, xtol=tol, ftol=tol)

        else:
            raise ValueError(f'Unsupported method: {method}')

        self._ivp_solver.pde.parameters = optimal_params
        self._optimal_solution = self._ivp_solver.solve(
            collection_interval=t_range / (len(self._ivp.expected_values) - 1),
            t_range=t_range,
            report_progress=False,
            dt=dt
        )[:-1]
        self._optimal_parameters = optimal_params
        self._optimal_mse = optimal_mse / self._ivp.expected_values.shape[1]

        pred = np.array(self._optimal_solution)[:, self._ivp.inner_radius:]
        self._optimal_rsqrd = (
                1
                - np.sum((expected_values - pred) ** 2)
                / np.sum((expected_values - np.mean(expected_values)) ** 2)
        )
        self._time_required = time.time() - start_time

        return optimal_params


class DynamicMeshResolutionOptimizer(OptimizerBase):
    """
    Optimizer for symmetric IVPBCs with dynamic mesh resolution.

    This class extends the `OptimizerBase` class and implements the `optimize` method to perform optimization
    of the IVBCP parameters with varying mesh resolutions. The optimization process starts with a low resolution
    and iteratively increases the resolution until a specified maximum resolution is reached.
    """

    @property
    def message(self):
        return self._message

    def optimize(
            self,
            parameters: Parameters,
            max_iterations_per_resolution: int = 50,
            tol: float = 1e-4,
            report_progress: bool = False,
            method: str | None = 'leastsq',
            dt: float = 1e-4,
            max_resolution=-1,
            min_resolution=40,
    ) -> Parameters:
        params = parameters.copy()

        if max_resolution == -1: max_resolution = self._ivp.width
        if min_resolution == -1: min_resolution = self._ivp.width
        if min_resolution > max_resolution:
            raise ValueError('`min_resolution` cannot be larger than `max_resolution`')

        total_number_of_iterations = 0
        total_time = 0

        for resolution in [
            min(min_resolution * 2 ** n, max_resolution)
            for n in range(int(np.ceil(np.log2(max_resolution / min_resolution))) + 1)
        ]:
            ivp = self._ivp.resized(resolution)

            optimizer = StaticMeshResolutionOptimizer(ivp)
            try:
                params = optimizer.optimize(
                    params,
                    max_iter=max_iterations_per_resolution,
                    tol=tol,
                    report_progress=False,
                    method=method,
                    dt=dt
                )
            except Exception as exp:
                raise RuntimeError(f'An exception has occurred at pde={type(self._ivp_solver.pde).__name__} '
                                   f'file={self._ivp.file_name} resolution={resolution}') from exp

            total_number_of_iterations += optimizer.number_of_iterations
            total_time += optimizer.time_required

            stats = {
                'pde': type(self._ivp_solver.pde).__name__,
                'file': self._ivp.file_name,
                'resolution': resolution,
                'iterations': optimizer.number_of_iterations,
                'time': optimizer.time_required,
                'mse': optimizer.optimal_mse,
                'rsqrd': optimizer.optimal_rsqrd,
                **{name: param.value for name, param in params.items()}
            }
            self._stats.append(stats)

            if report_progress:
                message = (
                    f'pde={type(self._ivp_solver.pde).__name__}, '
                    f'file={self._ivp.file_name}, '
                    f'resolution={resolution}, '
                    f'iterations={optimizer.number_of_iterations}, '
                    f'time={optimizer.time_required}, '
                    f'mse={optimizer.optimal_mse}, '
                    f'rsqrd={optimizer.optimal_rsqrd}, '
                )
                message += ', '.join([f'{name}={param.value}' for name, param in params.items()])
                print(message)
                self._message += message + '\n'

        # noinspection PyUnboundLocalVariable
        self._ivp_solver.pde.parameters = optimizer.optimal_parameters
        self._optimal_solution = optimizer.optimal_solution
        self._optimal_parameters = optimizer.optimal_parameters
        self._optimal_mse = optimizer.optimal_parameters
        self._optimal_rsqrd = optimizer.optimal_rsqrd
        self._number_of_iterations = total_number_of_iterations
        self._time_required = total_time

        return self.optimal_parameters
