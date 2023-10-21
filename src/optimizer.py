import math

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, Minimizer
from pde.visualization.movies import Movie

from src.pde_solver import SymmetricIVPPDESolver


class Optimizer:
    """
    A class for numerically optimizing the parameters of diffusion partial differential equations (PDEs).

    This class is designed to find the optimal parameters for PDEs to fit expected diffusion profiles.

    Methods:
        optimize(parameters, max_iter, tol, report_progress, method):
            Optimize the PDE parameters to fit the expected values.
        plot_profile_comparisons(movie, frame_offset, optimal_solution):
            Plot comparisons between expected and optimized profiles.

    Properties:
        optimal_solution (np.ndarray): Property to get the optimized diffusion profiles.
        optimal_mse (float): Property to get the mean squared error of the optimization.
        optimal_parameters (Parameters): Property to get the optimized parameters.
    """

    def __init__(
            self,
            pde_solver: SymmetricIVPPDESolver,
            expected_values: np.ndarray,
            t_range: float
    ):
        """
        Initialize an Optimizer.

        Args:
            pde_solver (SymmetricIVPPDESolver): A solver for symmetric initial value problems.
            expected_values (np.ndarray): The expected diffusion profiles as a 2d np.ndarray with (time, x) indices.
            t_range (float): The time range for the simulation.
        """
        if expected_values.ndim != 2:
            raise ValueError(f'expected_values must be 2d, but it was: {expected_values.ndim}')

        self._pde_solver = pde_solver
        self._expected_values = expected_values
        self._t_range = t_range

        self._optimal_solution = None
        self._optimal_mse = -1
        self._optimal_parameters = None

    @property
    def optimal_solution(self) -> np.ndarray:
        if self._optimal_solution is None:
            raise AttributeError('The method optimize should be called first.')
        return self._optimal_solution

    @property
    def optimal_mse(self) -> float:
        if self._optimal_mse < 0:
            raise AttributeError('The method optimize should be called first.')
        return self._optimal_mse

    @property
    def optimal_parameters(self) -> Parameters:
        if self._optimal_parameters is None:
            raise AttributeError('The method optimize should be called first.')
        return self._optimal_parameters

    def optimize(
            self,
            parameters: Parameters,
            max_iter: int = 1_000,
            tol: float = 1e-4,
            report_progress: bool = True,
            method: str | None = 'SLSQP'
    ) -> Parameters:
        """
        Optimize the PDE parameters to fit the expected diffusion profiles.

        Args:
            parameters (Parameters): Initial parameters for the optimization.
            max_iter (int): Maximum number of iterations for optimization.
            tol (float): Tolerance for optimization termination.
            report_progress (bool): Whether to report progress during optimization.
            method (str | None): Optimization method (e.g., 'SLSQP', 'leastsq', 'Powell').

        Returns:
            Parameters: The optimized parameters.
        """
        expected_values = self._expected_values[:, self._pde_solver.inner_radius:]

        def target_function(params: Parameters):
            self._pde_solver.pde.update_parameters(params)

            sol = self._pde_solver.solve(
                collection_interval=self._t_range / (len(expected_values) - 1),
                t_range=self._t_range,
                report_progress=False,
                dt=0.00001
            )[:-1]

            return expected_values - np.array(sol)[:, self._pde_solver.inner_radius:]

        optimal_params = None
        optimal_mse = math.inf

        def iter_cb(params, iter_number, resid):
            mse = (resid ** 2).sum()
            nonlocal optimal_mse

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

        minimizer = Minimizer(target_function, parameters, iter_cb=iter_cb, max_nfev=max_iter)

        if method in ('SLSQP', 'COBYLA'):
            minimizer.minimize(method=method, tol=tol)

        elif method in ('leastsq', 'Powell'):
            minimizer.minimize(method=method, xtol=tol, ftol=tol)

        else:
            raise ValueError(f'Unsupported method: {method}')

        self._pde_solver.pde.update_parameters(optimal_params)
        self._optimal_solution = self._pde_solver.solve(
            collection_interval=self._t_range / (len(self._expected_values) - 1),
            t_range=self._t_range,
            report_progress=False,
            dt=0.00001
        )[:-1]
        self._optimal_parameters = optimal_params
        self._optimal_mse = optimal_mse

        return optimal_params  # type: ignore

    def plot_profile_comparisons(
            self,
            movie: Movie | str,
            frame_offset: int = 0,
            optimal_solution: np.ndarray = None
    ) -> None:
        """
        Plot comparisons between expected and optimized diffusion profiles.

        Args:
            movie (Movie | str): A movie to save the comparisons.
            frame_offset (int): Frame offset for labeling.
            optimal_solution (np.ndarray): Optional optimized diffusion profiles.
        """

        if self.optimal_solution is None and optimal_solution is None:
            raise ValueError('The optimal_solution was not provided nor has the "optimize" method been called.')

        optimal_solution = optimal_solution if optimal_solution is not None else self.optimal_solution

        if isinstance(movie, str):
            movie = Movie(filename=movie, dpi=100)

        dpi = movie.dpi
        fig_width, fig_height = 1280 / dpi, 720 / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        y_min = min(
            np.min(np.array(optimal_solution)[:, self._pde_solver.inner_radius:]),
            np.min(self._expected_values)
        )
        y_max = max(
            np.max(np.array(optimal_solution)[:, self._pde_solver.inner_radius:]),
            np.max(self._expected_values)
        )

        for i, (expected_frame, solution_frame) in enumerate(zip(self._expected_values, optimal_solution)):
            ax.clear()
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, len(self._expected_values[0]))

            a = 0.7
            for j in range(i - 1, max(i - 5, 0), -1):
                ax.plot(optimal_solution[j], color='blue', alpha=a)
                a -= 0.2

            ax.plot(expected_frame, color='red')
            ax.plot(solution_frame, color='blue')
            ax.set_title(f'frame: {int(frame_offset + i):3d}')

            movie.add_figure(fig)

        movie.save()
