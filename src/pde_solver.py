import matplotlib.pyplot as plt
import numpy as np
from pde import SphericalSymGrid, ScalarField, CallbackTracker
from pde.visualization.movies import Movie
from typing import Tuple

from src.diffusion_PDEs import DiffusionPDEBase


class SymmetricIVPPDESolver:
    """
    A solver for solving symmetric initial value problems (IVPs) of diffusion partial differential equations (PDEs).

    This class is specifically designed to handle IVPs with radial Symetry.

    Methods:
        solve(t_range, dt, collection_interval, plot_kind, report_progress): Solve the PDE over a given time range.

    Properties:
        pde (DiffusionPDEBase): Property for getting and setting the diffusion PDE.
        inner_radius (int): Property for getting the inner radius of radial symmetry.
    """

    def __init__(
            self,
            pde: DiffusionPDEBase,
            initial_condition: np.ndarray,
            spatial_size: int = 10,
            inner_radius=0,
            movie: str | Movie = None,
    ):
        """
        Initialize a SymmetricIVPPDESolver.

        Args:
            pde (DiffusionPDEBase): The diffusion PDE to solve.
            initial_condition (np.ndarray): The initial condition for the PDE, a one dimensional np.ndarray with the
                values along a radius, the first `inner_radius` number of values will be treated as if they were nans.
            spatial_size (int): The spatial size of the initial condition (The physical distance between the first and
            last value in the `initial_condition` array).
            inner_radius (int): The inner radius of the radial symmetry.
            movie (str | Movie): A movie file to record the simulation (optional).
        """
        self._pde = pde

        if initial_condition.ndim != 1:
            raise ValueError(f'`initial_condition` must be 1D, but it was: {initial_condition.ndim}')

        self._initial_condition = initial_condition
        self._padding = len(initial_condition)
        self._spatial_size = spatial_size
        self._inner_radius = inner_radius

        if isinstance(movie, str):
            movie = Movie(filename=movie, dpi=100)
        self._movie = movie

    @property
    def pde(self):
        return self._pde

    @pde.setter
    def pde(self, new_pde):
        self._pde = new_pde

    @property
    def inner_radius(self):
        return self._inner_radius

    def _create_image_plot_callback(self, min_v, max_v, fig, ax, **_):
        inner_radius_np_nan = np.empty(self.inner_radius) * np.nan

        def create_image(padded_array):
            fill_value = padded_array[-1]
            padded_array = np.append(padded_array, fill_value)

            padded_array = np.concatenate((inner_radius_np_nan, padded_array))

            image_range = np.arange(len(self._initial_condition))
            xs, ys = np.meshgrid(image_range, image_range)
            distances = np.hypot(xs, ys)

            quarter_image = (
                np.interp(distances.flat, np.arange(len(padded_array)), padded_array)
                .reshape(distances.shape)
            )
            half_image = np.concatenate([np.fliplr(quarter_image), quarter_image], axis=1)
            full_image = np.concatenate([np.flipud(half_image), half_image], axis=0)

            return full_image

        def callback(state, time):
            image = create_image(state.data)

            ax.clear()
            ax.imshow(image, vmin=min_v, vmax=max_v)
            ax.set_title(f'time: {time:.2f}')
            ax.axis('off')

            self._movie.add_figure(fig)

        return callback

    def _create_line_plot_callback(self, unpadded_length, fig, ax, **_):
        previous_states = []
        inner_radius_np_nan = np.empty(self.inner_radius) * np.nan

        def callback(state, time):
            uncropped_state = state.data[:unpadded_length]

            uncropped_state = np.concatenate((inner_radius_np_nan, uncropped_state))

            if len(previous_states) > 5:
                previous_states.pop(0)

            ax.clear()

            # Plot the previous 5 frames in grey with decreasing opacity
            for i, previous_state in enumerate(previous_states[::-1]):
                alpha = 1 - 0.2 * i
                ax.plot(previous_state, color='grey', alpha=alpha)

            ax.plot(uncropped_state, color='blue')

            previous_states.append(uncropped_state.copy())
            ax.set_title(f'time: {time:.2f}')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, len(self._initial_condition))
            self._movie.add_figure(fig)

        return callback

    def _create_collect_callback(self, container: list):
        inner_radius_np_nan = np.empty(self.inner_radius) * np.nan

        def callback(state, _time):
            state = state.data[:len(self._initial_condition) - self.inner_radius]
            container.append(np.concatenate((inner_radius_np_nan, state.copy())))

        return callback

    def solve(
            self,
            t_range: float | Tuple[float, float],
            dt: float = 0.0001,
            collection_interval: float | None = None,
            plot_kind: str = 'none',
            report_progress: bool = True,
    ) -> list[np.ndarray]:
        """
        Solve the diffusion PDE over a given time range.

        Args:
            t_range (float | Tuple[float, float]): The time range for the simulation.
            dt (float): The time step for the simulation.
            collection_interval (float | None): Interval for collecting simulation states (optional).
            plot_kind (str): Type of plot to visualize the simulation ('none', 'image', or 'line').
            report_progress (bool): Whether to report progress during the simulation.

        Returns:
            A python list with the collected radial profiles (as np.ndarray). The last entry will always be the radial
            profile returned by BasePDE's sole method.

        Raises:
            ValueError: If plot_kind is not in ('none', 'image', 'line'), or if a movie is not provided when needed.
        """

        if plot_kind not in ('none', 'image', 'line'):
            raise ValueError(f'plot_kind ({plot_kind}) was not recognized, it must be: one of "none", "line", "image".')

        if self._movie is None and plot_kind != 'none':
            raise ValueError('plot was True but there were no movie provided in the constructor.')

        initial_condition = np.pad(
            self._initial_condition[self.inner_radius:],
            (0, self._padding),
            mode='constant',
            constant_values=self._initial_condition[-1]
        )
        grid = SphericalSymGrid(radius=self._spatial_size, shape=initial_condition.shape)
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

        if plot_kind != 'none':
            dpi = self._movie.dpi
            fig_width, fig_height = 1280 / dpi, 720 / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

            if plot_kind == 'image':
                plot_callback = self._create_image_plot_callback
            else:
                plot_callback = self._create_line_plot_callback

            plotting_tracker = CallbackTracker(
                plot_callback(
                    unpadded_length=len(self._initial_condition) - self.inner_radius,
                    min_v=np.min(initial_condition),
                    max_v=np.max(initial_condition),
                    fig=fig,
                    ax=ax
                ),
                interval=0.01
            )

            trackers.append(plotting_tracker)

        pde = self._pde
        end_state = pde.solve(state, t_range=t_range, dt=dt, tracker=trackers)

        state_container.append(np.concatenate((
            (np.empty(self.inner_radius) * np.nan),
            end_state.data[:len(self._initial_condition) - self.inner_radius].copy()
        )))

        if plot_kind != 'none':
            self._movie.save()

        return state_container
