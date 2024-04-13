import os
from dataclasses import dataclass
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from lmfit import Parameters, create_params
from matplotlib import pyplot as plt
from pde import ScalarField, CartesianGrid, CallbackTracker
from pde.visualization.movies import Movie

from diffusion_PDEs import LinearDiffusivityPDE, LogisticDiffusionPDE, SigmoidDiffusivityPDE
from diffusion_array import DiffusionArray
from homogenizer import Homogenizer
from ivbcp import SymmetricIVBCPBase, VanillaSymmetricIVBCP, DerivativeSymmetricIVBCP, NormalDistributionSymmetricIVBCP
from ivp_solver import SymmetricIVPSolver
from optimizer import StaticMeshResolutionOptimizer
from step_widget import PipeLineWidget, ClippingWidget, StartFrameWidget, StartPlaceWidget, BackgroundRemovalWidget, \
    NormalizingWidget


def solve_pde(ivp: SymmetricIVBCPBase) -> np.ndarray:
    t_range = ivp.frames * ivp.sec_per_frame

    # ivp = ivp.resized(420)
    sol = np.array(
        SymmetricIVPSolver(ivp).solve(collection_interval=t_range / ivp.frames, t_range=t_range, report_progress=True,
                                      dt=1e-4)[:-1])

    frame_of_max_intensity = np.argmax(np.max(ivp.expected_values, axis=1))
    print('area 1:', np.sum(sol[frame_of_max_intensity, ivp.inner_radius:] - sol[0, ivp.inner_radius:]))
    print('area 1.2:', np.sum(sol[frame_of_max_intensity, ivp.inner_radius:] - sol[0, ivp.inner_radius:])
          * (ivp.spatial_size / ivp.width))
    print('area 2:', np.sum(sol[0, ivp.inner_radius:] - ivp.initial_condition))
    print('area 3: ', np.sum(sol[-1, ivp.inner_radius:] - ivp.initial_condition))
    return sol


@dataclass
class PlotSpecifier:
    rtp: np.ndarray
    label: str
    start_frame: int


def create_line_plot_movie(movie: str, plots: List[PlotSpecifier]):
    movie = Movie(filename=movie, dpi=100)
    dpi = movie.dpi
    fig_width, fig_height = 1280 / dpi, 720 / dpi

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, left=0.05, bottom=0.075)

    y_min = 0
    y_max = 1

    max_number_of_frames = max(ps.rtp.shape[0] + ps.start_frame for ps in plots)
    max_distance = max(ps.rtp.shape[1] for ps in plots)

    print(list(ps.rtp.shape for ps in plots))
    for frame in range(max_number_of_frames):
        ax.clear()
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, max_distance)
        ax.set_xlabel('Distance from origin (pixel)')
        ax.set_ylabel('Intensity')

        for ps in plots:
            if frame < ps.start_frame or frame - ps.start_frame > ps.rtp.shape[0] - 1:
                ax.plot(np.nan, label=ps.label)
            else:
                ax.plot(ps.rtp[frame - ps.start_frame], label=ps.label)

        ax.set_title(f'frame: {frame:3d}')
        plt.legend(loc='upper right')
        movie.add_figure(fig)

    movie.save()


def plot_runner(
        darr_path,
        pde_name,
        vanilla_params,
        der_bc_params,
        norm_added_params
):
    pdename_to_pde = {
        'Logistic Diffusion': lambda: LogisticDiffusionPDE(),
        'Sigmoid Diffusion': lambda: SigmoidDiffusivityPDE(),
        'Linear Diffusion': lambda: LinearDiffusivityPDE(),
    }
    # darr_path = r'G:\rost\Ca2+_laser\raw_data\1133_3_laser@30sec006.nd2'

    pipeline = PipeLineWidget(None, display_mode=False)
    pipeline.add_step(ClippingWidget())
    pipeline.add_step(StartFrameWidget())
    pipeline.add_step(StartPlaceWidget())
    pipeline.add_step(BackgroundRemovalWidget())
    pipeline.add_step(NormalizingWidget())

    darr = DiffusionArray(darr_path).channel(0)
    darr, start_frame, start_place = pipeline.apply_pipeline(darr)
    # noinspection PyTypeChecker
    rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)

    plots = [PlotSpecifier(rtp.ndarray, 'Data', 0)]

    if vanilla_params:
        vanilla_pde = pdename_to_pde[pde_name]()
        vanilla_pde.update_parameters(vanilla_params)
        vanilla_sol = solve_pde(VanillaSymmetricIVBCP(rtp, start_frame, vanilla_pde))
        plots.append(PlotSpecifier(vanilla_sol, 'Vanilla', np.argmax(np.max(rtp, axis=1))))

    if der_bc_params:
        derivative_pde = pdename_to_pde[pde_name]()
        derivative_pde.update_parameters(der_bc_params)
        derivative_sol = solve_pde(DerivativeSymmetricIVBCP(rtp, start_frame, derivative_pde))
        plots.append(PlotSpecifier(derivative_sol, 'Derivative BC', start_frame))

    if norm_added_params:
        norm_added_pde = pdename_to_pde[pde_name]()
        norm_added_pde.update_parameters(norm_added_params)
        norm_added_sol = solve_pde(NormalDistributionSymmetricIVBCP(rtp, start_frame, norm_added_pde))
        plots.append(PlotSpecifier(norm_added_sol, 'Normal dist. added', start_frame))

    create_line_plot_movie(
        f'../video/{os.path.basename(darr_path)} {pde_name}.mp4',
        plots
    )


def main():
    darr_paths = []
    for directory in [r'G:\rost\Ca2+_laser\raw_data', r'G:\rost\kozep\raw_data', r'G:\rost\sarok\raw_data']:
        for root, _, files in os.walk(directory):
            darr_paths.extend([os.path.join(root, file) for file in files if file.endswith('.nd2')])
    darr_paths = map(str, darr_paths)

    def create_params(row):
        params = Parameters()
        for col, value in row.items():
            if not pd.isnull(value):
                params.add(col, value)

        return params

    df = pd.read_csv('../params/params_median.csv', sep=',')
    filename_to_path = {filename.split('\\')[-1].strip(): filename for filename in darr_paths}
    df['Filename'] = df['Filename'].map(filename_to_path)

    selected_columns = ['diffusivity', 'mu', 'alpha', 'beta', 'lambda_term', 's']
    pdetype_to_name = {
        'LogisticDiffusionPDE': 'Logistic Diffusion',
        'SigmoidDiffusivityPDE': 'Sigmoid Diffusion',
        'LinearDiffusivityPDE': 'Linear Diffusion',
    }
    pde_types = df['Eqname'].unique()
    filenames = df['Filename'].unique()
    for filename in filenames:
        for pde_type in pde_types:
            print('Currently plotting: ', filename, pde_type)
            filtered_df = df[(df['Filename'] == filename) & (df['Eqname'] == pde_type)]

            vanilla_df = filtered_df[filtered_df['Type'] == 'Vanilla']
            der_bc_df = filtered_df[filtered_df['Type'] == 'Derivative BC']
            norm_added_df = filtered_df[filtered_df['Type'] == 'Norm Dist']

            vanilla_params = create_params(vanilla_df.iloc[0][selected_columns]) if not vanilla_df.empty else None
            der_bc_params = create_params(der_bc_df.iloc[0][selected_columns]) if not der_bc_df.empty else None
            norm_added_params = create_params(
                norm_added_df.iloc[0][selected_columns]) if not norm_added_df.empty else None

            plot_runner(
                filename,
                pdetype_to_name[pde_type],
                vanilla_params,
                der_bc_params,
                norm_added_params
            )
            raise Exception()


def plot_single():
    # darr = DiffusionArray('G:\\rost\\Ca2+_laser\\raw_data\\1133_3_laser@30sec006.nd2').channel(0)
    darr = DiffusionArray('G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl018.nd2').channel(0)
    # darr = DiffusionArray('G:\\rost\\sarok\\raw_data\\1472_4_laser@30sec004.nd2').channel(0)
    pipeline = PipeLineWidget(None, display_mode=False)
    pipeline.add_step(ClippingWidget())
    pipeline.add_step(StartFrameWidget())
    pipeline.add_step(StartPlaceWidget())
    pipeline.add_step(BackgroundRemovalWidget())
    pipeline.add_step(NormalizingWidget())
    darr, start_frame, start_place = pipeline.apply_pipeline(darr)
    # noinspection PyTypeChecker
    rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)

    for method in ['SLSQP', 'COBYLA', 'Powell', 'leastsq']:
        print('method=', method)

        derivative_pde = LogisticDiffusionPDE()
        parameters = create_params(
            diffusivity={'value': 300, 'min': 0, 'max': 1000},
            lambda_term={'value': 0.5, 'min': 0, 'max': 1},
            alpha={'value': 1, 'min': 0.01, 'max': 10},
        )

        optimizer = StaticMeshResolutionOptimizer(VanillaSymmetricIVBCP(rtp, start_frame, derivative_pde))
        optimizer.optimize(parameters, report_progress=True, method=method, dt=1e-4)

        print('-----')
        print(optimizer.optimal_mse)
        print(optimizer.optimal_parameters)
        print(optimizer.optimal_rsqrd)
        print(optimizer.time_required)
        print(optimizer.number_of_iterations)
        print('\n\n')

    # derivative_pde.update_parameters(der_bc_params)
    # derivative_sol = solve_pde(DerivativeSymmetricIVP(rtp, start_frame, derivative_pde))
    # plots = [PlotSpecifier(rtp.ndarray, 'Data', 0)]
    # plots.append(PlotSpecifier(derivative_sol, 'Derivative BC', start_frame))
    # create_line_plot_movie(
    #     f'../video/test.mp4',
    #     plots
    # )


def idk():
    # pde = LinearDiffusivityPDE()
    # pde.parameters = create_params(
    #     diffusivity=20,
    #     mu=0,
    # )

    mu = 0.00  # Mean
    sigma = 0.25  # Standard deviation
    x = np.linspace(0, 3, 300)  # Range of x values
    # pdf_values = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    pdf_values = x * 0
    # pdf_values = 0 * x + 0.4

    pde = LogisticDiffusionPDE()
    pde.parameters = create_params(
        diffusivity=2,
        lambda_term=1,
        alpha=5,
    )

    # pdf_values += 0.1 * np.sin(4 * np.cos(6 * x)) * np.cos(8 * np.sin(12 * x))
    # pde = SigmoidDiffusivityPDE()
    # pde.parameters = create_params(
    #     diffusivity=-5,
    #     mu=0.04,
    #     gamma=5,
    #     beta=-2
    # )

    # pde.bc = [{'value': pdf_values[0]}, {'value': 0}]
    def _bc_function(_adjacent_value, _dx, _x, t):
        return 0.05 if t < 0.5 else 0

    pde.bc = [{'derivative_expression': _bc_function}, {'derivative': 0}]
    pde.bc = [{'value': 1}, {'value': 0}]

    grid = CartesianGrid(
        bounds=[(0, 10)],
        shape=pdf_values.shape,
        periodic=[False]
    )
    state = ScalarField(
        grid,
        data=pdf_values
    )

    trackers = ['progress']

    state_container = []

    def callback(state: ScalarField, _time: float):
        state_container.append(state.data.copy())

    interval = 0.15
    t_range = 2
    collecting_tracker = CallbackTracker(
        callback,
        interval=interval
    )

    plt.figure(dpi=300)

    trackers.append(collecting_tracker)
    pde.solve(state, t_range=t_range, dt=1e-5, tracker=trackers)
    cmap = matplotlib.colormaps['plasma']
    for i, state in enumerate(state_container):
        color = cmap(i / (len(state_container) - 1))
        plt.plot(x[:150], state[:150], color=color)

    plt.legend([f't={i * interval:.2f}' for i in range(int(np.ceil(t_range / interval)))], loc='upper right', )
    plt.xlabel('x')
    plt.ylabel('c')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    plot_single()
    # idk()
