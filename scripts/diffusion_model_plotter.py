import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from lmfit import Parameters
from matplotlib import pyplot as plt
from pde.visualization.movies import Movie

from core.diffusion_array import DiffusionArray
from core.homogenizer import Homogenizer
from core.step_widget import PipeLineWidget, ClippingWidget, StartFrameWidget, StartPlaceWidget, \
    BackgroundRemovalWidget, \
    NormalizingWidget
from ivbcps.diffusion_PDEs import LinearDiffusivityPDE, LogisticDiffusionPDE, SigmoidDiffusivityPDE, MixedPDE
from ivbcps.ivbcp import SymmetricIVBCPBase, VanillaSymmetricIVBCP, DerivativeSymmetricIVBCP, \
    NormalDistributionSymmetricIVBCP
from ivbcps.ivp_solver import SymmetricIVPSolver


def solve_pde(ivp: SymmetricIVBCPBase) -> np.ndarray:
    """
    Solves a PDE initial value boundary condition problem (IVBCP).

    Args:
        ivp (SymmetricIVBCPBase): The initial value boundary condition problem to solve.

    Returns:
        np.ndarray: The solution array.
    """
    t_range = ivp.frames * ivp.sec_per_frame

    sol = np.array(
        SymmetricIVPSolver(ivp).solve(
            collection_interval=t_range / ivp.frames,
            t_range=t_range,
            report_progress=True,
            dt=None)[:-1]
    )

    return sol


@dataclass
class PlotSpecifier:
    rtp: np.ndarray
    label: str
    start_frame: int


def create_line_plots(
        movie: str,
        plots: List[PlotSpecifier],
        save_individual_frames_instead: bool = False
):
    """
    Creates line plots for the given data and saves them as a movie or individual frames.

    Args:
        movie (str): The filename for the movie or individual frames.
        plots (List[PlotSpecifier]): The list of plots to create.
        save_individual_frames_instead (bool): Whether to save individual frames instead of a movie. Defaults to False.
    """
    plt.close("all")

    colors = ['mediumblue', 'firebrick', 'magenta', 'red', 'green', 'pink', 'brown', 'cyan', 'purple', 'yellow', 'lime',
              'black']

    if not save_individual_frames_instead:
        movie = Movie(filename=movie, dpi=300, framerate=15)
        dpi = movie.dpi
        fig_width, fig_height = 1280 / 125, 720 / 125
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    else:
        directory = os.path.dirname(movie)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig, ax = plt.subplots(dpi=300)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, left=0.05, bottom=0.075)

    y_min = 0
    y_max = 1

    max_number_of_frames = max(ps.rtp.shape[0] + ps.start_frame for ps in plots)
    max_distance = max(ps.rtp.shape[1] for ps in plots)

    print(list(ps.rtp.shape for ps in plots))
    for t, frame in enumerate(sorted(list(range(6, max_number_of_frames)) * 5)):
        ax.clear()
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, max_distance)
        ax.set_xlabel('Távolság a sebzéstől (pixel)')
        ax.set_ylabel('Koncentráció')

        for ps, color in zip(plots, colors):
            if frame < ps.start_frame or frame - ps.start_frame > ps.rtp.shape[0] - 1:
                ax.plot(np.nan, label=ps.label, color=color)
            else:
                if ps.label == 'Sigmoid Diffusion':
                    ax.plot(ps.rtp[frame - ps.start_frame], linestyle='--', dashes=(6, 8), linewidth=1.5,
                            label=ps.label, color=color)
                else:
                    ax.plot(ps.rtp[frame - ps.start_frame], label=ps.label, color=color)

        ax.set_title(f'Képkocka: {frame:3d}')
        plt.legend(loc='upper right')
        plt.tight_layout()

        if not save_individual_frames_instead:
            movie.add_figure(fig)
        else:
            directory = os.path.dirname(movie)
            plt.savefig(os.path.join(directory, f'{t}.png'))

    if not save_individual_frames_instead:
        movie.save()


def plot_runner(
        darr_path: str,
        pde_name: str,
        vanilla_params: List[tuple] | None,
        der_bc_params: List[tuple] | None,
        norm_added_params: List[tuple] | None,
        save_individual_frames: bool = False,
        output_dir: str = '../video'
):
    """
    Runs the plotting pipeline for the given parameters and saves the resulting plots.

    Args:
        darr_path (str): The path to the diffusion array file.
        pde_name (str): The name of the PDE.
        vanilla_params (List[tuple] | None): Parameters for the vanilla PDEs.
        der_bc_params (List[tuple] | None): Parameters for the derivative boundary condition PDEs.
        norm_added_params (List[tuple] | None): Parameters for the normal distribution added PDEs.
        save_individual_frames (bool): Whether to save individual frames instead of a movie. Defaults to False.
        output_dir (str): The directory to save the output files. Defaults to '../video'.
    """
    if vanilla_params and not isinstance(vanilla_params, list):
        vanilla_params = [('Vanilla', vanilla_params, pde_name)]
    if der_bc_params and not isinstance(der_bc_params, list):
        der_bc_params = [('Derivative BC', der_bc_params, pde_name)]
    if norm_added_params and not isinstance(norm_added_params, list):
        norm_added_params = [('Normal dist. added', norm_added_params, pde_name)]

    pdename_to_pde = {
        'Logistic Diffusion': lambda: LogisticDiffusionPDE(),
        'Sigmoid Diffusion': lambda: SigmoidDiffusivityPDE(),
        'Linear Diffusion': lambda: LinearDiffusivityPDE(),
        'Mixed Models': lambda: MixedPDE(),
    }

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
        for name, vanilla_param, pde_name in vanilla_params:
            vanilla_pde = pdename_to_pde[pde_name]()
            vanilla_pde.parameters = vanilla_param
            vanilla_sol = solve_pde(VanillaSymmetricIVBCP(rtp, start_frame, vanilla_pde))
            plots.append(PlotSpecifier(vanilla_sol, name, np.argmax(np.max(rtp, axis=1))))

    if der_bc_params:
        for name, der_bc_param, pde_name in der_bc_params:
            derivative_pde = pdename_to_pde[pde_name]()
            derivative_pde.parameters = der_bc_param
            derivative_sol = solve_pde(DerivativeSymmetricIVBCP(rtp, start_frame, derivative_pde))
            plots.append(PlotSpecifier(derivative_sol, name, start_frame))

    if norm_added_params:
        for name, norm_added_param, pde_name in norm_added_params:
            norm_added_pde = pdename_to_pde[pde_name]()
            norm_added_pde.parameters = norm_added_param
            norm_added_sol = solve_pde(NormalDistributionSymmetricIVBCP(rtp, start_frame, norm_added_pde))
            plots.append(PlotSpecifier(norm_added_sol, name, start_frame))

    base_filename = os.path.splitext(os.path.basename(darr_path))[0]
    if save_individual_frames:
        output_path = os.path.join(output_dir, base_filename, f'{base_filename} {pde_name}.mp4')
    else:
        output_path = os.path.join(output_dir, f'{base_filename}.mp4')

    create_line_plots(output_path, plots)


def main(parameter_csv: str, output_dir: str = '../video'):
    """
    Main function to run the plot runner for multiple diffusion array files.

    Args:
        parameter_csv (str): The path to the CSV file containing parameter values.
        output_dir (str): The directory to save the output files. Defaults to '../video'.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('../config.json', 'r') as json_file:
        config_file = json.load(json_file)

    darr_paths = []
    for details in config_file.values():
        path = details['path']
        darr_paths.append(path)

    def create_params(row):
        params = Parameters()
        for col, value in row.items():
            if not pd.isnull(value):
                params.add(col, value)

        return params

    df = pd.read_csv(parameter_csv, sep=',')
    filename_to_path = {filename.split('\\')[-1].strip(): filename for filename in darr_paths}
    df['Filename'] = df['Filename'].map(filename_to_path)

    selected_columns = ['diffusivity', 'mu', 'alpha', 'beta', 'lambda_term', 'gamma', 'D', 'phi']
    pdetype_to_name = {
        'LogisticDiffusionPDE': 'Logistic Diffusion',
        'LinearDiffusivityPDE': 'Linear Diffusion',
        'SigmoidDiffusivityPDE': 'Sigmoid Diffusion',
        'MixedPDE': 'Mixed Models'
    }
    pde_types = df['Eqname'].unique()
    filenames = df['Filename'].unique()
    print('These pde types were found in the given file:', pde_types)

    for filename in filenames:
        print('Currently plotting: ', filename)
        vanilla_params = []
        for pde_type in pde_types:
            filtered_df = df[(df['Filename'] == filename) & (df['Eqname'] == pde_type)]

            vanilla_df = filtered_df[filtered_df['Type'] == 'vanilla']

            vanilla_param = create_params(vanilla_df.iloc[0][selected_columns]) if not vanilla_df.empty else None
            vanilla_params.append((pdetype_to_name[pde_type], vanilla_param, pdetype_to_name[pde_type]))

        plot_runner(
            filename,
            '',
            vanilla_params=vanilla_params,
            der_bc_params=None,
            norm_added_params=None,
            output_dir=output_dir
        )


if __name__ == '__main__':
    # This script creates a video for each initial value problem, based on the optimized parameters contained
    # in `../params/adaptive_delta_t.csv`, and puts them into the `../video` directory

    #           -----     ----    ---   ---   --  --  Warning  --  --  ---   ---    ----     -----
    # Since this script also solves the initial value problems, the runtime could be long.

    main('../params/adaptive_delta_t.csv', '../video/adaptive_delta_t')
