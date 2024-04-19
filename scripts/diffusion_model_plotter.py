import os
from dataclasses import dataclass
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from lmfit import Parameters
from matplotlib import pyplot as plt, gridspec
from pde.visualization.movies import Movie

from core.diffusion_array import DiffusionArray
from core.homogenizer import Homogenizer
from core.step_widget import PipeLineWidget, ClippingWidget, StartFrameWidget, StartPlaceWidget, \
    BackgroundRemovalWidget, \
    NormalizingWidget
from ivbcps.diffusion_PDEs import LinearDiffusivityPDE, LogisticDiffusionPDE, SigmoidDiffusivityPDE
from ivbcps.ivbcp import SymmetricIVBCPBase, VanillaSymmetricIVBCP, DerivativeSymmetricIVBCP, \
    NormalDistributionSymmetricIVBCP
from ivbcps.ivp_solver import SymmetricIVPSolver


def solve_pde(ivp: SymmetricIVBCPBase) -> np.ndarray:
    t_range = ivp.frames * ivp.sec_per_frame

    sol = np.array(
        SymmetricIVPSolver(ivp).solve(collection_interval=t_range / ivp.frames, t_range=t_range, report_progress=True,
                                      dt=None)[:-1])

    return sol


@dataclass
class PlotSpecifier:
    rtp: np.ndarray
    label: str
    start_frame: int


def create_line_plots(movie: str, plots: List[PlotSpecifier], save_individual_frames_instead: bool = False):
    plt.close("all")

    if not save_individual_frames_instead:
        movie = Movie(filename=movie, dpi=300)
        dpi = movie.dpi
        fig_width, fig_height = 1280 / dpi, 720 / dpi
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
    for t, frame in enumerate(range(max_number_of_frames)):
        ax.clear()
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, max_distance)
        ax.set_title(f'T = {t:3d}')
        ax.set_xlabel('Távolság az origótól (pixel)')
        ax.set_ylabel('Koncentráció')

        for ps in plots:
            if frame < ps.start_frame or frame - ps.start_frame > ps.rtp.shape[0] - 1:
                ax.plot(np.nan, label=ps.label)
            else:
                if ps.label == 'Sigmoid Diffusion':
                    ax.plot(ps.rtp[frame - ps.start_frame], linestyle='--', dashes=(6, 8), linewidth=1.5,
                            label=ps.label)
                else:
                    ax.plot(ps.rtp[frame - ps.start_frame], label=ps.label)

        ax.set_title(f'Képkocka: {frame:3d}')
        plt.legend(loc='upper right')
        plt.tight_layout()

        if not save_individual_frames_instead:
            movie.add_figure(fig)
        else:
            plt.savefig(os.path.join(directory, f'{t}.png'))

    if not save_individual_frames_instead:
        movie.save()


def plot_runner(
        darr_path,
        pde_name,
        vanilla_params,
        der_bc_params,
        norm_added_params,
        save_individual_frames=False,
):
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
        # 'Mixed Models': lambda: MixedPDE(),
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

    if save_individual_frames:
        create_line_plots(
            f'../video/plots/der_bc/{os.path.splitext(os.path.basename(darr_path))[0]}/{os.path.basename(darr_path)} {pde_name}.mp4',
            plots
        )
    else:
        create_line_plots(
            f'../video/{os.path.splitext(os.path.basename(darr_path))[0]} {pde_name}.mp4',
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

    df = pd.read_csv('../params/dirichlet_boundary_adaptive_dt_optimized.csv', sep=',')
    filename_to_path = {filename.split('\\')[-1].strip(): filename for filename in darr_paths}
    df['Filename'] = df['Filename'].map(filename_to_path)

    selected_columns = ['diffusivity', 'mu', 'alpha', 'beta', 'lambda_term', 'gamma']
    pdetype_to_name = {
        'LogisticDiffusionPDE': 'Logistic Diffusion',
        'LinearDiffusivityPDE': 'Linear Diffusion',
        'SigmoidDiffusivityPDE': 'Sigmoid Diffusion',
    }
    pde_types = df['Eqname'].unique()
    filenames = df['Filename'].unique()
    print('These pde types were found in the given file:', pde_types)

    for filename in filenames:
        print('Currently plotting: ', filename)
        der_bc_params = []
        for pde_type in pde_types:
            filtered_df = df[(df['Filename'] == filename) & (df['Eqname'] == pde_type)]

            # der_bc_df = filtered_df[filtered_df['Type'] == 'Derivative BC']
            der_bc_df = filtered_df

            der_bc_param = create_params(der_bc_df.iloc[0][selected_columns]) if not der_bc_df.empty else None
            der_bc_params.append((pdetype_to_name[pde_type], der_bc_param, pdetype_to_name[pde_type]))

        plot_runner(
            filename,
            '',
            vanilla_params=None,
            # vanilla_params=der_bc_params,
            der_bc_params=der_bc_params,
            # der_bc_params=None,
            norm_added_params=None
        )


def create_tall_plot():
    darr_paths = []
    for directory in [r'G:\rost\Ca2+_laser\raw_data', r'G:\rost\kozep\raw_data', r'G:\rost\sarok\raw_data']:
        for root, _, files in os.walk(directory):
            darr_paths.extend([os.path.join(root, file) for file in files if file.endswith('.nd2')])
    darr_paths = map(str, darr_paths)

    def create_params_from_row(row):
        params = Parameters()
        # print(row)
        for col, value in row.items():
            if not pd.isnull(value):
                params.add(col, value)

        return params

    df = pd.read_csv('../params/dirichlet_boundary_adaptive_dt_optimized.csv', sep=',')
    filename_to_path = {filename.split('\\')[-1].strip(): filename for filename in darr_paths}
    df['Filename'] = df['Filename'].map(filename_to_path)

    filenames = df['Filename'].unique()
    selected_columns = ['diffusivity', 'mu', 'alpha', 'beta', 'gamma', 'lambda_term']

    for filename in filenames:
        print(filename)
        darr = DiffusionArray(filename).channel(0).frame('0:50')
        pipeline = PipeLineWidget(None, display_mode=False)
        pipeline.add_step(ClippingWidget())
        pipeline.add_step(StartFrameWidget())
        pipeline.add_step(StartPlaceWidget())
        pipeline.add_step(BackgroundRemovalWidget())
        pipeline.add_step(NormalizingWidget())
        darr, start_frame, start_place = pipeline.apply_pipeline(darr)
        # noinspection PyTypeChecker
        rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)

        filtered_df = df[(df['Filename'] == filename)]
        linear_df = filtered_df[filtered_df['Eqname'] == 'LinearDiffusivityPDE'].iloc[0][selected_columns]
        fisher_df = filtered_df[filtered_df['Eqname'] == 'LogisticDiffusionPDE'].iloc[0][selected_columns]
        sigmoid_df = filtered_df[filtered_df['Eqname'] == 'SigmoidDiffusivityPDE'].iloc[0][selected_columns]

        linear_pde = LinearDiffusivityPDE()
        linear_pde.parameters = create_params_from_row(linear_df)
        # linear_sol = solve_pde(VanillaSymmetricIVBCP(rtp, start_frame, linear_pde))
        linear_sol = solve_pde(DerivativeSymmetricIVBCP(rtp, start_frame, linear_pde))

        fisher_pde = LogisticDiffusionPDE()
        fisher_pde.parameters = create_params_from_row(fisher_df)
        # fisher_sol = solve_pde(VanillaSymmetricIVBCP(rtp, start_frame, fisher_pde))
        fisher_sol = solve_pde(DerivativeSymmetricIVBCP(rtp, start_frame, fisher_pde))

        sigmoid_pde = SigmoidDiffusivityPDE()
        sigmoid_pde.parameters = create_params_from_row(sigmoid_df)
        # sigmoid_sol = solve_pde(VanillaSymmetricIVBCP(rtp, start_frame, sigmoid_pde))
        sigmoid_sol = solve_pde(DerivativeSymmetricIVBCP(rtp, start_frame, sigmoid_pde))

        selected_frames = list(int(x ** 1.25) for x in range(15))
        # selected_frames = list(range(10))
        frame_of_max_intensity = np.argmax(np.max(rtp, axis=1))
        derivative_sols = [
            # rtp.ndarray[[f + frame_of_max_intensity for f in selected_frames]],
            rtp.ndarray[[f + start_frame for f in selected_frames]],
            linear_sol[selected_frames],
            fisher_sol[selected_frames],
            sigmoid_sol[selected_frames]
        ]
        v_min = np.min(derivative_sols[0])
        v_max = np.max(derivative_sols[0])
        titles = ['Mért adatok', 'Lineáris diffúzió', 'Fisher-KPP egyenlet', 'Nem lineáris diffúzió']
        plt.figure(figsize=(7, 12), dpi=300)
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 0.075])
        cmap = matplotlib.colormaps['plasma']

        for i, (spec, derivative_sol) in enumerate(zip(gs, derivative_sols)):
            ax = plt.subplot(spec)
            ax.set_title(titles[i])
            ax.set_ylim([v_min, v_max])
            for j, state in enumerate(derivative_sol):
                color = cmap(j / (len(derivative_sol) - 1))
                ax.plot(state, color=color)

            ax.xaxis.set_visible(i > len(derivative_sols) - 2)
            ax.set_xlabel('Távolság az origótól')
            ax.set_ylabel('Koncentráció')

        ax_colorbar = plt.subplot(gs[-1])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(selected_frames) - 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='horizontal', ticks=range(len(selected_frames)))
        # cbar.ax.set_xticklabels([f'{f + frame_of_max_intensity}' for f in selected_frames])
        cbar.ax.set_xticklabels([f'{f + start_frame}' for f in selected_frames])
        cbar.set_label('Képkocka')

        plt.tight_layout()
        # plt.title(filename)
        plt.show()


if __name__ == '__main__':
    main()
    # plot_single()
