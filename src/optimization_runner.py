import functools
import itertools
import os
import traceback
from multiprocessing import Pool

import numpy as np
from lmfit import Parameters, create_params

from diffusion_PDEs import SigmoidDiffusivityPDE, MixedPDE
from diffusion_array import DiffusionArray
from homogenizer import Homogenizer
from ivbcp import VanillaSymmetricIVBCP, DerivativeSymmetricIVBCP
from ivp_solver import SymmetricIVPSolver
from optimizer import DynamicMeshResolutionOptimizer
from step_widget import PipeLineWidget, ClippingWidget, StartFrameWidget, StartPlaceWidget, BackgroundRemovalWidget, \
    NormalizingWidget


def async_optimization_starter():
    """
    Start the asynchronous optimization process for multiple IVPBCs.

    This function sets up the pipeline for processing diffusion array data, creates a list of radial time profiles
    and starting frames, defines the PDEs and initial parameter values, and then starts the asynchronous optimization
    process using a multiprocessing pool.

    The optimization is performed for both `VanillaSymmetricIVBCP` and `DerivativeSymmetricIVBCP` instances,
    with different PDEs and initial parameter values.
    """
    darr_paths = []
    for directory in [
        r'G:\rost\Ca2+_laser\raw_data', r'G:\rost\kozep\raw_data', r'G:\rost\sarok\raw_data'
    ]:
        for root, _, files in os.walk(directory):
            darr_paths.extend([os.path.join(root, file) for file in files if file.endswith('.nd2')])

    pipeline = PipeLineWidget(None, display_mode=False)
    pipeline.add_step(ClippingWidget())
    pipeline.add_step(StartFrameWidget())
    pipeline.add_step(StartPlaceWidget())
    pipeline.add_step(BackgroundRemovalWidget())
    pipeline.add_step(NormalizingWidget())

    rtp_start_frame_tuples = []
    for darr_path in darr_paths:
        darr = DiffusionArray(darr_path).channel(0)
        darr, start_frame, start_place = pipeline.apply_pipeline(darr)
        rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)
        rtp_start_frame_tuples.append((rtp, start_frame))

    beta_bound = np.log((1 - 0.01) / 0.01)
    pdes = [
        # (
        #     LogisticDiffusionPDE(),
        #     create_params(
        #         diffusivity={'value': 300, 'min': 0, 'max': 2000},
        #         lambda_term={'value': 0.5, 'min': 0, 'max': 1},
        #         alpha={'value': 1, 'min': 0.01, 'max': 10},
        #         # s={'value': 1, 'min': 0, 'max': 40}
        #     ),
        # ),
        # (
        #     LinearDiffusivityPDE(),
        #     create_params(
        #         diffusivity={'value': 300, 'min': 0, 'max': 2000},
        #         mu={'value': 0.01, 'min': -1, 'max': 1},
        #         # s={'value': 1, 'min': 0, 'max': 40}
        #     ),
        # ),
        # # TODO let - values for Sigmoid and mixed
        (
            SigmoidDiffusivityPDE(),
            create_params(
                diffusivity={'value': 1, 'min': -50, 'max': 50},
                mu={'value': 0.05, 'min': -1, 'max': 1},
                beta={'value': 0, 'min': -beta_bound, 'max': beta_bound},
                # n={'value': 1, 'min': 0.25, 'max': 20},
                gamma={'value': 20, 'min': 0, 'max': 2000},
                # s={'value': 1, 'min': 0, 'max': 40}
            ),
        ),

        (
            MixedPDE(),
            create_params(
                D={'value': 5, 'min': 0, 'max': 2000},
                lambda_term={'value': 0.2284, 'min': 0, 'max': 1},
                alpha={'value': 0.5, 'min': 0.001, 'max': 2},
                diffusivity={'value': 5, 'min': -50, 'max': 50},
                gamma={'value': 20, 'min': 0, 'max': 2000},
                mu={'value': 0.5, 'min': -1, 'max': 1},
                beta={'value': 0, 'min': -beta_bound, 'max': beta_bound},
                phi={'value': 0.5, 'min': 0.1, 'max': 0.9}
            ),
        ),
    ]

    ivp_type = DerivativeSymmetricIVBCP
    ivp_type = VanillaSymmetricIVBCP
    ivps = [
        (
            ivp_type,
            item[0][0],
            item[0][1],
            item[1][0],
            item[1][1]  # params
        ) for item in itertools.product(rtp_start_frame_tuples, pdes)
    ]
    ivp_type = DerivativeSymmetricIVBCP

    with Pool(processes=8) as pool:
        values = pool.starmap(
            functools.partial(
                async_optimize_ivp,
                max_number_of_support_points=-1,
                min_number_of_support_points=40,
                max_number_of_iterations=100,
                dt=1e-4
            ),
            ivps
        )

        print('------')
        for value in values:
            print(value[0])

        print('\n -- -- -- -- ||ヽ(*￣▽￣*)ノミ|Ю -- -- -- -- \n')
        stats = functools.reduce(lambda l1, l2: l1 + l2, [value[1] for value in values])
        keys = list(sorted(functools.reduce(lambda a, b: a | b, map(lambda d: d.keys(), stats))))

        for d in stats:
            msg = ', '.join(f'{key}={d.get(key, "")}' for key in keys)
            print(msg)


def async_optimize_ivp(
        # ivp: SymmetricIVPBase,
        ivp_type,
        radial_time_profile,
        start_frame,
        pde,
        params: Parameters,
        max_number_of_support_points: int = -1,
        min_number_of_support_points: int = 40,
        max_number_of_iterations: int = 50,
        dt: float = 1e-4
):
    ivp = ivp_type(radial_time_profile, start_frame, pde)
    optim = DynamicMeshResolutionOptimizer(ivp)
    try:
        optim.optimize(
            parameters=params,
            max_iterations_per_resolution=max_number_of_iterations,
            max_resolution=max_number_of_support_points,
            min_resolution=min_number_of_support_points,
            report_progress=True,
            dt=dt
        )
        return (optim.message, optim.stats)
    except Exception as e:
        traceback.print_exc()
        try:
            return (optim.message, optim.stats)
        except Exception as e:
            return (f'optim was not successful {type(ivp_type)} {radial_time_profile.name} {pde}', [])


def main():
    pde = SigmoidDiffusivityPDE(n=2)

    pipeline = PipeLineWidget(None, display_mode=False)
    pipeline.add_step(ClippingWidget())
    pipeline.add_step(StartFrameWidget())
    pipeline.add_step(StartPlaceWidget())
    pipeline.add_step(BackgroundRemovalWidget())
    pipeline.add_step(NormalizingWidget())

    darr = DiffusionArray('G:\\rost\\kozep\\raw_data\\super_1472_5_laser_EC1flow_laserabl018.nd2').channel(0)
    darr, start_frame, start_place = pipeline.apply_pipeline(darr)
    rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)
    ivp = VanillaSymmetricIVBCP(rtp, start_frame, pde)
    ivp_solver = SymmetricIVPSolver(ivp)
    sol = ivp_solver.solve(ivp.sec_per_frame * ivp.frames)


if __name__ == '__main__':
    # main()
    async_optimization_starter()
