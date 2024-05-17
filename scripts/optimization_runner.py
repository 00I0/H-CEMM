import functools
import itertools
import json
import os
import traceback
from multiprocessing import Pool
from typing import List

import numpy as np
from lmfit import Parameters, create_params

from core.diffusion_array import DiffusionArray
from core.homogenizer import Homogenizer
from core.step_widget import PipeLineWidget, ClippingWidget, StartFrameWidget, StartPlaceWidget, \
    BackgroundRemovalWidget, \
    NormalizingWidget
from ivbcps.diffusion_PDEs import LinearDiffusivityPDE
from ivbcps.ivbcp import VanillaSymmetricIVBCP
from ivbcps.optimizer import DynamicMeshResolutionOptimizer


def async_optimization_starter(darr_paths: List[str]):
    """
    Start the asynchronous optimization process for multiple IVPBCs.

    This function sets up the pipeline for processing diffusion array data, creates a list of radial time profiles
    and starting frames, defines the PDEs and initial parameter values, and then starts the asynchronous optimization
    process using a multiprocessing pool.

    The optimization is performed for both `VanillaSymmetricIVBCP` and `DerivativeSymmetricIVBCP` instances,
    with different PDEs and initial parameter values.

    Args:
        darr_paths (List[str]): List of filepaths to the nd2 files.
    """

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
        frame_of_max_intensity = np.argmax(np.max(rtp, axis=1))
        darr.frame(f'0:{frame_of_max_intensity + 10}')
        rtp = Homogenizer.Builder().start_frame(start_frame).center_point(start_place).build().homogenize(darr)
        rtp_start_frame_tuples.append((rtp, start_frame))

    beta_bound = np.log((1 - 0.1) / 0.1)

    # By commenting out parts of the pde parameter pairs you can change which pdes to optimize, by commenting out a
    # specific parameter (a line inside create_params) you can specify the parameters to be optimized fo that pde.
    #
    # If you are to change the parameter values (the value associated with the 'value' key in the dicts inside
    # create_params), be extremely cautious as for some parameters there are no stable solutions for some pdes.
    pdes = [
        # (
        #     LogisticDiffusionPDE(),
        #     create_params(
        #         diffusivity={'value': 300, 'min': 0, 'max': 2000},
        #         lambda_term={'value': 0.5, 'min': 0, 'max': 1},
        #         alpha={'value': 1, 'min': 0.01, 'max': 10},
        #     ),
        # ),
        (
            LinearDiffusivityPDE(),
            create_params(
                diffusivity={'value': 300, 'min': 0, 'max': 2000},
                mu={'value': 0.01, 'min': -1, 'max': 1},
            ),
        ),
        # (
        #     SigmoidDiffusivityPDE(),
        #     create_params(
        #         diffusivity={'value': 1, 'min': -10, 'max': 10},
        #         mu={'value': 0.05, 'min': -1, 'max': 1},
        #         beta={'value': 0, 'min': -beta_bound, 'max': beta_bound},
        #         gamma={'value': 20, 'min': 0, 'max': 2000},
        #     ),
        # ),
        # (
        #     MixedPDE(),
        #     create_params(
        #         D={'value': 5, 'min': 0, 'max': 2000},
        #         lambda_term={'value': 0.2284, 'min': 0, 'max': 1},
        #         alpha={'value': 0.5, 'min': 0.001, 'max': 2},
        #         diffusivity={'value': 5, 'min': -10, 'max': 10},
        #         gamma={'value': 20, 'min': 0, 'max': 2000},
        #         mu={'value': 0.5, 'min': -1, 'max': 1},
        #         beta={'value': 0, 'min': -beta_bound, 'max': beta_bound},
        #         phi={'value': 0.5, 'min': 0.1, 'max': 0.9}
        #     ),
        # ),
    ]

    # By changing the ivp_type to `VanillaSymmetricIVBCP` you can specify that you want to only optimize for the
    # atp absorption / breakdown phase of the process, leaving it as `DerivativeSymmetricIVBCP` means you want to use
    # time dependent Neumann boundary conditions
    ivp_type = VanillaSymmetricIVBCP
    ivps = [
        (
            ivp_type,  # ivp type
            item[0][0],  # radial time profile
            item[0][1],  # start_frame
            item[1][0],  # pde
            item[1][1]  # params
        ) for item in itertools.product(rtp_start_frame_tuples, pdes)
    ]

    with Pool(processes=os.cpu_count()) as pool:
        values = pool.starmap(
            functools.partial(
                async_optimize_ivp,
                max_number_of_support_points=-1,
                min_number_of_support_points=40,
                max_number_of_iterations=100,  # max iterations for a resolution
                dt=None  # if this is none, dt will be calculated based on the parameter at that iteration
                # (i.e. adaptive dt will be used)
            ),
            ivps
        )

        print('\n -- -- -- -- Optimization results -- -- -- -- \n')
        stats = functools.reduce(lambda l1, l2: l1 + l2, [value[1] for value in values])
        keys = list(sorted(functools.reduce(lambda a, b: a | b, map(lambda d: d.keys(), stats))))

        for d in stats:
            msg = ', '.join(f'{key}={d.get(key, "")}' for key in keys)
            print(msg)


def async_optimize_ivp(
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
    """
    Optimize the initial value problem (IVP) using dynamic mesh resolution.

    Args:
        ivp_type: The type of IVP to solve.
        radial_time_profile: The radial time profile.
        start_frame: The start frame for the IVP.
        pde: The partial differential equation (PDE) to solve.
        params (Parameters): The parameters for the PDE.
        max_number_of_support_points (int): Maximum number of support points. Defaults to -1.
        min_number_of_support_points (int): Minimum number of support points. Defaults to 40.
        max_number_of_iterations (int): Maximum number of iterations per resolution. Defaults to 50.
        dt (float): Time step for the optimization. Defaults to 1e-4.

    Returns:
        tuple: The optimization message and statistics.
    """
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
        return optim.message, optim.stats
    except Exception as e:
        traceback.print_exc()
        try:
            return optim.message, optim.stats
        except Exception as e:
            return f'optim was not successful {type(ivp_type)} {radial_time_profile.name} {pde}', []


def main(selected_aliases: List[str] | None):
    """
    Main function to start the asynchronous optimization process.

    This function reads the configuration file, filters the diffusion array file paths based on selected aliases,
    and starts the optimization process.

    Args:
        selected_aliases (List[str] | None): List of selected aliases to process. If None, all aliases are processed.
    """
    with open('../config.json', 'r') as json_file:
        config_file = json.load(json_file)

    darr_paths = []
    for details in config_file.values():
        path = details['path']
        alias = details['alias']
        if alias in selected_aliases:
            darr_paths.append(path)

    async_optimization_starter(darr_paths)


if __name__ == '__main__':
    #           -----     ----    ---   ---   --  --  Warning  --  --  ---   ---    ----     -----
    #   the optimization will use all available CPU cores and could potentially run for more than 10 hours,
    #   depending on the number of selected aliases, please note that you can customize the optimization further in
    #   the `async_optimization_starter` method.
    #
    #   Right now this script is configured to only find the optimal parameters for the linear diffusion model, using
    #   Dirichlet boundary conditions; and there are only two initial conditions are selected: `ATP-kozep-017` and
    #   `ATP-kozep-018`. The pde solver is set up, so that it utilizes the radial symmetry, and adaptive delta t is used
    #   based on the CFL conditions.
    #   This configuration results in a relatively fast (on my PC it was 5 mins) running time.

    selected_aliases = ['ATP-kozep-017', 'ATP-kozep-018']
    main(selected_aliases)
