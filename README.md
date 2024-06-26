# Project Overview

**H-CEMM: Modeling ATP Diffusion Using Partial Differential Equations**

This project is focused on the simulation and analysis of ATP molecule diffusion in biological environments using
mathematical models based on partial differential equations (PDEs).

The project's approach provides an integrative framework for fitting complex diffusion models to empirical data,
allowing for detailed exploration and optimization of model parameters to closely match observed patterns.

# Installation and Setup

To get started with the H-CEMM project, follow these steps to set up your environment:

0. **Have Python:**
   Please have Python 3.12 installed. Having and IDE like PyCharm would make the next steps easier for you, although it
   is not strictly necessary.


1. **Clone the Repository:**
   Ensure you have git installed and clone the project repository to your local machine using:

   ```git clone https://github.com/00I0/H-CEMM.git```


2. **Install Python Dependencies:**
   The project requires Python 3.12 and several dependencies which are listed in the `requirements.txt` file. Install
   these using pip:

   ```pip install -r requirements.txt```


3. **Download the .nd2 files:**
   Download the nd2 files for which you want to run the program then ensure that those parts of the files in scripts
   directory that are related to io are referencing actual files.

# Folder Structure and File Descriptions

The project is organized into several directories:

- **params/**
    - Contains optimized parameter values for the PDE models.
    - `adaptive_dt.csv` - Contains optimized parameters using Dirichlet boundary conditions
      and adaptive delta t.

- **scripts/**
    - Houses scripts for running simulations and generating plots.
    - `diffusion_model_plotter.py` - Plots results from the diffusion model simulations.
    - `misc_plotter.py` - Provides miscellaneous plotting functionalities.
    - `optimization_runner.py` - Script to run the optimization process for fitting PDEs to measured data.
    - `fisher_phase_plot.nb` - A Mathematica file for creating the phase portrait of the Fisher-KPP equation.

- **src/**
    - Source code for the project.
    - `core/` - Includes the main logic for diffusion analysis.
        - `analyzer.py` - Finds the start frame and the start places of the process.
        - `diffusion_array.py` - Manages diffusion data arrays.
        - `homogenizer.py` - Implementation of the homogenization algorithm.
        - `mask.py` - Could be used to select parts of the data.
        - `radial_time_profile.py` - Implementation of the Radial Time Profile matrix.
        - `step_widget.py` - Provides interactive widgets for step-by-step preprocessing control.
        - `step_widget.css` - CSS for styling step widgets.
    - `file_handling/` - Manages file input/output operations.
        - `downloader.py` - Handles downloading of external data.
        - `file_meta.py` - Manages metadata associated with files.
        - `reader.py` - Reads data from files.
        - `writer.py` - Writes data to files.
    - `ivbcps/` - Contains implementation of initial and boundary value problems for PDEs.
        - `diffusion_PDEs.py` - Contains the PDE definitions for diffusion processes.
        - `ivbcp.py` - Handles initial and boundary value condition setup.
        - `ivp_solver.py` - Solves initial value problems.
        - `optimizer.py` - Optimizes PDE parameters to fit experimental data.
    - `interactive_plots.ipynb` - Jupyter notebook for interactive visualization of the data and some processing steps.
      Please keep in mind that since this notebook uses costume output widgets it might not work with specific cloud
      services such as Google Colab
    - `cell_segmentation.ipynb` - This notebook focuses on segmenting cell images and analyzing their properties. It
      includes methods for preprocessing images, applying segmentation algorithms, and extracting quantitative
      information about the cells. It also explores the implications of cell segmentation to the homogenizing algorithm.

# Script Functionality

- **diffusion_model_plotter.py:**

  This script generates plots from the output of an IVBCP fitting. It uses parameters defined
  in csv file with similar structure to `adaptive_dt.csv`.


- **misc_plotter.py:**

  Provides functionality for creating a variety of additional plots that are not directly related to the core diffusion
  simulations but help in data analysis and presentation.


- **optimization_runner.py:**

  Runs the parameter optimization algorithm to fit the IVBCP-s to the actual measured data. The script outputs the
  optimized parameters

