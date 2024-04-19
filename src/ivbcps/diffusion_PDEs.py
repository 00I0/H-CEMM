import functools
import warnings
from abc import ABC, abstractmethod
from builtins import float
from typing import Callable, Sequence

import numpy as np
from lmfit import Parameters
from numba import njit
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData


class DiffusionPDEBase(PDEBase, ABC):
    """
    Base class for diffusion partial differential equations (PDEs).

    This class provides a common interface and functionality for different types of diffusion PDEs.
    Subclasses are required to implement the methods for calculating the evolution rate, right-hand side of the PDE,
    and the specific delta t calculation.

    Properties:
        bc (BoundariesData): The boundary conditions for the PDE.

    Methods:
        evolution_rate: Calculate the right hand side of the PDE at a given state and time.
        _make_pde_rhs_numba: Create a Numba-compiled function for calculating the PDE right-hand side.
        calculate_delta_t: Calculate the appropriate time step (delta t) for numerical solutions based on stability
            criteria.
    """

    def __init__(
            self,
            parameters: Parameters,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initializes a DiffusionPDEBase instance with specified parameters and boundary conditions.

        Args:
            parameters (Parameters): lmfit.Parameters object containing the parameters for the PDE.
            bc (BoundariesData, optional): Boundary conditions for the PDE. Defaults to 'auto_periodic_neumann'.

        Raises:
            KeyError: If a parameter provided in kwargs is not applicable to the PDE.
        """
        super().__init__(noise=0, rng=None)
        self._bc = bc
        self._is_bc_invalid = False

        self.norm_mu = 0
        self.norm_sigma = 1
        self.T = 0

        self._cached_numba_rhs = None

        self._parameters = parameters

        for key in kwargs:
            if key not in self._parameters:
                raise KeyError(f'Parameter {key} is not applicable to this pde({type(self)})')
            else:
                self._parameters[key].value = kwargs[key]

    @abstractmethod
    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Abstract method to calculate the evolution rate / right hand side of the PDE at a given state and time.

        Args:
            state (ScalarField): The current state of the simulation.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.

        Note:
            Must be implemented by subclasses to define specific behavior.
        """
        raise NotImplementedError()

    @abstractmethod
    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Abstract method to create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A function compiled with Numba that takes the state data and time
            as inputs and returns the evolution rate of the state data.

        Note:
            Must be implemented by subclasses to define the numerical computation for the PDE.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_delta_t(self, delta_x):
        """
        Abstract method to calculate the appropriate time step (delta t) based on spatial resolution (delta x).

        Args:
            delta_x (float): The spatial resolution of the grid.

        Note:
            Must be implemented by subclasses based on the stability criteria of the numerical method used.
        """
        raise NotImplementedError()

    @property
    def parameters(self) -> Parameters:
        """
        Gets the parameters for the PDE.

        Returns:
            Parameters: The current set of parameters for the PDE.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Parameters) -> None:
        """
        Sets the parameters for the PDE.

        Args:
            parameters (Parameters): A Parameters instance from lmfit with updated values for the PDE.

        Raises:
            KeyError: If a parameter in the provided Parameters instance is not applicable to this PDE.
        """
        for parameter in parameters.values():
            if parameter.name not in self._parameters:
                raise KeyError(f'Parameter {parameter.name} is not applicable to this pde({type(self)})')
            else:
                self._parameters[parameter.name] = parameter

    @property
    def parameter_values(self):
        """
        Provides indexed access to the parameter values stored in this PDE.

        This property returns an instance of a nested class that supports indexed access by parameter name or a sequence
        of names, allowing for easy retrieval of one or more parameter values at once.

        Returns:
            ParameterValueAccessor: An accessor object for the parameter values.

        Example:
            >>> pde = LinearDiffusivityPDE()
            >>> diffusivity = pde.parameter_values['diffusivity']
            >>> mu, s = pde.parameter_values['mu', 's']
            >>> print(f"Diffusivity: {diffusivity}, Mu: {mu}, S: {s}")
            Diffusivity: 0.36, Mu: -0.04, S: 0.0

            This example shows how to access single and multiple parameter values.
        """

        class ParameterValueAccessor:
            # noinspection PyMethodParameters
            def __getitem__(_, keys: str | Sequence[str]) -> list[float]:
                """
                Retrieves the value of one or more parameters.

                Args:
                    keys (Union[str, Sequence[str]]): A single parameter name or a list/tuple of parameter names.

                Returns:
                    Union[float, List[float]]: The value of the requested parameter if a single name is provided, or
                        a list of values if multiple names are provided.
                """
                if isinstance(keys, str):
                    keys = [keys]
                return [self.parameters[key].value for key in keys]

        return ParameterValueAccessor()

    @property
    def bc(self) -> BoundariesData:
        """
        Gets the current boundary conditions.

        Returns:
            BoundariesData: The boundary conditions of the PDE.
        """
        return self._bc

    @bc.setter
    def bc(self, bc: BoundariesData) -> None:
        """
        Sets new boundary conditions for the PDE.

        Args:
            bc (BoundariesData): New boundary conditions.

        Notes:
            Changing boundary conditions after the RHS function has been compiled will not affect the existing function.
        """
        self._bc = bc
        self._is_bc_invalid = True

    def normal_dist(self, shape, t) -> np.ndarray:
        sigma = self.norm_sigma
        mu = self.norm_mu

        if sigma is None or mu is None:
            return np.zeros(shape)

        if self.T is not None and t >= self.T:
            return np.zeros(shape)

        return DiffusionPDEBase._normal_dist(shape, sigma, mu)

    @staticmethod
    @functools.cache
    def _normal_dist(shape, sigma, mu):
        grid = np.meshgrid(*[np.arange(0, size) for size in shape], indexing='ij')
        x = np.sqrt(sum(coord ** 2 for coord in grid))

        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def _validate_delta_t(func):
        """
        Decorator to validate the delta t value calculated by any calculate_delta_t method.

        This decorator checks the computed delta t value to ensure it's within a reasonable range.
        If delta t is too small or negative, warnings are issued and adjusted accordingly.

        Args:
            func (Callable): A function that calculates delta t.

        Returns:
            Callable: A wrapped function with delta t validation.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delta_t = func(*args, **kwargs)
            if delta_t <= 1e-7:
                warnings.warn(f'Small delta t encountered: {delta_t}, this may result in long computation times.')
            if delta_t <= 0:
                warnings.warn(f'Negative delta t encountered: {delta_t}, using default delta t ({1e-4}) instead.')
                delta_t = 1e-4
            return delta_t

        return wrapper


class LinearDiffusivityPDE(DiffusionPDEBase):
    r"""
    Implements a linear diffusion equation with breakdown of the concentration over time.

    The mathematical definition is:

    The linear diffusion equation modeled by this class is given by the PDE:
    .. math::
        \partial_t c = D \nabla^2 c - \mu c^n

    where :math:`c` is a scalar field, :math:`D` denotes the diffusivity and :math:`\mu` the breakdown coefficient.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initializes the LinearDiffusivityPDE with specific parameters and boundary conditions.

        Args:
            bc (BoundariesData, optional): Boundary conditions for the PDE. Defaults to 'auto_periodic_neumann'.
            **kwargs: Additional keyword arguments for overriding default parameter values, for 'diffusivity'
                    'mu' and 'n'.
        """

        params = Parameters()
        params.add('diffusivity', 0.36, min=0, max=20)
        params.add('mu', -0.04, min=-10, max=10)
        params.add('s', 0, min=0, max=10)
        params.add('n', 1)

        super().__init__(parameters=params, bc=bc, **kwargs)

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Calculate the evolution rate of the LinearDiffusivityPDE.

        Args:
            state (ScalarField): The current state of the PDE.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        diffusivity, mu, s, n = self.parameter_values['diffusivity', 'mu', 's', 'n']

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})

        return diffusivity * laplace_applied - mu * (state.data ** n) + s * self.normal_dist(state.data.shape, t)

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A Numba-compiled function for calculating the right-hand side.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        @njit(cache=True)
        def linear_diffusion(
                state_data: np.ndarray,
                t: float,
                diffusivity: float,
                mu: float,
                n: float
        ) -> np.ndarray:
            return diffusivity * laplace_operator(state_data, args={'t': t}) - mu * (state_data ** n)

        if self._cached_numba_rhs is None or self._is_bc_invalid:
            self._cached_numba_rhs = linear_diffusion
            self._is_bc_invalid = False

        def pde_rhs(state_data: np.ndarray, t: float):
            rhs = self._cached_numba_rhs(
                state_data,
                t,
                self.parameters['diffusivity'].value,
                self.parameters['mu'].value,
                self.parameters['n'].value
            )

            if self.parameters['s'].value == 0:
                return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs

    @DiffusionPDEBase._validate_delta_t
    def calculate_delta_t(self, delta_x: float) -> float:
        """
        Calculates an optimal time step based on spatial discretization to ensure stability of the numerical solution.

        Args:
            delta_x (float): The spatial step size.

        Returns:
            float: The recommended time step size.
        """
        diffusivity, mu = self.parameter_values['diffusivity', 'mu']
        delta_t = 0.95 * 2 / ((4 * diffusivity / (delta_x ** 2)) + mu)
        return delta_t


class SigmoidDiffusivityPDE(DiffusionPDEBase):
    r"""
    A diffusion equation where the diffusion term is dependent on the concentration value.

    The mathematical definition is:

    .. math::
        \partial_t c = \frac{\gamma}{1 + e^{{\left(\beta - D  c\right)}}} \nabla^2 c - \mu c^n

    where :math: `c` is a scalar field.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initializes the SigmoidDiffusivityPDE with specific parameters and boundary conditions.

        Args:
            bc (BoundariesData, optional): Boundary conditions for the PDE. Defaults to 'auto_periodic_neumann'.
            **kwargs: Additional keyword arguments for overriding default parameter values, for 'diffusivity'
                    'mu', 'beta', 'n' and 'gamma'.
        """

        beta_bound = np.log((1 - 0.01) / 0.01)
        params = Parameters()
        params.add('diffusivity', 10, min=-20, max=20)
        params.add('mu', 0.04, min=-10, max=10)
        params.add('beta', -4.59, min=-beta_bound, max=beta_bound)
        params.add('s', 0, min=0, max=10)
        params.add('n', 1)
        params.add('gamma', 1, min=-50, max=50)

        super().__init__(parameters=params, bc=bc, **kwargs)

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Calculate the evolution rate of the SigmoidDiffusivityPDE.

        Args:
            state (ScalarField): The current state of the PDE.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        diffusivity, beta, mu, s, n, gamma = self.parameter_values['diffusivity', 'beta', 'mu', 's', 'n', 'gamma']
        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})

        diff_term = gamma / (1 + np.exp(beta - diffusivity * state.data))
        return diff_term * laplace_applied - mu * (state.data ** n) + s * self.normal_dist(state.data.shape, t)

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A Numba-compiled function for calculating the right-hand side.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        @njit(cache=True)
        def sigmoid_diffusion(
                state_data: np.ndarray,
                t: float,
                beta: float,
                diffusivity: float,
                mu: float,
                n: float,
                gamma: float
        ) -> np.ndarray:
            diff_term = gamma / (1 + np.exp(beta - diffusivity * state_data))
            return diff_term * laplace_operator(state_data, args={'t': t}) - mu * (state_data ** n)

        if self._cached_numba_rhs is None or self._is_bc_invalid:
            self._cached_numba_rhs = sigmoid_diffusion
            self._is_bc_invalid = False

        def pde_rhs(state_data: np.ndarray, t: float):
            rhs = self._cached_numba_rhs(
                state_data,
                t,
                self.parameters['beta'].value,
                self.parameters['diffusivity'].value,
                self.parameters['mu'].value,
                self.parameters['n'].value,
                self.parameters['gamma'].value
            )

            if self.parameters['s'].value == 0:
                return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs

    @DiffusionPDEBase._validate_delta_t
    def calculate_delta_t(self, delta_x: float) -> float:
        """
        Calculates an optimal time step based on spatial discretization to ensure stability of the numerical solution.

        Args:
            delta_x (float): The spatial step size.

        Returns:
            float: The recommended time step size.
        """
        gamma, mu = self.parameter_values['gamma', 'mu']
        delta_t = 0.95 * 2 / ((4 * gamma / (delta_x ** 2)) + mu)
        return delta_t


class LogisticDiffusionPDE(DiffusionPDEBase):
    r"""
    Implements the Fisher-KPP equation.

     The mathematical definition is:

    .. math::
        \partial_t c = D \cdot \nabla^2 c + \alpha c \cdot (\lambda - c)

    where :math: `c` is a scalar field.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initializes the LogisticDiffusionPDE with specific parameters and boundary conditions.

        Args:
            bc (BoundariesData, optional): Boundary conditions for the PDE. Defaults to 'auto_periodic_neumann'.
            **kwargs: Additional keyword arguments for overriding default parameter values, for 'diffusivity'
                    'lambda_term', and 'alpha'.
        """

        params = Parameters()
        params.add('diffusivity', 0.17, min=0, max=1)
        params.add('lambda_term', 0.22, min=0, max=1)
        params.add('alpha', 1, min=0, max=1)
        params.add('s', 0, min=0, max=10)

        super().__init__(parameters=params, bc=bc, **kwargs)

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Calculate the evolution rate of the LogisticDiffusionPDE.

        Args:
            state (ScalarField): The current state of the PDE.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        diffusivity, alpha, lambda_term, s = self.parameter_values['diffusivity', 'alpha', 'lambda_term', 's']
        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})

        return (
                diffusivity * laplace_applied + alpha * state * (lambda_term - state)
                + s * self.normal_dist(state.data.shape, t)
        )

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A Numba-compiled function for calculating the right-hand side.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        @njit(cache=True)
        def logistic_diffusion(
                state_data: np.ndarray,
                t: float,
                diffusivity: float,
                lambda_term: float,
                alpha: float,
        ) -> np.ndarray:
            return (diffusivity * laplace_operator(state_data, args={'t': t})
                    + alpha * state_data * (lambda_term - state_data))

        if self._cached_numba_rhs is None or self._is_bc_invalid:
            self._cached_numba_rhs = logistic_diffusion
            self._is_bc_invalid = False

        def pde_rhs(state_data: np.ndarray, t: float):
            rhs = self._cached_numba_rhs(
                state_data,
                t,
                self.parameters['diffusivity'].value,
                self.parameters['lambda_term'].value,
                self.parameters['alpha'].value
            )

            if self.parameters['s'].value == 0:
                return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs

    @DiffusionPDEBase._validate_delta_t
    def calculate_delta_t(self, delta_x: float) -> float:
        """
        Calculates an optimal time step based on spatial discretization to ensure stability of the numerical solution.

        Args:
            delta_x (float): The spatial step size.

        Returns:
            float: The recommended time step size.
        """
        diffusivity, lambda_term, alpha = self.parameter_values['diffusivity', 'lambda_term', 'alpha']
        delta_t = 0.95 * (-3 + np.sqrt(9 + 4 * alpha * (lambda_term + 1) * delta_x ** 2 * (1 / diffusivity))) / (
                2 * alpha * (lambda_term + 1))

        return delta_t


class MixedPDE(DiffusionPDEBase):
    r"""
    Implements a mixture of the Logistic and Sigmoid equations.

    The mathematical definition is:

    .. math::
        \partial_t c = \phi \left(  D \cdot \nabla^2 c + \alpha c \cdot (\lambda - c) \right)
            \cdot (1-\phi)\left( \frac{\gamma}{1 + e^{{\left(\beta - Diffusivity  c\right)}}} \nabla^2 c - \mu c \right)

    where :math: `c` is a scalar field.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initializes the MixedPDE with specific parameters and boundary conditions.

        Args:
            bc (BoundariesData, optional): Boundary conditions for the PDE. Defaults to 'auto_periodic_neumann'.
            **kwargs: Additional keyword arguments for overriding default parameter values, for 'diffusivity'
                    'mu', 'beta', 'gamma', 'D', 'lambda_term', 'alpha' and 'phi'.
        """

        params = Parameters()
        params.add('D', 0.17, min=0, max=1)
        params.add('lambda_term', 0.22, min=0, max=1)
        params.add('alpha', 1, min=0, max=1)

        beta_bound = np.log((1 - 0.1) / 0.1)
        params.add('diffusivity', 10, min=-20, max=20)
        params.add('mu', 0.04, min=-10, max=10)
        params.add('beta', -4.59, min=-beta_bound, max=beta_bound)
        params.add('gamma', 0.1, min=0, max=1)

        params.add('phi', value=0.5, min=0, max=1)

        super().__init__(parameters=params, bc=bc, **kwargs)

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Calculate the evolution rate of the LogisticDiffusionPDE.

        Args:
            state (ScalarField): The current state of the PDE.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        # noinspection PyPep8Naming
        D, alpha, lambda_term = self.parameter_values['D', 'alpha', 'lambda_term']
        diffusivity, mu, beta, gamma = self.parameter_values['diffusivity', 'mu', 'beta', 'gamma']

        phi = self.parameters['phi'].value

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})

        diff_term = gamma / (1 + np.exp(beta - diffusivity * state.data))

        return (
                phi * (D * laplace_applied + alpha * state * (lambda_term - state))
                + (1 - phi) * (diff_term * laplace_applied - mu * state.data)
        )

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A Numba-compiled function for calculating the right-hand side.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        # noinspection PyPep8Naming
        @njit(cache=True)
        def combined_diffusion(
                state_data: np.ndarray,
                t: float,
                D: float,
                lambda_term: float,
                alpha: float,
                diffusivity: float,
                mu: float,
                beta: float,
                phi: float,
                gamma: float
        ) -> np.ndarray:
            diff_term = gamma / (1 + np.exp(beta - diffusivity * state_data))
            return (
                    phi * (D * laplace_operator(state_data, args={'t': t})
                           + alpha * state_data * (lambda_term - state_data))
                    + (1 - phi) * (diff_term * laplace_operator(state_data, args={'t': t}) - mu * state_data)
            )

        if self._cached_numba_rhs is None or self._is_bc_invalid:
            self._cached_numba_rhs = combined_diffusion
            self._is_bc_invalid = False

        def pde_rhs(state_data: np.ndarray, t: float):
            rhs = self._cached_numba_rhs(
                state_data,
                t,
                self.parameters['D'].value,
                self.parameters['lambda_term'].value,
                self.parameters['alpha'].value,
                self.parameters['diffusivity'].value,
                self.parameters['mu'].value,
                self.parameters['beta'].value,
                self.parameters['phi'].value,
                self.parameters['gamma'].value
            )

            return rhs

        return pde_rhs

    @DiffusionPDEBase._validate_delta_t
    def calculate_delta_t(self, delta_x: float) -> float:
        """
        Calculates an optimal time step based on spatial discretization to ensure stability of the numerical solution.

        Args:
            delta_x (float): The spatial step size.

        Returns:
            float: The recommended time step size.
        """
        # noinspection PyPep8Naming
        D, lambda_term, alpha = self.parameter_values['D', 'lambda_term', 'alpha']
        delta_t_logistic = 0.95 * ((-3 + np.sqrt(9 + 2 * alpha * (lambda_term + 1) * delta_x ** 2 * (1 / D)))
                                   / (2 * alpha * (lambda_term + 1)))

        gamma, mu = self.parameter_values['gamma', 'mu']
        delta_t_sigmoid = 0.95 * 1.5 / ((4 * gamma / delta_x ** 2) + mu)

        delta_t = min(delta_t_sigmoid, delta_t_logistic)
        return delta_t
