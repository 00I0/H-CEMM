import functools
from abc import ABC, abstractmethod
from builtins import float
from typing import Optional, Callable, Sequence

import numpy as np
from lmfit import Parameters
from numba import njit
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData


class DiffusionPDEBase(PDEBase, ABC):
    """
    Base class for diffusion partial differential equations (PDEs).

    This class defines the common interface for diffusion PDEs. Subclasses should implement the `evolution_rate` method
     to specify the PDE's behavior. The following methods should also be implemented: `_make_pde_rhs_numba` and
     `update_parameters`

    Properties:
        bc (BoundariesData): The boundary conditions for the PDE.

    Attributes:
        diffusivity (float): The diffusivity coefficient of the PDE.

    Methods:
        evolution_rate(state, t): Calculate the evolution rate of the PDE at a given state and time.
        _make_pde_rhs_numba(state, **kwargs): Create a Numba-compiled function for calculating the PDE right-hand side.
        update_parameters(parameters): Update PDE parameters from a set of lmfit.Parameters.

    Properties:
        bc (BoundariesData): Property for getting and setting the boundary conditions.
        diffusivity (float): Property for getting and setting the diffusivity coefficient.
    """

    def __init__(
            self,
            parameters: Parameters,
            # T: int,
            bc: BoundariesData = 'auto_periodic_neumann',
            # noise: float = 0,
            # rng: Optional[np.random.Generator] = None,
            # s: float = 0,
            # norm_mu: Optional[float] = None,
            # norm_sigma: Optional[float] = None
            **kwargs
    ):
        """
        Initialize a DiffusionPDEBase.

        Args:
            diffusivity (float): The diffusivity coefficient of the PDE.

            bc (BoundariesData): The boundary conditions for the PDE.
            noise (float): Variance of the additive Gaussian white noise that is supported for all PDEs by default.
                If set to zero, a deterministic partial differential equation will be solved.
            rng (Optional[np.random.Generator]): Random number generator for noise.
                Note that this random number generator is only used for numpy functions, while compiled numba code is
                unaffected. Using the same generator for solving PDEs concurrently is strongly discouraged.
        """
        super().__init__(noise=0, rng=None)
        self._bc = bc
        self._is_bc_invalid = False

        # self.s = s
        self.norm_mu = 0
        self.norm_sigma = 1
        self.T = 0
        #
        # self.diffusivity = diffusivity

        self._cached_numba_rhs = None

        self._parameters = parameters

        # existing_parameter_names = [param.name for param in self._parameters]
        for key in kwargs:
            if key not in self._parameters:
                raise KeyError(f'Parameter {key} is not applicable to this pde({type(self)})')
            else:
                self._parameters[key].value = kwargs[key]

    @abstractmethod
    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Calculate the evolution rate of the PDE at a given state and time.

        Args:
            state (ScalarField): The current state of the PDE.
            t (float): The current time.

        Returns:
            ScalarField: The evolution rate of the PDE.
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Parameters) -> None:
        """
        Update PDE parameters from a set of lmfit.Parameters.

        Args:
            parameters (Parameters): The lmfit.Parameters object containing parameter values.
        """
        # existing_parameter_names = [param.name for param in self._parameters]
        for parameter in parameters.values():
            if parameter.name not in self._parameters:
                raise KeyError(f'Parameter {parameter.name} is not applicable to this pde({type(self)})')
            else:
                self._parameters[parameter.name] = parameter

    @property
    def parameter_values(self):
        class ParameterValueAccessor:
            def __getitem__(_, keys: str | Sequence[str]) -> list[float]:
                if isinstance(keys, str):
                    keys = [keys]
                return [self.parameters[key].value for key in keys]

        return ParameterValueAccessor()

    @property
    def bc(self) -> BoundariesData:
        return self._bc

    @bc.setter
    def bc(self, bc: BoundariesData) -> None:
        """
        A setter for boundary conditions.

        Please be cautious changing the boundary conditions after rhs has been numba compiled, will have no effect.
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


class LinearDiffusivityPDE(DiffusionPDEBase):
    r"""
    A simple diffusion equation with linear breakdown of the concentration.

    The mathematical definition is:

    .. math::
        \partial_t c = D \cdot \nabla^2 c - \mu c

    where :math:`c` is a scalar field, :math:`D` denotes the diffusivity and :math:`\mu` the breakdown coefficient.

    Attributes:
        diffusivity (float): :math:`D`, the diffusivity coefficient of the PDE.
        mu (float):  :math:`\mu`, the breakdown coefficient of the PDE.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initialize a LinearDiffusivityPDE.

        Args:
            diffusivity (float): The diffusivity coefficient of the PDE.
            mu (float): The breakdown coefficient of the PDE.

            bc (BoundariesData): The boundary conditions for the PDE.
            noise (float): Variance of the additive Gaussian white noise that is supported for all PDEs by default.
                If set to zero, a deterministic partial differential equation will be solved.
            rng (Optional[np.random.Generator]): Random number generator for noise.
                Note that this random number generator is only used for numpy function, while compiled numba code is
                unaffected. Using the same generator for solving PDEs concurrently is strongly discouraged.
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

        # diffusivity = self.parameters['diffusivity'].value
        # mu = self.parameters['mu'].value
        # s = self.parameters['s'].value
        # n = self.parameters['n'].value
        #
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

            if self.parameters['s'].value == 0: return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs


class SigmoidDiffusivityPDE(DiffusionPDEBase):
    r"""
    A diffusion equation where the diffusion term is dependent on the concentration value.

    The mathematical definition is:

    .. math::
        \partial_t c = \frac{1}{1 + e^{{\left(\beta - D  c\right)}}} \nabla^2 c - \mu c^n

    where :math: `c` is a scalar field, :math: `\beta` is the sigmoid translation factor, :math: `D` is the diffusivity
    and :math:`\mu` the breakdown coefficient.

    Attributes:
        diffusivity (float): :math:`D`, the diffusivity coefficient of the PDE.
        mu (float): :math: `\mu`, the breakdown coefficient of the PDE.
        beta (float): :math: `\beta`, the sigmoid translation factor.
        n (int): :math: `n`, the exponent of the sigmoid function.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initialize a SigmoidDiffusivityPDE.

        Args:
            diffusivity (float): The diffusivity coefficient of the PDE.
            mu (float): The breakdown coefficient of the PDE.
            beta (float): The sigmoid translation factor.
            n (int): The exponent of the sigmoid function.

            bc (BoundariesData): The boundary conditions for the PDE.
            noise (float): Variance of the additive Gaussian white noise that is supported for all PDEs by default.
                If set to zero, a deterministic partial differential equation will be solved.
            rng (Optional[np.random.Generator]): Random number generator for noise.
                Note that this random number generator is only used for numpy function, while compiled numba code is
                unaffected. Using the same generator for solving PDEs concurrently is strongly discouraged.
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

        # diffusivity = self.parameters['diffusivity'].value
        # beta = self.parameters['beta'].value
        # mu = self.parameters['mu'].value
        # s = self.parameters['s'].value
        # n = self.parameters['n'].value

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

            if self.parameters['s'].value == 0: return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs


class LogisticDiffusionPDE(DiffusionPDEBase):
    r"""
    A diffusion equation with a source term for the concentration.

     The mathematical definition is:

    .. math::
        \partial_t c = D \cdot \nabla^2 c + \alpha c \cdot (\lambda - c)

    where :math: `c` is a scalar field, :math: `\alpha` is the scaling factor, :math: `D` is the diffusivity
    and :math:`\lambda` is the translation factor.

    Attributes:
        diffusivity (float): :math:`D`, the diffusivity coefficient of the PDE.
        lambda_term (float): :math: `\lambda`, the translation factor of the source term.
        alpha (float): :math: `\alpha`, the scaling factor of the source term.
    """

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        """
        Initialize a LogisticDiffusionPDE.

        Args:
            diffusivity (float): The diffusivity coefficient of the PDE.
            lambda_term (float): The translation factor of the source term.
            alpha (float): The scaling factor of the source term.

            bc (BoundariesData): The boundary conditions for the PDE.
            noise (float): Variance of the additive Gaussian white noise that is supported for all PDEs by default.
                If set to zero, a deterministic partial differential equation will be solved.
            rng (Optional[np.random.Generator]): Random number generator for noise.
                Note that this random number generator is only used for numpy function, while compiled numba code is
                unaffected. Using the same generator for solving PDEs concurrently is strongly discouraged.
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

        diffusivity = self.parameters['diffusivity'].value
        alpha = self.parameters['alpha'].value
        lambda_term = self.parameters['lambda_term'].value
        s = self.parameters['s'].value

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

            if self.parameters['s'].value == 0: return rhs

            norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            return rhs + norm_dist

        return pde_rhs


class MixedPDE(DiffusionPDEBase):

    def __init__(
            self,
            bc: BoundariesData = 'auto_periodic_neumann',
            **kwargs
    ):
        params = Parameters()
        params.add('D', 0.17, min=0, max=1)
        params.add('lambda_term', 0.22, min=0, max=1)
        params.add('alpha', 1, min=0, max=1)
        # params.add('s', 0, min=0, max=10)

        beta_bound = np.log((1 - 0.01) / 0.01)
        params.add('diffusivity', 10, min=-20, max=20)
        params.add('mu', 0.04, min=-10, max=10)
        params.add('beta', -4.59, min=-beta_bound, max=beta_bound)
        params.add('gamma', 0.1, min=0, max=1)
        # params.add('s', 0, min=0, max=10)
        # params.add('n', 1)

        params.add('phi', value=0.5, min=0, max=1)

        super().__init__(parameters=params, bc=bc, **kwargs)

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        # D = self.parameters['D'].value
        # alpha = self.parameters['alpha'].value
        # lambda_term = self.parameters['lambda_term'].value
        D, alpha, lambda_term = self.parameter_values['D', 'alpha', 'lambda_term']

        # diffusivity = self.parameters['diffusivity'].value
        # mu = self.parameters['mu'].value
        # beta = self.parameters['beta'].value
        diffusivity, mu, beta, gamma = self.parameter_values['diffusivity', 'mu', 'beta', 'gamma']

        phi = self.parameters['phi'].value

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})

        diff_term = gamma / (1 + np.exp(beta - diffusivity * state.data))

        return (
                phi * (D * laplace_applied + alpha * state * (lambda_term - state))
                + (1 - phi) * (diff_term * laplace_applied - mu * state.data)
        )

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
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
                    phi * (D * laplace_operator(state_data, args={'t': t}) + alpha * state_data * (
                    lambda_term - state_data))
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
            # if self.parameters['s'].value == 0: return rhs
            #
            # norm_dist = self.parameters['s'].value * self.normal_dist(state_data.shape, t)
            # return rhs + norm_dist

        return pde_rhs
