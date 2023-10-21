from abc import ABC, abstractmethod
from builtins import float

import numba as nb
import numpy as np
from lmfit import Parameters
from numba import njit
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData
from typing import Optional, Callable


class DiffusionPDEBase(PDEBase, ABC):
    """
    Base class for diffusion partial differential equations (PDEs).

    This class defines the common interface for diffusion PDEs. Subclasses should implement the `evolution_rate` method
     to specify the PDE's behavior. The following methods should also be implemented: `_make_pde_rhs_numba` and
     `update_parameters`

    Attributes:
        bc (BoundariesData): The boundary conditions for the PDE.
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
            diffusivity: float = 0.5,
            bc: BoundariesData = 'auto_periodic_neumann',
            noise: float = 0,
            rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize a DiffusionPDEBase.

        Args:
            diffusivity (float): The diffusivity coefficient of the PDE.

            bc (BoundariesData): The boundary conditions for the PDE.
            noise (float): Variance of the additive Gaussian white noise that is supported for all PDEs by default.
                If set to zero, a deterministic partial differential equation will be solved.
            rng (Optional[np.random.Generator]): Random number generator for noise.
                Note that this random number generator is only used for numpy function, while compiled numba code is
                unaffected. Using the same generator for solving PDEs concurrently is strongly discouraged.
        """
        super().__init__(noise=noise, rng=rng)
        self.bc = bc
        self.diffusivity = diffusivity

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

    @abstractmethod
    def update_parameters(self, parameters: Parameters) -> None:
        """
        Update PDE parameters from a set of lmfit.Parameters.

        Args:
            parameters (Parameters): The lmfit.Parameters object containing parameter values.
        """
        raise NotImplementedError()

    @property
    def bc(self) -> BoundariesData:
        return self._bc

    @bc.setter
    def bc(self, value: BoundariesData):
        self._bc = value

    @property
    def diffusivity(self) -> float:
        return self._diffusivity

    @diffusivity.setter
    def diffusivity(self, value: float):
        self._diffusivity = value


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
            diffusivity: float = 0.2636,
            mu: float = 0.0200,
            bc: BoundariesData = 'auto_periodic_neumann',
            noise: float = 0,
            rng: Optional[np.random.Generator] = None,
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
        super().__init__(diffusivity=diffusivity, bc=bc, noise=noise, rng=rng)
        self.mu = mu

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value

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

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})
        return self.diffusivity * laplace_applied - self.mu * state  # type: ignore

    def _make_pde_rhs_numba(self, state: ScalarField, **kwargs) -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Create a Numba-compiled function for calculating the PDE right-hand side.
        Should only be used internally.

        Args:
            state (ScalarField): The current state of the PDE.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[np.ndarray, float], np.ndarray]: A Numba-compiled function for calculating the PDE right-hand side.
        """
        if not isinstance(state, ScalarField):
            raise ValueError('state must be a ScalarField.')

        array_type = nb.typeof(state.data)
        signature = array_type(array_type, nb.double)

        diffusivity = self.diffusivity
        mu = self.mu
        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        @njit(signature)
        def pde_rhs(state_data: np.ndarray, t: float) -> np.ndarray:
            return diffusivity * laplace_operator(state_data, args={'t': t}) - mu * state_data

        return pde_rhs

    def update_parameters(self, parameters: Parameters) -> None:
        """
        Update LinearDiffusivityPDE parameters (`diffusivity`, `mu`) from a set of lmfit.Parameters.

        Args:
            parameters (Parameters): The lmfit.Parameters object containing parameter values.
        """
        parameter_names = ('diffusivity', 'mu')
        for parameter in parameters.values():
            if parameter.name in parameter_names:
                setattr(self, parameter.name, parameter.value)


class SigmoidDiffusivityPDE(DiffusionPDEBase):
    r"""
    A diffusion equation where the diffusion term is dependent on the concentration value.

    The mathematical definition is:

    .. math::
        \partial_t c = \frac{1}{1 + e^{{\left(\beta - D  c\right)}^n}} \nabla^2 c - \mu c

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
            diffusivity: float = 13.2528,
            mu: float = 0.2500,
            beta: float = 3.9847,
            n: int = 1,
            bc: BoundariesData = 'auto_periodic_neumann',
            noise: float = 0,
            rng: Optional[np.random.Generator] = None,
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
        super().__init__(diffusivity=diffusivity, bc=bc, noise=noise, rng=rng)
        self.mu = mu
        self.beta = beta
        self.n = n

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, value: float):
        self._mu = value

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value

    @staticmethod
    @njit
    def _power_sigmoid(x: float | np.ndarray[float], n: float) -> float | np.ndarray[float]:
        """
        Should only be used internally.
        """
        return 1 / (1 + np.exp(-x ** n))

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

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})
        diff_term = SigmoidDiffusivityPDE._power_sigmoid(self.beta - self.diffusivity * state.data, self.n)
        return diff_term * laplace_applied - self.mu * state

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

        array_type = nb.typeof(state.data)
        signature = array_type(array_type, nb.double)

        diffusivity = self.diffusivity
        mu = self.mu
        beta = self.beta
        n = self.n
        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)
        power_sigmoid = SigmoidDiffusivityPDE._power_sigmoid

        @njit(signature)
        def pde_rhs(state_data: np.ndarray, t: float) -> np.ndarray:
            diff_term = power_sigmoid(beta - diffusivity * state_data, n)
            return diff_term * laplace_operator(state_data, args={'t': t}) - mu * state_data

        return pde_rhs

    def update_parameters(self, parameters: Parameters) -> None:
        """
        Update LinearDiffusivityPDE parameters (`diffusivity`, `mu`, `beta`, `n`) from a set of lmfit.Parameters.

        Args:
            parameters (Parameters): The lmfit.Parameters object containing parameter values.
        """
        parameter_names = ('diffusivity', 'mu', 'beta', 'n')
        for parameter in parameters.values():
            if parameter.name in parameter_names:
                setattr(self, parameter.name, parameter.value)


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
            diffusivity: float = 0.1674,
            lambda_term: float = 0.2284,
            alpha: float = 1,
            bc: BoundariesData = 'auto_periodic_neumann',
            noise: float = 0,
            rng: Optional[np.random.Generator] = None,
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
        super().__init__(diffusivity=diffusivity, bc=bc, noise=noise, rng=rng)
        self.lambda_term = lambda_term
        self.alpha = alpha

    @property
    def lambda_term(self) -> float:
        return self._lambda_term

    @lambda_term.setter
    def lambda_term(self, value: float):
        self._lambda_term = value

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

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

        laplace_applied = state.laplace(bc=self.bc, label='evolution rate', args={'t': t})
        return self.diffusivity * laplace_applied + self.alpha * state * (self.lambda_term - state)  # type: ignore

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

        array_type = nb.typeof(state.data)
        signature = array_type(array_type, nb.double)

        diffusivity = self.diffusivity
        lambda_term = self.lambda_term
        alpha = self.alpha
        laplace_operator = state.grid.make_operator('laplace', bc=self.bc)

        @njit(signature)
        def pde_rhs(state_data: np.ndarray, t: float) -> np.ndarray:
            return (diffusivity * laplace_operator(state_data, args={'t': t})
                    + alpha * state_data * (lambda_term - state_data))

        return pde_rhs

    def update_parameters(self, parameters: Parameters):
        """
        Update LogisticDiffusionPDE parameters (`diffusivity`, `lambda_term`, `alpha`) from a set of lmfit.Parameters.

        Args:
            parameters (Parameters): The lmfit.Parameters object containing parameter values.
        """
        parameter_names = ('diffusivity', 'lambda_term', 'alpha')
        for parameter in parameters.values():
            if parameter.name in parameter_names:
                setattr(self, parameter.name, parameter.value)
