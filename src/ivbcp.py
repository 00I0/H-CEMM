import functools
from abc import abstractmethod, ABC
from typing import Optional, List

import numpy as np

from diffusion_PDEs import DiffusionPDEBase
from radial_time_profile import RadialTimeProfile


class SymmetricIVBCPBase(ABC):
    """
    Abstract base class for symmetric Initial Value Boundary Condition Problems (IVPBCs).

    Properties:
        inner_radius (int): The inner radius of the radial time profile, calculated based on the starting frame and
            mean intensity.
        file_name (str): The original name of the radial time profile.
        spatial_size (int): The width of the original radial time profile.
        sec_per_frame (float): The number of seconds per frame (default: 0.3).
        width (int): The width of the resized radial time profile.
        pde (DiffusionPDEBase): The diffusion PDE associated with the IVPBC.
        initial_condition (np.ndarray): The initial condition for the IVPBC (must be implemented by subclasses).
        bc (List): The boundary conditions for the IVPBC (must be implemented by subclasses).
        frames (int): The number of frames for the IVPBC (must be implemented by subclasses).
        expected_values (np.ndarray): The expected values for the IVPBC (must be implemented by subclasses).

    Methods:
        resized(resolution: int): Abstract method to resize the problem to a new resolution.
    """

    def __init__(
            self,
            radial_time_profile: RadialTimeProfile,
            start_frame: int,
            pde: DiffusionPDEBase,
            _resolution: Optional[int] = None
    ):
        """
        Initializes a new instance of the SymmetricIVBCPBase.

        Args:
            radial_time_profile (RadialTimeProfile): Radial profile describing the time evolution of the problem.
            start_frame (int): Frame at which the PDE simulation should start.
            pde (DiffusionPDEBase): Partial differential equation to be solved.
            _resolution (Optional[int]): Resolution to resize the radial time profile to. Defaults to the profile's
                width. Should only be used internally, or in a derived class.
        """
        self._radial_time_profile = radial_time_profile
        self._start_frame = start_frame
        self._pde = pde
        self._resolution = _resolution or radial_time_profile.width

        frame_of_max_intensity = np.argmax(np.max(self._radial_time_profile, axis=1))
        pde.bc = self.bc
        pde.norm_mu = np.argmax(self._resized_rtp.frame(frame_of_max_intensity)) - self.inner_radius
        pde.norm_sigma = pde.norm_mu / 2
        pde.T = -1

    @abstractmethod
    def resized(self, resolution: int) -> 'SymmetricIVBCPBase':
        """
        Resizes the problem to a new resolution. Must be implemented by subclasses.

        Args:
            resolution (int): The new resolution.

        Returns:
            SymmetricIVBCPBase: A new instance of a subclass with the specified resolution.
        """
        raise NotImplementedError

    @property
    @functools.cache
    def _resized_rtp(self) -> RadialTimeProfile:
        return self._radial_time_profile.resized(self._resolution, 1)

    @property
    @functools.cache
    def inner_radius(self) -> int:
        resized_rtp = self._resized_rtp
        inner_radius = resized_rtp.width
        mean = np.mean(resized_rtp.frame(self._start_frame - 1))

        for f in range(self._start_frame, resized_rtp.number_of_frames - 1):
            for r in range(np.argmax(resized_rtp.frame(f)), -1, -1):
                if resized_rtp.frame(f)[r] >= mean >= resized_rtp.frame(f)[r - 1]:
                    inner_radius = min(inner_radius, r)
                    break

        return inner_radius

    @property
    def file_name(self) -> str:
        return self._radial_time_profile.original_name

    @property
    def spatial_size(self) -> int:
        return self._radial_time_profile.width

    @property
    def sec_per_frame(self) -> float:
        return 0.3

    @property
    def width(self) -> int:
        return self._resized_rtp.width

    @property
    def pde(self) -> DiffusionPDEBase:
        return self._pde

    @property
    @abstractmethod
    def initial_condition(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def bc(self) -> List:
        raise NotImplementedError

    @property
    @abstractmethod
    def frames(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def expected_values(self) -> np.ndarray:
        raise NotImplementedError


class VanillaSymmetricIVBCP(SymmetricIVBCPBase):
    """
    Subclass of SymmetricIVBCPBase representing an IVBCP where the initial condition is the frame with the greatest
        intensity. The border condition on the left is this: c(t, 0) = c(0,0).
    """

    def resized(self, resolution: int) -> 'VanillaSymmetricIVBCP':
        return VanillaSymmetricIVBCP(self._radial_time_profile, self._start_frame, self._pde, resolution)

    @property
    def initial_condition(self) -> np.ndarray:
        rtp = self._radial_time_profile
        return self._resized_rtp.frame(np.argmax(np.max(rtp, axis=1)))[self.inner_radius:]

    @property
    def bc(self):
        return ['periodic', [{'value': self.initial_condition[0]}, {'derivative': 0}]]

    @property
    def frames(self) -> int:
        rtp = self._radial_time_profile
        return rtp.number_of_frames - 1 - np.argmax(np.max(rtp, axis=1))

    @property
    def expected_values(self) -> np.ndarray:
        frame_of_max_intensity = np.argmax(np.max(self._radial_time_profile, axis=1))
        return self._resized_rtp.frame(f'{frame_of_max_intensity}:-1')[:, self.inner_radius:]


class NormalDistributionSymmetricIVBCP(SymmetricIVBCPBase):
    """
    Subclass of SymmetricIVBCPBase representing an IVBCP where the initial condition is the frame where the process
    begun, the border condition on the left is this: c(t, 0) = c(0,0).

    The intended use-case is when a normal distribution is added from the first frame up until the frame of
    max intensity, for that please ensure that the pde's `s` parameter is not 0.
    """

    def __init__(
            self,
            radial_time_profile: RadialTimeProfile,
            start_frame: int,
            pde: DiffusionPDEBase,
            _resolution: Optional[int] = None
    ):
        super().__init__(radial_time_profile, start_frame, pde, _resolution)

        frame_of_max_intensity = np.argmax(np.max(self._radial_time_profile, axis=1))
        pde.T = frame_of_max_intensity * self.sec_per_frame

    def resized(self, resolution: int) -> 'NormalDistributionSymmetricIVBCP':
        return NormalDistributionSymmetricIVBCP(self._radial_time_profile, self._start_frame, self._pde, resolution)

    @property
    def initial_condition(self) -> np.ndarray:
        return self._resized_rtp.frame(self._start_frame)[self.inner_radius:]

    @property
    def bc(self) -> List:
        return ['periodic', [{'value': self.initial_condition[0]}, {'derivative': 0}]]

    @property
    def frames(self) -> int:
        return self._resized_rtp.number_of_frames - self._start_frame - 1

    @property
    def expected_values(self) -> np.ndarray:
        return self._resized_rtp.frame(f'{self._start_frame}:-1')[:, self.inner_radius:]


class DerivativeSymmetricIVBCP(SymmetricIVBCPBase):
    """
    Subclass of SymmetricIVBCPBase representing an IVBCP where the initial condition is the frame where the process
    begun, the border condition on the left is this: c'(t, 0) = k if t < frame of max intensity and c'(t, 0) = 0
    otherwise.
    """

    def resized(self, resolution: int) -> 'DerivativeSymmetricIVBCP':
        return DerivativeSymmetricIVBCP(self._radial_time_profile, self._start_frame, self._pde, resolution)

    @property
    def initial_condition(self) -> np.ndarray:
        return self._resized_rtp.frame(self._start_frame)[self.inner_radius:]

    @staticmethod
    def _bc_function(_adjacent_value, _dx, _x, _y, t, area_between_the_curves, dt):
        return area_between_the_curves / dt if t <= dt else 0

    @property
    def bc(self) -> List:
        frame_of_max_intensity = np.argmax(np.max(self._resized_rtp, axis=1))
        dt = (frame_of_max_intensity - self._start_frame) * self.sec_per_frame

        area_between_the_curves = np.sum(
            self._radial_time_profile.frame(frame_of_max_intensity) - self._radial_time_profile.frame(self._start_frame)
        )

        def _bc_function(_adjacent_value, _dx, _x, _y, t):
            return area_between_the_curves / (dt * 1200) if t <= dt else 0

        return ['periodic', [{'derivative_expression': _bc_function}, {'derivative': 0}]]

    @property
    def frames(self) -> int:
        return self._resized_rtp.number_of_frames - 1 - self._start_frame

    @property
    def expected_values(self) -> np.ndarray:
        return self._resized_rtp.frame(f'{self._start_frame}:-1')[:, self.inner_radius:]
