from abc import abstractmethod, ABC

import nd2
import numpy as np
from nd2.structures import Metadata


class Reader(ABC):
    """Abstract base class for file readers."""

    @abstractmethod
    def read(self, path: str) -> np.array:
        """
        Abstract method to read data from a file.

        Parameters:
            path (str): The path to the file.

        Returns:
            np.array: The loaded data as a numpy array.
        """
        raise NotImplementedError("Write method must be implemented in a derived class.")

    @staticmethod
    @abstractmethod
    def supported_extension() -> str:
        """
        Abstract method to return the supported file extension for the reader.

        Returns:
            str: The supported file extension for the reader.
        """
        raise NotImplementedError("supported_extension method must be implemented in a derived class.")

    @staticmethod
    def of_type(path: str) -> 'Reader':
        """
        Static method to create an appropriate reader based on the file type.

        Parameters:
            path (str): The path to the file.

        Returns:
            Reader: An instance of the specific reader class.

        Raises:
            ValueError: If the file extension is not supported.
        """

        extension = path.split('.')[-1].lower()

        for subclass in Reader.__subclasses__():
            if extension == subclass.supported_extension():
                return subclass()

        raise ValueError(f"Unsupported file extension: {extension}")


class NPZReader(Reader):
    @staticmethod
    def supported_extension() -> str:
        return 'npz'

    def read(self, path: str) -> np.array:
        if not path.lower().endswith('.npz'):
            raise ValueError("Invalid file extension. Expected '.npz'")
        a = np.load(path)
        return np.load(path)['arr_0']


class ND2Reader(Reader):
    @staticmethod
    def supported_extension() -> str:
        return 'nd2'

    def read(self, path: str) -> np.array:
        if not path.lower().endswith('.nd2'):
            raise ValueError("Invalid file extension. Expected '.nd2'")
        return nd2.imread(path)

    @staticmethod
    def meta(path: str) -> Metadata | dict:
        with nd2.ND2File(path) as nd_file:
            return nd_file.metadata
