from abc import ABC, abstractmethod

import numpy as np


class Writer(ABC):
    """Abstract base class for file writers."""

    @abstractmethod
    def write(self, path: str, data: np.array):
        """
        Abstract method to write data to a file.

        Parameters:
            path (str): The path to the file.
            data (np.array): The data to be written.

        Raises:
            NotImplementedError: If the write method is not implemented in a derived class.
        """
        raise NotImplementedError("Write method must be implemented in a derived class.")

    @staticmethod
    @abstractmethod
    def supported_extension() -> str:
        """
        Abstract method to return the supported file extension for the writer.

        Returns:
            str: The supported file extension for the writer.
        """
        raise NotImplementedError("supported_extension method must be implemented in a derived class.")

    @staticmethod
    def of_type(path: str) -> 'Writer':
        """
        Static method to create an appropriate writer based on the file type.

        Parameters:
            path (str): The path to the file.

        Returns:
            Writer: An instance of the specific writer class based on the file extension.

        Raises:
            ValueError: If the file extension is not supported.
        """
        extension = path.split('.')[-1].lower()

        for subclass in Writer.__subclasses__():
            if extension == subclass.supported_extension():
                return subclass()

        raise ValueError(f"Unsupported file extension: {extension}")


class NPZWriter(Writer):
    def write(self, path: str, data: np.array):
        if not path.lower().endswith('.npz'):
            raise ValueError("Invalid file extension. Expected '.npz'")

        np.savez(path, data)

    @staticmethod
    def supported_extension() -> str:
        return 'npz'
