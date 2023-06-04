from abc import ABC, abstractmethod
from typing import Iterator, Any

import numpy as np

from reader import Reader
from writer import Writer


class DiffusionArray:
    """
    Wrapper class for a numpy array representing the diffusion of ATP in fish.

    Attributes:
        ndarray (np.array): The numpy array representing the diffusion data.
    """

    def __init__(self, path: str | None, ndarray: np.ndarray | None = None):
        """
        Constructs a DiffusionArray object by loading data from a file.

        Parameters:
            path (str): The path to the file containing the diffusion data.
            ndarray (np.ndarray): an array already containing the data, if given path will be ignored

        Raises:
            ValueError: If the file format is not supported or the file cannot be loaded, or path and ndarray are
            both None
        """

        self._index_strategy = _DefaultIndexStrategy()

        if path is None and ndarray is None:
            raise ValueError('path and ndarray were both none')

        if ndarray is None:
            self.ndarray = Reader.of_type(path).read(path)
        else:
            self.ndarray = ndarray

        if self.ndarray.ndim != 4:
            # assuming channel dimension is missing TODO: dimension generating strategy
            self.ndarray = np.expand_dims(self.ndarray, axis=1)

    def save(self, path: str):
        """
        Save the diffusion data to a file. The whole file is saved, regardless of the index strategy

        Parameters:
            path (str): The path to save the diffusion data.

        Raises:
            ValueError: If the file extension is not supported or the data cannot be saved.
        """
        try:
            Writer.of_type(path).write(path, self.ndarray)
        except Exception as e:
            raise ValueError("Failed to save diffusion data to file.") from e

    def __iter__(self) -> Iterator:
        """
        Returns an iterator over the diffusion ndarray.

        Returns:
            iterator: An iterator over the diffusion ndarray.
        """
        return self._index_strategy.iter(self.ndarray)

    def __getitem__(self, key: Any) -> Any:
        """
        Returns the value(s) at the specified index or slice of the diffusion ndarray.

        Parameters:
            key: Index or slice object specifying the indices or range of indices to retrieve.

        Returns:
            np.array: The value(s) at the specified index or slice of the diffusion ndarray.
        """
        return self._index_strategy.getitem(self.ndarray, key)

    def __array__(self) -> np.ndarray:
        """
        Returns the underlying  ndarray, sliced by the index strategies.

        Returns:
            np.ndarray: The diffusion ndarray.
        """
        return self[:]

    @property
    def ndarray(self) -> np.array:
        """
        Getter method for the data attribute.

        Returns:
            np.array: The diffusion data array.
        """
        return self._ndarray

    @ndarray.setter
    def ndarray(self, new_data: np.array):
        """
        Setter method for the data attribute.

        Parameters:
            new_data (np.array): The new diffusion data array.
        """
        self._ndarray = new_data

    def frame(self, frame: slice | int | str) -> 'DiffusionArray':
        """
        Extracts frame(s) from the DiffusionArray object, using an index strategy

        Parameters:
            frame: The index of the frame to extract. Can be an integer,
                a slice, or a slice represented as a string (e.g., '1:4').

        Returns:
            DiffusionArray: a new DiffusionArray object with the new index strategy
        """
        if isinstance(frame, str):
            frame = slice(*([int(x) for x in frame.split(':')]))

        other = DiffusionArray(path=None, ndarray=self.ndarray)
        other._index_strategy = self._index_strategy.frame_extracted(frame)
        return other

    def channel(self, channel: slice | int | str) -> 'DiffusionArray':
        """
        Extracts  channel(s) from the DiffusionArray object, using an index strategy

        Parameters:
            channel: The index of the frame to extract. Can be an integer,
                a slice, or a slice represented as a string (e.g., '1:4').

        Returns:
            DiffusionArray: a new DiffusionArray object with the new index strategy.
        """
        if isinstance(channel, str):
            channel = slice(*([int(x) for x in channel.split(':')]))

        other = DiffusionArray(path=None, ndarray=self.ndarray)
        other._index_strategy = self._index_strategy.channel_extracted(channel)
        return other

    @property
    def number_of_frames(self) -> int:
        """
        Returns the number of frames in the ndarray.

        Returns:
        - int: The number of frames.
        """
        return self.ndarray.shape[0]

    @property
    def number_of_channels(self) -> int:
        """
        Returns the number of channels in the ndarray.

        Returns:
        - int: The number of channels.
        """
        return self.ndarray.shape[1]

    @property
    def shape(self):
        return self[:].shape


# index: frame, channel, x, y
class _IndexStrategy(ABC):
    """
    Base class for index strategies in the DiffusionArray class.

    Subclasses of _IndexStrategy function as a finite state machine to handle frame and channel extraction.
    They provide methods for modifying the index strategy and iterating over the data array.
    """

    @abstractmethod
    def frame_extracted(self, frame) -> '_IndexStrategy':
        """
        Modifies the index strategy to extract frame(s) from the ndarray.

        Parameters:
            frame: The index of the frame to extract.

        Returns:
            _IndexStrategy: A new instance of the index strategy with the frame extraction applied.
        """
        raise NotImplementedError("frame_extracted method must be implemented in a derived class.")

    @abstractmethod
    def channel_extracted(self, channel) -> '_IndexStrategy':
        """
        Modifies the index strategy to extract channel(s) from the ndarray.

        Parameters:
            channel: The index of the channel to extract.

        Returns:
            _IndexStrategy: A new instance of the index strategy with the channel extraction applied.
        """
        raise NotImplementedError("channel_extracted method must be implemented in a derived class.")

    @abstractmethod
    def iter(self, ndarray: np.ndarray) -> Iterator:
        """
        Iterates over the data array based on the index strategy.

        Parameters:
            ndarray (np.ndarray): The data array to iterate over.

        Returns:
            Iterator: An iterator over the selected elements of the data array.
        """
        raise NotImplementedError("iter method must be implemented in a derived class.")

    @abstractmethod
    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        """
        Retrieves the selected elements from the data array based on the index strategy.

        Parameters:
            ndarray (np.ndarray): The data array to extract elements from.
            key (Any): The key used to select elements from the array.

        Returns:
            Any: The selected elements from the data array.
        """
        raise NotImplementedError("getitem method must be implemented in a derived class.")


class _DefaultIndexStrategy(_IndexStrategy):
    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray)

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        return ndarray[key]

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _FrameExtractedIndexStrategy(frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelExtractedIndexStrategy(channel)


class _ChannelExtractedIndexStrategy(_IndexStrategy):
    def __init__(self, channel):
        self.channel = channel

    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray[:, self.channel, :, :])

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        return ndarray[:, self.channel, :, :][key]

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(self.channel, frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelExtractedIndexStrategy(channel)


class _FrameExtractedIndexStrategy(_IndexStrategy):
    def __init__(self, frame):
        self.frame = frame

    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray[self.frame, ...])

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        return ndarray[self.frame, ...][key]

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _FrameExtractedIndexStrategy(frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(channel, self.frame)


class _ChannelAndFrameExtractedIndexStrategy(_IndexStrategy):
    def __init__(self, channel, frame):
        self.channel = channel
        self.frame = frame

    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray[self.frame, self.channel, :, :])

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        return ndarray[self.frame, self.channel, :, :][key]

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(self.channel, frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(channel, self.frame)