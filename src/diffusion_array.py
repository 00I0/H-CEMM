import os
import re
from abc import ABC, abstractmethod

import numpy as np
from typing import Iterator, Any, Optional, List, Callable, Tuple

from src.file_meta import FileMeta
from src.mask import Mask
from src.reader import Reader
from src.writer import Writer


class DiffusionArray:
    """
    Wrapper class for a numpy array representing the diffusion of ATP or Ca2+ in fish.
    """

    def __init__(self, path: str | FileMeta | None, ndarray: np.ndarray | None = None):
        """
        Constructs a DiffusionArray object by loading data from a file. If the data is already in memory a NumPy array
        it could be passed as a NumPy array, in that case metadata about the file will not be created. The NumPy array
        must contain the two spacial dimensions, the time dimension and optionally the channels.

        Parameters:
            path (str | FileMeta): The path to the file containing the diffusion data
            (could also be wrapped in a FileMeta object).
            ndarray (np.ndarray): an array already containing the data, if given path will be ignored

        Raises:
            ValueError: If the file format is not supported or the file cannot be loaded, or path and ndarray are
            both None
        """

        self._index_strategy = _DefaultIndexStrategy()

        if path is None and ndarray is None:
            raise ValueError('path and ndarray were both none')

        if ndarray is None:
            if isinstance(path, FileMeta):
                self._meta = path
                self._ndarray = Reader.of_type(path.name).read(path.name)
            else:
                self._meta = FileMeta.from_path(path)
                self._ndarray = Reader.of_type(path).read(path)
        else:
            if ndarray.ndim not in (3, 4):
                raise ValueError('The numpy array must contain the two spacial dimensions and the time dimension')
            self._ndarray = ndarray

        self._ndarray = self.ndarray.astype(np.int32)

        self._cached = {}

        if self.ndarray.ndim == 3:
            # assuming channel dimension is missing TODO: dimension generating strategy
            self._ndarray = np.expand_dims(self.ndarray, axis=1)
            self._index_strategy = self._index_strategy.channel_extracted(0)

    @staticmethod
    def get_all_from_directory(directory: str, regex: Optional[str] = '.*\\.(nd2|npz)$') -> List['DiffusionArray']:
        """
        Create DiffusionArray instances from files matching a regex pattern in a directory.

        This method takes a directory path and an optional regex pattern as input, scans the directory
        for files matching the pattern (or all files if no pattern is provided), and creates DiffusionArray
        instances for each matching file.

        Args:
            directory (str): The directory path where files are located.
            regex (str, optional): A regular expression pattern as a string.
                If not provided, all files in the directory are considered matching.

        Returns:
            List[DiffusionArray]: A list of DiffusionArray instances created from matching files.
        """
        matching_files = []
        regex = re.compile(regex)

        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and regex.match(filename):
                matching_files.append(DiffusionArray(filepath))

        return matching_files

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

    def __setitem__(self, key: Any, value: int | float | np.ndarray):
        """
        Sets the value(s) at the specified index or slice of the diffusion ndarray, to the value provided

        Parameters:
            key: Index or slice object specifying the indices or range of indices to retrieve.
            value: The new value of the items selected by the key

        """
        self._ndarray = self._index_strategy.setitem(self._ndarray, key, value)

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
        other._cached = self._cached
        if hasattr(self, '_meta'):
            other._meta = self.meta
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
        other._cached = self._cached
        if hasattr(self, '_meta'):
            other._meta = self.meta
        return other

    @property
    def number_of_frames(self) -> int:
        """
        Returns the number of frames in the ndarray.

        Returns:
            int: The number of frames.
        """
        return self.ndarray.shape[0]

    @property
    def number_of_channels(self) -> int:
        """
        Returns the number of channels in the ndarray.

        Returns:
            int: The number of channels.
        """
        return self.ndarray.shape[1]

    @property
    def width(self) -> int:
        """
        Returns the width of the ndarray.

        Returns:
            int: The width of the underlying ndarray
        """
        return self.ndarray.shape[-2]

    @property
    def height(self) -> int:
        """
        Returns the height of the ndarray.

        Returns:
            int: The height of the underlying ndarray
        """
        return self.ndarray.shape[-1]

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the underlying ndarray with the proper index strategy applied

        Returns:
            tuple: A tuple representing the shape of the object.
        """
        return self[:].shape

    def resized(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> 'DiffusionArray':
        """
        Crop and resize the DiffusionArray object to a new rectangular region defined by the 'top_left' and
        'bottom_right' corners. Please ensure that both 'top_left' and 'bottom_right' are within a valid range of the
        original arrays size and the rectangle defined by them has sides with positive lengths.

        Args:
            top_left (Tuple[int, int]): The (x, y) coordinates of the top-left corner of the new region.
            bottom_right (Tuple[int, int]): THe (x, y) coordinates ot hte bottom-right corner of the new region.

        Returns:
            DiffusionArray: A new DiffusionArray containing with the cropped region.
        """
        first_x, first_y = top_left
        end_x, end_y = bottom_right

        if first_x < 0 or first_y < 0:
            raise ValueError(f'top_left must have non-negative coordinates, but they were: ({first_x}, {first_y})')

        if first_x >= end_x or first_y >= end_y:
            raise ValueError(f'The pairwise difference of bottom_right and top_left must be positive, but it was: '
                             f'({end_x - first_x}, {end_y - first_y})')

        if end_y > self.height or end_x > self.width:
            raise ValueError(f'The coordinates of bottom_right must be inside the height and width of the diffusion '
                             f'array ({self.height}, {self.width}), but they were: ({end_x}, {end_y})')

        return self.updated_ndarray(
            self.ndarray[:, :, first_x: end_x, first_y: end_y]
        )

    def updated_ndarray(self, ndarray: np.ndarray) -> 'DiffusionArray':
        """
        Creates a new DiffusionArray with a new ndarray.

        Parameters:
            ndarray (np.ndarray): The new ndarray to update the DiffusionArray.

        Returns:
            DiffusionArray: A new DiffusionArray object with the updated ndarray.
        """
        darr = DiffusionArray(path=None, ndarray=ndarray)
        darr._index_strategy = self._index_strategy

        try:
            darr._meta = self.meta
        except RuntimeError:
            pass
        return darr

    def copy(self) -> 'DiffusionArray':
        """
        Creates a copy of the DiffusionArray object.

        Returns:
            DiffusionArray: A new DiffusionArray object with the same ndarray and index strategy.
        """
        darr = DiffusionArray(path=None, ndarray=self.ndarray)
        darr._index_strategy = self._index_strategy
        darr._cached = self._cached.copy()
        if hasattr(self, '_meta'):
            darr._meta = self.meta
        return darr

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions of the underlying ndarray with the proper index strategy applied.

        Returns:
            int: The number of dimensions of the object.
        """
        return self[:].ndim

    @property
    def meta(self) -> FileMeta:
        """
        Returns the FileMeta object associated with the DiffusionArray.

        Returns:
            FileMeta: The FileMeta object associated with the DiffusionArray.

        Raises:
            RuntimeError: If the FileMeta object was not provided.
        """
        if not hasattr(self, '_meta'):
            raise RuntimeError('File meta was not provided')
        return self._meta

    def cache(self, **kwargs):
        """
        Cache the provided key-value pairs in the object's cache. Cache is persistent between DiffusionArrays created
        by frame() or channel(). It is  not persistent between DiffusionArrays created by copy() or updtate_array()

        Parameters:
            **kwargs (key-value pairs): Key-value pairs to be cached.

        Returns:
            None
        """
        self._cached.update(kwargs)

    def get_cached(self, key) -> Any:
        """
        Retrieve the cached value associated with the specified key.

        Parameters:
            key: The key for which the cached value is desired.

        Returns:
            The cached value corresponding to the key, or None if the key is not found.
        """
        try:
            return self._cached[key]
        except KeyError:
            return None

    def normalized(self, new_min: int | float = 0, new_max: int | float = 1) -> 'DiffusionArray':
        """
        This method scales the values in the DiffusionArray object to be between 'new_min' and 'new_max'. Scaling is
        performed using a linear operator where the minimum value in the array becomes 'new_min' and the maximum
        'new_max'.

        Args:
            new_min (int | float): The minimum value for the scaled range (default is 0).
            new_max (int | float): THe maximum value for the scaled range (default is 1).

        Returns:
            DiffusionArray: A new DiffusionArray object with values scaled to the given range.
        """
        min_value = self.ndarray.min()
        max_value = self.ndarray.max()

        normalized_array = new_min + (self.ndarray * (new_max - new_min) - min_value) / (max_value - min_value)
        return self.updated_ndarray(normalized_array)

    def background_removed(self, background_slices: str = '0:3', aggregator: Callable = np.mean) -> 'DiffusionArray':
        """
        This method removes the background form a DiffusionArray, by subtracting an aggregated background form all
        frames. The aggregate background is constructed by applying the 'aggregator' function on the frames selected by
        'background_slices'.

        Args:
            background_slices (str): A string specifying the slice range of frames to be used as background reference.
            By default, it is '0:3'.
            aggregator (Callable): A function to aggregate the selected frames. The default aggregator is np.mean.

        Returns:
            DiffusionArray: A new DiffusionArray object with the background removed.
        """
        return self.updated_ndarray(self[:] - aggregator(self.frame(background_slices), axis=0))


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
        Returns the appropriate index strategy to extract frame(s) from the ndarray.

        Parameters:
            frame: The index of the frame to extract.

        Returns:
            _IndexStrategy: A new instance of the index strategy with the frame extraction applied.
        """
        raise NotImplementedError("frame_extracted method must be implemented in a derived class.")

    @abstractmethod
    def channel_extracted(self, channel) -> '_IndexStrategy':
        """
        Returns the appropriate index strategy to extract channel(s) from the ndarray.

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

    @abstractmethod
    def setitem(self, ndarray: np.ndarray, key: Any, value: Any) -> np.ndarray:
        """
        Sets the selected elements in the data array based on the index strategy to the value provided

        Parameters:
            ndarray (np.ndarray): The data array to extract elements from.
            key (Any): The key used to select elements from the array.
            value (Any) The new value of the selected items

        Returns:
            the modified ndarray

        """
        raise NotImplementedError("getitem method must be implemented in a derived class.")


class _DefaultIndexStrategy(_IndexStrategy):
    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray)

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        print(type(key))
        return ndarray[key]

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _FrameExtractedIndexStrategy(frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelExtractedIndexStrategy(channel)

    def setitem(self, ndarray, key, value) -> np.ndarray:
        if isinstance(key, Mask):
            raise ValueError('A channel must be selected first to apply a mask')
        else:
            ndarray[key] = value
        return ndarray


class _ChannelExtractedIndexStrategy(_IndexStrategy):
    def __init__(self, channel):
        self.channel = channel

    def iter(self, ndarray: np.ndarray) -> Iterator:
        return iter(ndarray[:, self.channel, :, :])

    def getitem(self, ndarray: np.ndarray, key: Any) -> Any:
        if isinstance(key, Mask):
            ans = []
            for i in range(ndarray.shape[0]):
                for_frame = key.for_frame(i)
                ans.append(ndarray[i, self.channel, for_frame])
            return np.array(ans)
        else:
            return ndarray[:, self.channel, :, :][key]

    def setitem(self, ndarray: np.ndarray, key: Any, value: Any) -> np.ndarray:

        if isinstance(key, Mask):
            for i in range(ndarray.shape[0]):
                for_frame = key.for_frame(i)
                if isinstance(value, np.ndarray):
                    ndarray[i, self.channel, for_frame] = value[i]
                else:
                    ndarray[i, self.channel, for_frame] = value

        else:
            ndarray[:, self.channel, :, :][key] = value

        return ndarray

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

    def setitem(self, ndarray: np.ndarray, key: Any, value: Any) -> np.ndarray:
        if isinstance(key, Mask):
            raise ValueError('A channel must be selected first to apply a mask')
        else:
            ndarray[self.frame, ...][key] = value
        return ndarray

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

        if isinstance(key, Mask):
            if isinstance(self.frame, int):
                return ndarray[self.frame, self.channel, key.for_frame(self.frame)]

            if isinstance(self.frame, slice):
                selected = []
                for i in range(self.frame.start, self.frame.stop, self.frame.step if None else 1):
                    for_frame = key.for_frame(i)
                    selected.append(ndarray[i, self.channel, for_frame])
                try:
                    return np.array(selected)
                except ValueError:
                    max_len = max(len(arr) for arr in selected)
                    padded_selected = [np.pad(arr, (0, max_len - len(arr)), mode='empty') for arr in selected]
                    return np.array(padded_selected)

            else:
                raise ValueError(f'Unexpected frame type ({type(self.frame)})')

        return ndarray[self.frame, self.channel, :, :][key]

    def setitem(self, ndarray: np.ndarray, key: Any, value: Any) -> np.ndarray:
        if isinstance(key, Mask):
            for_frame = key.for_frame(self.frame)
            if isinstance(value, np.ndarray):
                try:
                    ndarray[self.frame, self.channel, for_frame] = value[self.frame]
                except ValueError:
                    ndarray[self.frame, self.channel, for_frame] = value[self.frame, np.newaxis]
            else:
                ndarray[self.frame, self.channel, for_frame] = value
        else:
            if isinstance(value, np.ndarray):
                ndarray[self.frame, self.channel, ...] = value[self.frame]
            else:
                ndarray[self.frame, self.channel, ...] = value

        return ndarray

    def frame_extracted(self, frame) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(self.channel, frame)

    def channel_extracted(self, channel) -> '_IndexStrategy':
        return _ChannelAndFrameExtractedIndexStrategy(channel, self.frame)
