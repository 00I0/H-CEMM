import os.path


class FileMeta:
    """
    Represents metadata for a file. Contains information about the files name, its Google Drive id, the parent folder
    and the Google Drive id of the homogenized version of itself
    """

    def __init__(self, name: str, drive_id: str, folder: str, homogenized_id: str):
        self._name = name
        self._drive_id = drive_id
        self._folder = folder
        self._homogenized_id = homogenized_id

    @staticmethod
    def from_path(path: str) -> 'FileMeta':
        """
        Creates a FileMeta object based on the provided path. drive_id and homogenized_id will be None.

        Parameters:
            path (str): The fully qualified path for the diffusion data

        Returns:
            The FileMeta object for this path
        """

        split = path.split(os.path.sep)
        return FileMeta(split[-1], None, split[-2], None)

    @staticmethod
    def _non_null_getter(getter):
        def inner(self):
            value = getter(self)
            if value is None:
                field_name = getter.__name__.replace("_get_", "")
                raise AttributeError(f'{field_name} is None')
            return value

        return inner

    @property
    @_non_null_getter
    def name(self) -> str:
        return self._name

    @property
    @_non_null_getter
    def drive_id(self) -> str:
        return self._drive_id

    @property
    @_non_null_getter
    def folder(self) -> str:
        return self._folder

    @property
    @_non_null_getter
    def homogenized(self) -> str:
        return self._homogenized_id

    @property
    @_non_null_getter
    def is_corner(self):
        """
        Checks if the file is located in the 'sarok' folder.
        """
        return self.folder in ('sarok', 'Ca2+')
