import os.path


class FileMeta:
    """
    Represents metadata for a file. Contains information about the files name, its Google Drive id, the parent folder
    and an alias for the filename.
    """

    def __init__(self, name: str, drive_id: str, folder: str, alias: str):
        self._name = name
        self._drive_id = drive_id
        self._folder = folder
        self._alias = alias

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
        return FileMeta(split[-1], None, split[-2] if len(split) > 1 else '.', None)

    @staticmethod
    def from_name(name):
        """
        Creates a FileMeta object where drive_id, folder and homogenized_id are None.
        """
        return FileMeta(name, None, None, None)

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
    def alias(self) -> str:
        return self._alias

    @property
    @_non_null_getter
    def is_corner(self):
        """
        Checks if the file is located in the 'sarok' folder.
        """
        return self.folder in ('sarok', 'Ca2+')
