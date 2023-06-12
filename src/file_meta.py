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

    @property
    def name(self) -> str:
        return self._name

    @property
    def drive_id(self) -> str:
        return self._drive_id

    @property
    def folder(self) -> str:
        return self._folder

    @property
    def homogenized(self) -> str:
        return self._homogenized_id

    @property
    def is_corner(self):
        """
        Checks if the file is located in the 'sarok' folder.
        """
        return self.folder in ('sarok', 'Ca2+')
