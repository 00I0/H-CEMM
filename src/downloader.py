import json

import gdown


class Downloader:
    """
    A class for downloading files from Google Drive using file names and their corresponding Google Drive IDs.

    The class requires a JSON file that contains a mapping of file names to their Google Drive IDs. It provides a
    method to download a file given its name.

    Args:
        config (dict): A dictionary mapping file names to their corresponding Google Drive IDs.

    Methods:
        download_file: Downloads a file from Google Drive given its name.
        from_json: Creates a Downloader object from a JSON file containing file names and Google Drive IDs.
    """

    def __init__(self, config: dict[str, str]):
        self.config = config

    def download_file(self, file_name: str) -> None:
        """
        Downloads a file from Google Drive given its name.

        Args:
            file_name (str): The name of the file to be downloaded.

        Raises:
            ValueError: If the provided file_name is not in self.config.

        Returns:
            None
        """
        if file_name not in self.config:
            raise ValueError(f"The provided file_name ({file_name}) must be in self.config, but it was not")

        file_id = self.config[file_name]
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)

    def list_file_names(self) -> list[str]:
        """
        Returns a list of file names available in the Downloader's configuration.

        Returns:
            list[str]: A list of file names.
        """
        return list(self.config.keys())

    @staticmethod
    def from_json(json_file_id: str) -> 'Downloader':
        """
        Creates a Downloader object from a JSON file containing file names and Google Drive IDs.

        Args:
            json_file_id (str): The file ID of the JSON file in Google Drive.

        Returns:
            Downloader: The created Downloader object.
        """
        url = f"https://drive.google.com/uc?id={json_file_id}"
        gdown.download(url, "data/config.json", quiet=False)
        with open("data/config.json", "r") as file:
            config = json.load(file)
            return Downloader(dict(config.items()))
