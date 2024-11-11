import json
import logging
import lzma
import os
import shutil
from pathlib import Path
import requests
import yaml
from dotenv import load_dotenv
from jsonschema import validate, ValidationError


class SwissdoxDataset:
    def __init__(self,
                 dataset_name,
                 query_file_path=None,
                 status_on_api=None,
                 default_folders=None
                 ):

        self.dataset_name = dataset_name

        if query_file_path:
            self.query_file_path = os.path.normpath(query_file_path)
        else:
            self.query_file_path = None

        self.is_initialized = False
        self.full_path = None

        self.status_on_api = status_on_api

        if not default_folders:
            self.default_folders = ["raw-data", "intermediate-data"]

        self.DEFAULT_SCHEMA_VERSION = "1.2"
        self.SCHEMA_CONFIG_PATH = os.path.normpath(
            os.path.join(Path(__file__).parent.resolve(), "schemas/available-schemas.yaml"))

    def initialize_dataset(self, data_root_folder, validate_query_file=True, query_string=None):
        """
        Initialize dataset from the given data root directory.
        :param data_root_folder:
        :param validate_query_file:
        :param query_string:
        :return:
        """
        if data_root_folder:
            self.full_path = os.path.normpath(os.path.join(data_root_folder, self.dataset_name))
            existing_dataset_query_file_path = os.path.join(self.full_path, self.dataset_name + ".yaml")

            # Check if dataset and query file exists 
            if Path(self.full_path).is_dir():
                if Path(existing_dataset_query_file_path).is_file():
                    self.query_file_path = existing_dataset_query_file_path
                # If dataset exists but no query file we check if we can copy it.
                elif self.query_file_path:
                    logging.info(f"No local query file found.")
                    shutil.copy2(src=self.query_file_path, dst=existing_dataset_query_file_path)
                    logging.info(f"Query file from {self.query_file_path} imported to dataset.")
                else:
                    self.create_query_file_from_string(data_root_folder, query_string)

            else:
                logging.info(f"Creating a new dataset {self.dataset_name} in {data_root_folder}.")
                Path(self.full_path).mkdir(parents=True, exist_ok=True)
                for folder in self.default_folders:
                    Path(os.path.join(self.full_path, folder)).mkdir(parents=True, exist_ok=True)

                if self.query_file_path:
                    logging.info(f"No local query file found.")
                    shutil.copy2(src=self.query_file_path, dst=existing_dataset_query_file_path)
                    logging.info(f"Query file from {self.query_file_path} imported to dataset.")
                elif query_string is not None:
                    self.create_query_file_from_string(data_root_folder, query_string)

            if self.query_file_path and validate_query_file:
                self.validate_query(
                    self.load_query_file(self.query_file_path),
                    self.load_schema_file()
                )

            self.is_initialized = True

        else:
            logging.info(f"No data root folder was provided so no existing dataset is expected.")

    def create_query_file_from_string(self, data_root_folder, query_string):
        """
        Create query file from string.
        :param data_root_folder
        :param query_string
        :return:
        """
        logging.info("Creating query file from string.")
        with open(os.path.join(data_root_folder, self.dataset_name, self.dataset_name + ".yaml"), "w") as query_file:
            query_file.write(query_string)

    def validate_query(self, query, schema):
        """
        Validate query against schema.
        :param query:
        :param schema:
        :return:
        """
        try:
            validate(query, schema=json.loads(schema))
            logging.info("Query is valid.")
        except ValidationError as ve:
            logging.error("Query is invalid")
            logging.exception("Your YAML query does not fit the schema...")
            return False

        return True

    def load_query_file(self, query_file_path):
        """
        Load and return query from query file path. Show asset message if file path does not exist.
        :param query_file_path:
        :return: query
        """
        assert Path(query_file_path).is_file(), f"Query file {query_file_path} does not exist."
        with open(query_file_path, "r") as query_file:
            query = yaml.safe_load(query_file)

        return query

    def load_schema_file(self, schema_version=None):
        """
        Load schema file.
        :param schema_version:
        :return:
        """
        with open(self.SCHEMA_CONFIG_PATH, "r") as schema_config:
            schema_paths = yaml.safe_load(schema_config.read())

        if not schema_version:
            self.schema_version = self.DEFAULT_SCHEMA_VERSION
        else:
            self.schema_version = schema_version

        assert self.schema_version in schema_paths

        schema_path = schema_paths[self.schema_version]
        if not os.path.isabs(schema_path):
            schema_path = os.path.join(Path(__file__).parent.resolve(), schema_path)

        return Path(schema_path).read_text()

    def initialize_from_api(self, api_client, data_root_folder):
        """
        Initialize dataset from api client and store it in the data root folder.
        :param api_client:
        :param data_root_folder:
        :return:
        """
        assert api_client.datasets, f"Datasets are not loaded in the client. Do that first."
        assert self.dataset_name in api_client.datasets, f"Could not find {api_client.datasets} in your account. Alternatives: {api_client.datasets.keys()}"

        query_string = api_client.datasets[self.dataset_name]["yaml"]
        self.initialize_dataset(data_root_folder, query_string=query_string)


class SwissdoxAPI:
    def __init__(self):
        self.api_key = None
        self.api_secret = None
        self.headers = None

        self.API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
        self.DEFAULT_CREDENTIAL_FILE = "./swissdox-creds.env"

    def set_api_credentials(self, key=None, secret=None):
        """
        Set API credentials with given key and secret or loading API credentials from swissdox-creds.env file.
        The swissdox-creds.env file should be in the same directory as this file. A `SWISSDOX_KEY` variable for the key
        and a `SWISSDOX_SECRET` variable for the secret has to be set.
        :param key:
        :param secret:
        :return:
        """
        if (not key and not secret and Path(self.DEFAULT_CREDENTIAL_FILE).exists()):
            logging.info(f"Loading API credentials from {self.DEFAULT_CREDENTIAL_FILE}")
            load_dotenv(self.DEFAULT_CREDENTIAL_FILE)
            self.api_key = os.getenv("SWISSDOX_KEY")
            self.api_secret = os.getenv("SWISSDOX_SECRET")
            self.headers = self.create_http_header()
            return

        assert key and secret, "Please provide an API key and secret."

        self.api_key = key
        self.api_secret = secret
        self.headers = self.create_http_header()

    def create_http_header(self):
        """
        Create and return HTTP headers. This is needed to access the data from the Swissdox@LiRI API.
        :return:
        """
        return {
            "X-API-Key": self.api_key,
            "X-API-Secret": self.api_secret
        }

    def update_status(self):
        """
        Update the status by doing a GET request to the status endpoint.
        :return:
        """
        assert self.headers, "You have not loaded the API keys yet."
        result = requests.get(
            f"{self.API_BASE_URL}/status",
            headers=self.headers
        )

        assert result.ok, f"The API response was not OK: {result.json()}"
        status = result.json()
        logging.debug(f"Retrieved {len(status)} queries.")
        self.status = status
        self.datasets = {stat["name"]: stat for stat in status}

    def get_status(self, existing_dataset=None):
        """
        Get status if a given dataset already exists or not.
        :param existing_dataset:
        :return:
        """
        self.update_status()

        if not existing_dataset:
            return self.datasets
        else:
            assert existing_dataset.dataset_name, "Please provide a valid Swissdox Dataset Object."
            return self.datasets[existing_dataset.dataset_name]

    def download_dataset(self, existing_dataset, overwrite=False):
        """
        Download dataset. You can specify if you prefer to overwrite the existing dataset.
        :param existing_dataset:
        :param overwrite:
        :return:
        """
        dataset_config = self.get_status(existing_dataset)

        assert existing_dataset.is_initialized, "Please initialize the dataset first."
        assert existing_dataset.full_path, "Please initilaize the dataset with a path. This is required to download the files."

        files_in_raw_folder = os.listdir(os.path.join(existing_dataset.full_path, "raw-data"))
        assert not ([any(".tsv" in file) for file in
                     files_in_raw_folder] and overwrite), "Set flag to overwrite existing .tsv files"

        self.download(url=dataset_config["downloadUrl"],
                      save_directory=os.path.join(existing_dataset.full_path, "raw-data"),
                      decompress=True)

    def download(self, url, save_directory, decompress=True):
        """
        Download from URL and store the decompressed data if possible in given directory if it exists otherwise create
        specified directory first.
        :param url:
        :param save_directory:
        :param decompress:
        :return:
        """

        file_name = url.split("/")[-1]
        assert file_name, "URL does not point to a file."

        save_path = os.path.join(save_directory, file_name)

        Path(save_directory).mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading dataset to {save_path}")
        with requests.get(url, headers=self.headers, stream=True) as requests_stream:
            with open(save_path, "wb") as file_handler:
                shutil.copyfileobj(requests_stream.raw, file_handler)
                logging.info(f"Finished download.")

        # Decompression gate
        if not decompress:
            logging.info(f"Downloaded file to {save_path}")
            return save_path

        if file_name.endswith("xz"):
            logging.info("Decompressing xz file.")
            with lzma.open(save_path, "rb") as file_src:
                decompressed_path = save_path[:-3]
                with open(decompressed_path, "wb") as file_dst:
                    shutil.copyfileobj(file_src, file_dst)

            os.remove(save_path)
            logging.info(f"Decompressed file to {decompressed_path}")
        else:
            logging.warning(f"Decompression for filetype not implemented. Could not decompress.")
            logging.info(f"Downloaded file to {save_path}")


if __name__ == "__main__":
    pass
