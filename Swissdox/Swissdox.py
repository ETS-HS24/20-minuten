import logging
import os
import sys
import yaml
import json
import shutil
import requests
import lzma

from dotenv import load_dotenv
from pprint import pprint
from pathlib import Path
from jsonschema import validate, ValidationError
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
        self.SCHEMA_CONFIG_PATH = os.path.normpath(os.path.join(Path(__file__).parent.resolve(), "schemas/available-schemas.yaml"))

    def initialize_dataset(self, data_root_folder, validate_query_file=True, query_string=None):
        if data_root_folder:
            self.full_path = os.path.normpath(os.path.join(data_root_folder, self.dataset_name))
            existing_dataset_query_file_path = os.path.join(self.full_path, self.dataset_name+".yaml")

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
        logging.info("Creating query file from string.")
        with open(os.path.join(data_root_folder,self.dataset_name , self.dataset_name+".yaml"), "w") as query_file:
            query_file.write(query_string)

    
    def validate_query(self, query, schema):
        try:
            validate(query, schema=json.loads(schema))
            logging.info("Query is valid.")
        except ValidationError as ve:
            logging.error("Query is invalid")
            logging.exception("Your YAML query does not fit the schema...")
            return False
    
        return True
    
    def load_query_file(self, query_file_path):
        assert Path(query_file_path).is_file(), f"Query file {query_file_path} does not exist."
        with open(query_file_path, "r") as query_file:
            query = yaml.safe_load(query_file)
        
        return query

    def load_schema_file(self, schema_version=None):
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
        
    # TODO: Get rid of coupling
    def initialize_from_api(self, api_client, data_root_folder):
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
        return {
            "X-API-Key":    self.api_key,
            "X-API-Secret": self.api_secret
        }

    def update_status(self):
        assert self.headers, "You have not loaded the API keys yet."
        result = requests.get(
            f"{self.API_BASE_URL}/status",
            headers=self.headers
        )

        assert result.ok, f"The API respons was not OK: {result.json()}"
        status = result.json()
        logging.debug(f"Retrieved {len(status)} queries.")
        self.status = status
        
        #self.datasets = [SwissdoxDataset(status_on_api=stat) for stat in status]
        self.datasets = {stat["name"]:stat for stat in status}

    def get_status(self, existing_dataset=None):
        self.update_status()

        if not existing_dataset:
            return self.datasets
        else:
            assert existing_dataset.dataset_name, "Please provide a valid Swissdox Dataset Object."
            return self.datasets[existing_dataset.dataset_name]
        
    def download_dataset(self, existing_dataset, overwrite=False):
        dataset_config = self.get_status(existing_dataset)

        assert existing_dataset.is_initialized, "Please initialize the dataset first."
        assert existing_dataset.full_path, "Please initilaize the dataset with a path. This is required to download the files."

        files_in_raw_folder = os.listdir(os.path.join(existing_dataset.full_path, "raw-data"))
        assert not ([any(".tsv" in file) for file in files_in_raw_folder] and overwrite), "Set flag to overwrite existing .tsv files"

        self.download(url=dataset_config["downloadUrl"],
                      save_directory=os.path.join(existing_dataset.full_path, "raw-data"),
                      decompress=True)

    # TODO: Make this more pretty
    def download(self, url, save_directory, decompress=True):

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
            
    
def test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG) 

    def excepthook(*args):
        logging.getLogger().error('Uncaught exception:', exc_info=args)

    sys.excepthook = excepthook

    #sd.load_query_file("2024-09-13 15:29 Test-Load-inspect dataset copy.yaml")
    #print(sd.get_download_url_from_query_name("2024-09-13 15:29 Test-Load-inspect dataset"))
    #sd.get_status()
    sd = SwissdoxAPI()
    sd.set_api_credentials()
    sd.get_status()

    ds = SwissdoxDataset(dataset_name="2024-09-13 15:29 Test-Load-inspect dataset.yaml")
    ds.initialize_from_api(sd, "../data")

    sd.download_dataset(ds, overwrite=False)
    #sd.get_status()
    #ds.initialize_from_api(sd, "../data")
    #ds.initialize_dataset("../data")

    return sd

if __name__ == "__main__":
    sd = test()
    sd.update_status()
    print(sd.datasets)