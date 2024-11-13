# 20-minuten

# Swissdox@LiRI

In this project data of the Swissdox@LiRI database is used. LiRI cooperates with SMD (Schweizer Mediendatenbank AG) to
make the Swissdox database easily accessible to researchers. The Swissdox@LiRI database includes approximately 23
million published media articles from a wide range of Swiss media sources (both print and digital) covering many 
decades, and is updated daily with approximately 5000 to 6000 new articles.

## SwissdoxDataset

To use the SwissdoxAPI a SwissdoxDataset instance needs to be created first. The dataset name has to match with the name
specified on the https://swissdox.linguistik.uzh.ch/queries page. Next, after the SwissdoxAPI instance is created and 
the API credentials is all set up, the Swissdox dataset can be initialized from the API, specifying the SwissdoxAPI 
instance and the directory path where the data should be stored. 

When using the Swissdox data it is only allowed to use the data locally. It is NOT ALLOWED to store and share the 
articles on GitHub or somewhere else on the Internet. This is a restriction from Swissdox@LiRI.

## SwissdoxAPI

The SwissdoxAPI is responsible for setting up credentials, creating a connection to Swissdox, checking for connection,
updating status, and downloading of the desired dataset. The dataset name can be specified via SwissdoxDataset.

The API credentials can be set up with a given key and secret during instance creation or the API credentials can be 
loaded from a swissdox-creds.env file. The swissdox-creds.env file should be in the same directory as the Swissdox.py
file. A `SWISSDOX_KEY` variable for the key and a `SWISSDOX_SECRET` variable for the secret has to be set in the 
swissdox-creds.env file. The key and secret you can get from Swissdox@LiRI (https://swissdox.linguistik.uzh.ch/).


### Dataset usage example

```
from Swissdox.Swissdox import SwissdoxAPI, SwissdoxDataset

sd = SwissdoxAPI()
sd.set_api_credentials()
sd.get_status()

ds = SwissdoxDataset(dataset_name="20min-test-query-2020-jan-jun")
ds.initialize_from_api(sd, "../data")

sd.download_dataset(ds, overwrite=False)
```