import requests
import pandas as pd
import json


# Step 1: Fetch data
url = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAAG@DF_NAAG_I?dimensionAtObservation=AllDimensions&format=jsondata"
response = requests.get(url)
data = response.json()

with open("sample.json", "w") as outfile:
    outfile.write(str(data))


with open("sample.json", "r") as openfile:

    # Reading from json file
    json_object = json.load(openfile)


print(json_object)
