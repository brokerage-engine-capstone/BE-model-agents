import requests
import pandas as pd
import json
import numpy as np
from pprint import pprint
import env

API = env.API_KEY
BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json?address="
addresses = pd.read_csv("50_TX_addresses.csv", header=None, skiprows=1)

addresses.shape

addresses.fillna("", inplace=True)

cleaned_addresses = (
    addresses[0]
    .str.cat(addresses[1], sep=",")
    .str.cat(addresses[2], sep=",")
    .values
)

list_of_addresses = cleaned_addresses
list_of_addresses

for address in list_of_addresses:
    if not isinstance(address, str):
        print(type(address))
        print(address)

# +
batches = np.array_split(list_of_addresses, 112)

len(batches)

# +
counter = 0
results = []

for batch in batches:
    for address in batch:
        print(counter)
        address = str(address).replace(" ", "+")
        url = BASE_URL + address + API
        print(url)
        try:
            results.append(requests.get(url).json())
        except:
            results.append({"error": "error fetching {}".format(url)})
            print(f"error with {url}")
        counter += 1
        with open("50_TX_cleaned_addresses.json", "w") as f:
            json.dump(results, f)
# -


len(results)

with open("50_TX_cleaned_addresses.json", "w") as f:
    json.dump(results, f)

with open("50_TX_cleaned_addresses.json", "r") as f:
    results = json.load(f)
len(results)

results

pprint(results[0])

indexes

# +
new_addresses = []
for error in indexes:
    address = list_of_addresses[error]
    address = str(address).replace("#", "%23")
    # print(error, address)
    new_addresses.append(address)

new_addresses

# +
new_results = []

for address in new_addresses:
    print(counter)
    url = BASE_URL + address + API
    # print(url)
    try:
        new_results.append(requests.get(url).json())
    except:
        new_results.append({"error": "error fetching {}".format(url)})
        print(f"error with {url}")
    counter += 1
    with open("50_TX_cleaned_addresses_p2.json", "w") as f:
        json.dump(new_results, f)
# -

len(new_results)

print(len(results))
print(len(list_of_addresses))
print(len(new_results))

results[15]

list_of_addresses[47]

# +
new_indexes = []
for i, result in enumerate(new_results):
    if result["status"] == "REQUEST_DENIED":
        # print(f'{i} result')
        new_indexes.append(i)

new_indexes

# +
non_errored_addresses = []
address_dictionary = {}
for i, result in enumerate(results):
    print(f"Parsing Record {i} of {len(results)}")
    address_dictionary = {
        # 'full_address': result['results'][0]['formatted_address'],
        "lat": result["results"][0]["geometry"]["location"]["lat"],
        "lng": result["results"][0]["geometry"]["location"]["lng"],
    }
    non_errored_addresses.append(address_dictionary)

print(len(non_errored_addresses))
# -

len(new_results)

# +
errored_addresses = []
address_dictionary = {}
for i, result in enumerate(new_results):
    print(f"Parsing Record {i} of {len(new_results)}")
    address_dictionary = {
        #'full_address': result['results'][0]['formatted_address'],
        "lat": result["results"][0]["geometry"]["location"]["lat"],
        "lng": result["results"][0]["geometry"]["location"]["lng"],
    }
    errored_addresses.append(address_dictionary)

print(len(errored_addresses))
# -

errored_addresses

df1 = pd.DataFrame(errored_addresses, index=indexes)
print(df1.shape)
df1.head()

with open("50_TX_cleaned_addresses.json") as f:
    big_file = json.load(f)
with open("50_TX_cleaned_addresses_p2.json") as f:
    small_file = json.load(f)

print(len(big_file))
print(indexes)

for i in indexes:
    print(big_file[i])

# +
for i, j in enumerate(indexes):
    big_file[j] = small_file[i]

#    prop[indexes] = small_file[i]
# -

for i in indexes:
    print(big_file[i])

# +
cleaned_addresses = []
address_dictionary = {}
bad_address = {}
for i, result in enumerate(big_file):
    print(f"Parsing Record {i} of {len(big_file)}")
    if result["status"] != "ZERO_RESULTS":
        address_dictionary = {
            "full_address": result["results"][0]["formatted_address"],
            "lat": result["results"][0]["geometry"]["location"]["lat"],
            "lng": result["results"][0]["geometry"]["location"]["lng"],
        }
        cleaned_addresses.append(address_dictionary)
    else:
        bad_address = {
            "full_address": "INVALID_ADDRESS",
            "lat": "INVALID_LAT",
            "lng": "INVALID_LNG",
        }
        cleaned_addresses.append(bad_address)

print(len(cleaned_addresses))
# -

print(len(big_file))

df = pd.DataFrame.from_dict(cleaned_addresses)
df.full_address.value_counts(dropna=False)

df.shape

split_addy = df.full_address.str.split(pat=",", expand=True)
split_addy[0].value_counts(dropna=False)

split_state = split_addy[2].str.split(pat=" ", expand=True)
split_state[1].value_counts()

lat_lng = df[["lat", "lng"]]
lat_lng.shape

lat_lng.to_csv("lat_lng.csv", index=None)

lldf = pd.read_csv("lat_lng.csv")
lldf.info()

df1 = pd.merge(
    left=split_addy, right=split_state, on=split_addy.index, right_index=False
)
df1 = df1.drop(columns=["key_0", 3, 4, "2_x", "0_y"])
df1 = df1.rename(
    columns={"0_x": "address", "1_x": "city", "1_y": "state", "2_y": "zip"}
)

df1 = pd.merge(left=df1, right=lat_lng, on=df1.index).drop(columns="key_0")

print(df1.shape)
df1.head()

df1.sample(10)

df1[df1.zip.isna()]
