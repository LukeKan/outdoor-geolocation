import json
import time

import pandas as pd
import requests
import concurrent.futures
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import tqdm
from tqdm import tqdm

API_URL= "https://nominatim.openstreetmap.org/reverse?format=json&"
PATH = "/data"


def reverse_geotag_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=False)
    #locator = Nominatim(user_agent="outdoor", timeout=10)
    #rgeocoder = RateLimiter(locator.reverse,min_delay_seconds=0.00001)
    #df["geom"] = df["lat"].map(str) + ',' + df["long"].map(str)
    #tqdm.pandas()
    #df['address'] = df["geom"].progress_apply(rgeocoder)

    for index, row in df.iterrows():
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            #if str(row["country"]) != "nan":
            #    print("OK row {}".format(index))
            #    continue
            executor.submit(rev_geotag_row, index, row, df)
        if index % 1000 == 0:
            df.to_csv(PATH + "train_flickr_info.csv", index=False)
            print("Writing checkpoint up to {}".format(index))
    df.to_csv(PATH + "train_flickr_info.csv", index=False)


def rev_geotag_row(index, row, df):
    #time.sleep(1)#avoid ban
    lat = str(row["lat"])
    lng = str(row["long"])
    url = API_URL + "lat=" + lat + "&lon=" + lng
    result = requests.get(url)
    result.encoding = result.apparent_encoding
    result = result.json()
    address = result["address"]
    #if result["continent"] is not None:
    #    df.loc[index, "continent"] = result["continent"]
    #else:
    #    df.loc[index, "continent"] = "None"
    if address["country"] is not None:
        df.loc[index, "country"] = address["country"]
        print(address["country"])
    else:
        df.loc[index, "country"] = "None"
    if address["city"] is not None:
        df.loc[index, "city"] = address["city"]
    else:
        df.loc[index, "city"] = "None"
    print("Done {}.".format(index))


if __name__ == '__main__':
    reverse_geotag_from_csv(PATH + "train_flickr_clean.csv")