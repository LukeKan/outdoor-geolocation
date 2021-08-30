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
PATH = "../dataset/crawl/flickr/maps/cells/"


def reverse_geotag_from_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=False)
    #locator = Nominatim(user_agent="outdoor", timeout=10)
    #rgeocoder = RateLimiter(locator.reverse,min_delay_seconds=0.00001)
    #df["geom"] = df["lat"].map(str) + ',' + df["long"].map(str)
    #tqdm.pandas()
    #df['address'] = df["geom"].progress_apply(rgeocoder)
    with tqdm(total=99) as pbar:
        for index, row in df.iterrows():
            pbar.update(1)
            #with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                #if str(row["country"]) != "nan":
                #    print("OK row {}".format(index))
                #    continue
            rev_geotag_row( index, row, df)
            """if index % 1000 == 0:
                df.to_csv(PATH + "train_flickr_info.csv", index=False)
                print("Writing checkpoint up to {}".format(index))"""
    #df.to_csv("../dataset/crawl/flickr/train_flickr_info.csv", index=False)
    fill_targets(df)


def rev_geotag_row(index, row, df):
    #time.sleep(1)#avoid ban
    lat = str(row["latitude_mean"])
    lng = str(row["longitude_mean"])
    url = API_URL + "lat=" + lat + "&lon=" + lng
    result = requests.get(url)
    result.encoding = result.apparent_encoding
    result = result.json()
    try:
        address = result["address"]
        if "country_code" in address.keys():
            df.loc[index, "country_code"] = address["country_code"]

            # print(address["country"])
        else:
            df.loc[index, "country_code"] = "None"
    except:
        df.loc[index, "country_code"] = "None"
    #if result["continent"] is not None:
    #    df.loc[index, "continent"] = result["continent"]
    #else:
    #    df.loc[index, "continent"] = "None"

    """if address["city"] is not None:
        df.loc[index, "city"] = address["city"]
    else:
        df.loc[index, "city"] = "None"""
    #print("Done {}.".format(index))


def fill_targets(df):
    dest_df = pd.read_csv("../dataset/crawl/flickr/train_flickr_cells.csv", index_col=False)
    with tqdm(total=dest_df.shape[0]) as pbar:
        for _,row in dest_df.iterrows():
            pbar.update(1)
            debug = df.loc[df['class_label'] == row["lvl_2"]].country_code
            dest_df.loc[_,"country_code"] = debug.iloc[0]
    dest_df.to_csv("../dataset/crawl/flickr/train_flickr_info.csv", index=False)

if __name__ == '__main__':
    reverse_geotag_from_csv(PATH + "cells_500_10000_images_306676_lvl_2.csv")
