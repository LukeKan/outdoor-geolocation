import concurrent.futures
import csv
import os
import time

import requests
from flickrapi import FlickrAPI
import pandas as pd
import sys

key = '2b12562559c7f1b3790201553d9c970b'
secret = '460c6648f5ee8275'


def get_urls(image_tag, MAX_COUNT):
    flickr = FlickrAPI(key, secret)
    count = 0
    completed = False
    page = 1
    elems = []
    while count < MAX_COUNT and completed is False:
        photos = flickr.photos.search(text=image_tag,
                                      tags=image_tag,
                                      extras='url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o, geo',
                                      sort='relevance',
                                      content_type=1,
                                      has_geo=1,
                                      page=page)
        photos = photos.find('photos')
        if page == photos.get('pages'):
            completed = True
        page += 1

        photos = photos.findall('photo')
        for photo in photos:
            if count < MAX_COUNT:
                count += 1
                print("Fetching url for image number {}".format(count))
                try:
                    url = photo.get('url_m')
                    lat = photo.get('latitude')
                    long = photo.get('longitude')
                    if url != None:
                        elems.append(
                            {'url': url,
                             'lat': lat,
                             'long': long,
                             'tag': image_tag}
                        )
                except:
                    print("Url for image number {} could not be fetched".format(count))
            else:
                print("Done fetching urls, fetched {} urls out of {}".format(len(elems), MAX_COUNT))
                break

    elems = pd.DataFrame(elems)
    elems.drop_duplicates()
    print("Writing out the urls in the current directory")
    elems.to_csv("train_flickr_diff.csv", mode='a', header=None, index=False)
    print("Done!!!")


def put_images(FILE_NAME):
    urls = []
    with open(FILE_NAME, newline="") as csvfile:
        doc = csv.reader(csvfile, delimiter=",")
        for row in doc:
            if row[1].startswith("https"):
                urls.append(row[1])
    if not os.path.isdir(os.path.join(os.getcwd(), FILE_NAME.split("_")[0])):
        os.mkdir(FILE_NAME.split("_")[0])
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for url in enumerate(urls):
            executor.submit(download_from_url, url, FILE_NAME)
    t1 = time.time()
    print("Done with download, job took {} seconds".format(t1 - t0))


def download_from_url(url, FILE_NAME):
    try:
        resp = requests.get(url[1], stream=True)
        path_to_write = os.path.join(os.getcwd(), FILE_NAME.split("_")[0], url[1].split("/")[-1])
        outfile = open(path_to_write, 'wb')
        outfile.write(resp.content)
        outfile.close()
        print("Done downloading {} ".format(url[0] + 1))
    except:
        print("Failed to download url number {}".format(url[0]))


def main():
    data_crawler = [
        {
            'tag': 'monument',
            'MAX_COUNT': 10000
        },
        {
            'tag': 'thoroughfare',
            'MAX_COUNT': 10000
        }
    ]
    for elem in data_crawler:
        get_urls(elem['tag'], elem['MAX_COUNT'])
    #put_images("train_flickr_diff.csv")
    #merge_csvs("train_flickr.csv", "train_flickr_diff.csv")


def merge_csvs(main_path, diff_path):
    main = pd.read_csv(main_path)
    diff = pd.read_csv(diff_path)
    main = pd.concat(main, diff)
    main.to_csv(main_path)

if __name__ == '__main__':
    main()
