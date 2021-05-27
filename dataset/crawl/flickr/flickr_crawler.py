import concurrent.futures
import csv
import os
import time

import requests
from PyDictionary import PyDictionary
from flickrapi import FlickrAPI
import pandas as pd
import sys

key = 'c0204735d662efa8c73e187de7fec31f'
secret = '11e1ce9d726f10fb'
data_crawler = ['city'
                , 'building'
                , 'countryside'
                , 'landscape'
                , 'mountain'
                , 'woods'
                , 'avenue'
                , 'highway'
                , 'street'
                , 'town'
                , 'monument'
                , 'glacier'
                , 'parkway'
                , 'lane'
                , 'roadway'
                , 'neighborhood'
                , 'thoroughfare'
                , 'motorway'
]
MAX_COUNT = 3000


def get_urls(image_tag, term, MAX_COUNT, csv_path, elems):
    print("Fetching urls for {}".format(term))
    flickr = FlickrAPI(key, secret)
    count = 0
    completed = False
    page = 1
    stop_counter = 0
    switch_date = 0
    end_date = 1590479033
    start_date = 1580114633
    DELTA_DATE = 10364400
    while count < MAX_COUNT and completed is False and stop_counter < 3:
        #time.sleep(2)
        try:
            photos = flickr.photos.search(text=term,
                                          tags=term,
                                          extras='url_sq, url_t, url_s, url_q, url_m, url_n, url_z, url_c, url_l, url_o, geo',
                                          sort='date-taken-asc',
                                          geo_context=2,
                                          has_geo=1,
                                          page=page,
                                          per_page=500
                                          #min_taken_date=start_date,
                                          #max_taken_date=end_date
                                          )

            photos = photos.find('photos')
            if page == photos.get('pages'):
                completed = True
            page += 1

            photos = photos.findall('photo')
            c = 0
            for photo in photos:
                if count < MAX_COUNT:
                    count += 1
                    print("Fetching url for image number {}".format(count))
                    try:
                        url = photo.get('url_m')
                        lat = photo.get('latitude')
                        long = photo.get('longitude')
                        #print(elems['url'])
                        if url is not None and url not in elems['url']:
                            try:
                                elems.loc[-1] = {'url': url,
                                     'lat': lat,
                                     'long': long,
                                     'tag': image_tag}
                                c += 1
                                elems.index +=1
                                elems.sort_index()
                                #print(url)
                            except Exception as e:
                                print(e)
                                count -= 1
                        else:
                            count -=1
                    except Exception as e:
                        print("Url for image number {} could not be fetched because {}".format(count, e))
                else:
                    print("Done fetching urls, fetched {} urls".format(len(elems)))
                    break
            if c == 0:
                stop_counter += 1
            else:
                stop_counter = 0

            if switch_date == 10:
                switch_date = 0
                stop_counter += 1
                page = 300
                end_date = start_date
                start_date -= DELTA_DATE
        except:
            print('Server ERROR')
    print("Stop fetching urls for query {}".format(term))



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
            if os.path.exists(os.path.join(os.getcwd(), FILE_NAME.split("_")[0], url[1].split("/")[-1])) or \
                    os.path.exists(os.path.join(os.getcwd(), "test", url[1].split("/")[-1])):
                continue
            executor.submit(download_from_url, url)
    t1 = time.time()
    print("Done with download, job took {} seconds".format(t1 - t0))


def download_from_url(url):
    try:
        resp = requests.get(url[1], stream=True)
        c = 0
        """
        while resp.status_code != 200 and c < 10:
            print("Failed to download")
            time.sleep(3)
            resp = requests.get(url[1], stream=True)
            c += 1
        """
        path_to_write = os.path.join(os.getcwd(), "test", url[1].split("/")[-1])
        outfile = open(path_to_write, 'wb')
        outfile.write(resp.content)
        outfile.close()
        print("Done downloading {} ".format(url[0] + 1))
    except Exception as e:
        print("Failed to download url number {} because {}".format(url[0], e))


def main():
    elems =pd.read_csv("train_flickr.csv", index_col=0)
    dictionary = PyDictionary()
    synonym_list = []
    for elem in data_crawler:
        get_urls(elem, elem, MAX_COUNT, 'train_flickr_test.csv', elems)
        for synonym in dictionary.synonym(elem):
            if synonym not in synonym_list:
                synonym_list.append(synonym)
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    executor.submit(get_urls, elem,synonym,MAX_COUNT,'train_flickr_test.csv',elems)
                    #get_urls(elem, synonym, MAX_COUNT, 'train_flickr_test.csv', elems)
    # elems = list({v['url']: v for v in elems}.values())
    elems = pd.DataFrame(elems)
    elems.drop_duplicates()
    print("Writing out the urls in the current directory")
    elems.to_csv('train_flickr_t.csv')
    print("Done!!!")
    #put_images("train_flickr_t.csv")
    # merge_csvs("train_flickr_diff_t.csv", "train_flickr_diff.csv")


def merge_csvs(main_path, diff_path):
    main = pd.read_csv(main_path, index_col=False)
    print(main)
    diff = pd.read_csv(diff_path, index_col=False)
    print(diff)
    main.append(diff, ignore_index=True)
    print(main)
    main.to_csv(main_path)


if __name__ == '__main__':
    main()
    # df = pd.read_csv("train_flickr.csv", index_col=1)
    # print(df.columns)
    # df.drop_duplicates(subset=['Unnamed: 0'])
    # df.drop_duplicates()

    # df.to_csv('test.csv')
