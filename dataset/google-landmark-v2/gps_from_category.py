import csv
import json

import geopy

import requests


def gps_from_category(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        results = []
        for row in csv_reader:

            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
                results.append(['landmark_id', 'category', 'lat', 'long'])
            else:
                print(f'\tid:{row[0]} \turl:{row[1]}')
                f = requests.get(row[1])
                html_file = f.text
                lat_position = html_file.find('data-lat')
                if lat_position == -1:
                    query = row[1].split(":")[2].split("(")[0].replace("_", "%20")
                    g = requests.get('https://nominatim.openstreetmap.org/search?q='+query+'&format=json&polygon_geojson=1&addressdetails=1')
                    if g.text == '[]':
                        query_struct = query.split("%20")
                        query = query_struct[0]

                        g = requests.get(
                            'https://nominatim.openstreetmap.org/search?q=' + query + '&format=json&polygon_geojson=1&addressdetails=1')

                    json_elem = json.loads(g.text)
                    if len(json_elem) > 0:
                        latitude = json_elem[0]["lat"]
                        longitude = json_elem[0]["lon"]
                    else:
                        latitude = -100.0
                        longitude = -100.0
                else:
                    long_position = html_file.find('data-lon')
                    latitude = html_file[lat_position + 10:lat_position + 20].split('"')[0]
                    longitude = html_file[long_position + 10:long_position + 20].split('"')[0]
                results.append([row[0], row[1], latitude, longitude])
                print(f'\tlat:{latitude} \tlong:{longitude}')




        print(f'Processed {line_count} lines.')

    with open('data/labels_coord.csv', mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for r in results:
            writer.writerow(r)

if __name__ == '__main__':
    gps_from_category("data/train_label_to_category.csv")