import json

import requests


class GPS_Extractor:
    def getGPSFromCategoryURL(self, category_url):
        f = requests.get(category_url)
        html_file = f.text
        lat_position = html_file.find('data-lat')
        if lat_position == -1:
            query = category_url.split(":")[2].split("(")[0].replace("_", "%20")
            g = requests.get(
                'https://nominatim.openstreetmap.org/search?q=' + query + '&format=json&polygon_geojson=1&addressdetails=1')
            if g.text == '[]':
                query_struct = query.split("%20")
                query = query_struct[0]

                g = requests.get(
                    'https://nominatim.openstreetmap.org/search?q=' + query + '&format=json&polygon_geojson=1&addressdetails=1')

            json_elem = json.loads(g.text)
            latitude = json_elem[0]["lat"]
            longitude = json_elem[0]["lon"]
        else:
            long_position = html_file.find('data-lon')
            latitude = html_file[lat_position + 10:lat_position + 20].split('"')[0]
            longitude = html_file[long_position + 10:long_position + 20].split('"')[0]
        return latitude, longitude


GPS_Extractor.getGPSFromCategoryURL = staticmethod(GPS_Extractor.getGPSFromCategoryURL)
