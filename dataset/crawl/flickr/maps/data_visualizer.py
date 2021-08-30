import folium as folium
from folium import plugins
import pandas as pd


def main():

    map_images = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )
    df = pd.read_csv("../test.csv")
    df.apply(lambda row: folium.CircleMarker(location=[row["lat"], row["long"]], clustered_marker=True, radius=0.01).add_to(map_images), axis=1)

    map_images.save("www/mapImages.html")


if __name__ == '__main__':
    main()
