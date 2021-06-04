import folium as folium
from folium import plugins
import pandas as pd


def main():
    map_grid = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )
    map_images = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )

    cells = pd.read_csv("cells_10_2000_images_318391.csv", error_bad_lines=False)
    levels = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    for k,v in cells.iterrows():
        v0 = str(v["v0"]).split(",")
        v1 = str(v["v1"]).split(",")
        v2 = str(v["v2"]).split(",")
        v3 = str(v["v3"]).split(",")
        try:
            color = levels[v["level"]]
        except:
            color="red"
        folium.Polygon(locations=[(float(v0[0]),float(v0[1])),
                                  (float(v1[0]),float(v1[1])),
                                  (float(v2[0]),float(v2[1])),
                                  (float(v3[0]),float(v3[1]))],
            fill=True,
            color=color,
            tooltip=k
        ).add_to(map_grid)
    cells.apply(lambda row: folium.CircleMarker(location=[row["latitude_mean"], row["longitude_mean"]], clustered_marker=True,
                                                 radius=0.01).add_to(map_grid), axis=1)

    df = pd.read_csv("train_flickr.csv")
    df.apply(lambda row: folium.CircleMarker(location=[row["lat"], row["long"]], clustered_marker=True, radius=0.01).add_to(map_images), axis=1)

    map_grid.save("Maps/mapGrid.html")
    map_images.save("Maps/mapImages.html")


if __name__ == '__main__':
    main()
