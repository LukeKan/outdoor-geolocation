import folium as folium
import pandas as pd


def main():
    map1 = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )
    df = pd.read_csv("train_flickr.csv")
    df.apply(lambda row: folium.CircleMarker(location=[row["lat"], row["long"]], clustered_marker=True, radius=0.1).add_to(map1), axis=1)
    map1.save("map.html")

if __name__ == '__main__':
    main()