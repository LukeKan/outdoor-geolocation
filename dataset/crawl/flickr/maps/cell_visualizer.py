import folium as folium
from folium import plugins
import pandas as pd
LEVELS = 2

def generate_map(level):
    map_grid = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )
    cells = pd.read_csv("cells/cells_3000_30000_images_970565_0_30_bb.csv", error_bad_lines=False)
    levels = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
               'pink', 'lightblue', 'lightgreen', 'gray', 'black']
    for k, v in cells.iterrows():
        v0 = str(v["v0"]).split(",")
        v1 = str(v["v1"]).split(",")
        v2 = str(v["v2"]).split(",")
        v3 = str(v["v3"]).split(",")
        try:
            color = levels[v["level"] - 2]
        except:
            color = "red"
        popup = """
                id : <b>%s</b><br>
                img per cell : <b>%s</b><br>
                level : <b>%s</b><br>
                """ % (v["class_label"], v["imgs_per_cell"], v["level"])
        folium.Polygon(locations=[(float(v0[0]), float(v0[1])),
                                  (float(v1[0]), float(v1[1])),
                                  (float(v2[0]), float(v2[1])),
                                  (float(v3[0]), float(v3[1]))],
                       fill=True,
                       color=color,
                       tooltip=popup
                       ).add_to(map_grid)
    cells.apply(
        lambda row: folium.CircleMarker(location=[row["latitude_mean"], row["longitude_mean"]], clustered_marker=True,
                                        radius=0.01).add_to(map_grid), axis=1)
    return map_grid


def main():
    for i in range(2, LEVELS+1):
        map_grid = generate_map(i)
        map_grid.save("www/mapGrid_full_3k_30k.html")


if __name__ == '__main__':
    main()
