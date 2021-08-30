import folium as folium
import io
from PIL import Image


def drawCellsOnWorldMap(cell_info_arr):
    c = 0
    map_grid = folium.Map(
        location=[59.338315, 18.089960],
        tiles='cartodbpositron',
        zoom_start=1,
    )
    curr = -1
    for cell_info in cell_info_arr:
        v0 = str(cell_info["v0"]).split(",")
        v1 = str(cell_info["v1"]).split(",")
        v2 = str(cell_info["v2"]).split(",")
        v3 = str(cell_info["v3"]).split(",")


        if c == 0:
            color = "green"
            c += 1
            curr = cell_info["class_label"]
        else:
            if curr == cell_info["class_label"]:
                color = "blue"
            else:
                color = "red"
        popup = """
                id : <b>%s</b><br>
                img per cell : <b>%s</b><br>
                level : <b>%s</b><br>
                """ % (cell_info["class_label"], cell_info["imgs_per_cell"], cell_info["level"])
        folium.Polygon(locations=[(float(v0[0]), float(v0[1])),
                              (float(v1[0]), float(v1[1])),
                              (float(v2[0]), float(v2[1])),
                              (float(v3[0]), float(v3[1]))],
                   fill=True,
                   color=color,
                   tooltip=popup
                   ).add_to(map_grid)
        folium.CircleMarker(location=[cell_info["latitude_mean"], cell_info["longitude_mean"]], clustered_marker=True,
                                            radius=0.01).add_to(map_grid)
    return map_grid
