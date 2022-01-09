import folium as folium
from folium.plugins import FloatImage

LATITUDE_BOUND = 90
LONGITUDE_BOUND = 180
epsilon = 0.000001


def drawCellsOnWorldMap(ground_truth, cell_info_arr, alpha_levels):
    c = 0
    map_grid = folium.Map(
        location=[59.338315, 18.089960],
        tiles=None,
        zoom_start=1,
        max_bounds=True,
        min_zoom=1
    )
    folium.raster_layers.TileLayer(tiles="cartodbpositron", no_wrap=True).add_to(map_grid)
    curr = -1

    cell_intensities = []
    for i in range(len(alpha_levels[0])):

        cell_int = alpha_levels[0][i] * 5
        if cell_int > 0.8:
            cell_int = 0.8
        if cell_int < 0.05:
            cell_int = 0.
        cell_intensities.append(cell_int)

    for _, cell_info in cell_info_arr.iterrows():
        v0 = str(cell_info["v0"]).split(",")
        v1 = str(cell_info["v1"]).split(",")
        v2 = str(cell_info["v2"]).split(",")
        v3 = str(cell_info["v3"]).split(",")
        if float(v3[1]) < -LONGITUDE_BOUND + epsilon or float(v2[1]) < -LONGITUDE_BOUND + epsilon:  # border correction
            v3[1] = -1.0 * float(v3[1])
            v2[1] = -1.0 * float(v2[1])

        color = "red"
        folium.Polygon(locations=[(float(v0[0]), float(v0[1])),
                                  (float(v1[0]), float(v1[1])),
                                  (float(v2[0]), float(v2[1])),
                                  (float(v3[0]), float(v3[1]))],
                       fill=True,
                       color=color,
                       weight=0,
                       tooltip="Confidence: " + str(round(alpha_levels[0][cell_info["class_label"]] * 100, 1)) + " %",
                       fill_opacity=str(cell_intensities[cell_info["class_label"]])
                       ).add_to(map_grid)

    FloatImage("legend.jpg", bottom=90, left=75).add_to(map_grid)
    color = "blue"
    folium.CircleMarker(location=[ground_truth["latitude_mean"], ground_truth["longitude_mean"]],
                        clustered_marker=True, color=color,
                        radius=0.01).add_to(map_grid)
    style = "<style>img{ width: 100px;}</style>"
    map_grid.get_root().html.add_child(folium.Element(style))

    return map_grid
