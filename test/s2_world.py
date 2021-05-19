from s2sphere import *


def test_world_coverage():
    region1 = LatLngRect(LatLng.from_degrees(-90, 0), LatLng.from_degrees(90, 180))
    r1 = RegionCoverer()
    r1.min_level, r1.max_level = (5, 30)
    cell_ids1 = r1.get_covering(region1)

    region2 = LatLngRect(LatLng.from_degrees(-90, 180), LatLng.from_degrees(90, 0))
    r2 = RegionCoverer()
    r2.min_level, r2.max_level = (5, 30)
    cell_ids2 = r2.get_covering(region2)

    all_cell_ids = set(cell_ids1) | set(cell_ids2)

    for i in all_cell_ids:
        print(i.to_lat_lng())


def alternative_world_coverage(min_lat, max_lat, min_lng, max_lng):
    point_nw = LatLng.from_degrees(max_lat, min_lng)
    point_se = LatLng.from_degrees(min_lat, max_lng)

    rc = RegionCoverer()
    rc.min_level = 8
    rc.max_level = 15
    rc.max_cells = 50000

    cellids = rc.get_covering(LatLngRect.from_point_pair(point_nw, point_se))
    for c in cellids:
        print(c.to_lat_lng())

if __name__ == '__main__':
    alternative_world_coverage(-90.0, 90.0, -180.0, 180.0)
