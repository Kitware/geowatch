import kwimage


class Foo:
    def __init__(self):
        self.data = kwimage.Polygon.random()


class Bar:
    def __init__(self):
        self.data = kwimage.Polygon.random()

    @property
    def __geo_interface__(self):
        return self.data.to_shapely().__geo_interface__

import geopandas as gpd


foo_arr = [Foo() for _ in range(3)]
bar_arr = [Bar() for _ in range(3)]

gdf = gpd.GeoDataFrame({'c': [Bar().__geo_interface__ for _ in range(3)], 'a': [1, 2, 3], 'foo': foo_arr, 'bar': bar_arr})
print(gdf)


renamed = gdf.rename({'bar': 'baz'}, axis=1)
print(renamed)


renamed['a'] = renamed['a'].map(lambda x: Bar())
# renamed['geometry'] = bar_arr
renamed['c'] = bar_arr
print(renamed)
