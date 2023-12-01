from kwcoco.util.dict_proxy2 import AliasedDictProxy
import ubelt as ub


class CocoGeoVideo(AliasedDictProxy, ub.NiceRepr):
    """
    # Note: hard coded while CocoVideo under development in kwcoco
    TODO: general coco scalars

    Example:
        >>> from geowatch.geoannots.geococo_objects import *  # NOQA
        >>> import geowatch
        >>> coco_dset = geowatch.coerce_kwcoco('geowatch-msi', heatmap=True, geodata=True, dates=True)
        >>> video = coco_dset.videos().objs[0]
        >>> self = CocoGeoVideo(video, coco_dset)
    """
    __alias_to_primary__ = {}

    def __init__(self, video, dset=None):
        self._proxy = video
        self.dset = dset

    def __nice__(self):
        from kwcoco.util.util_truncate import smart_truncate
        from functools import partial
        stats = ub.udict(self._proxy)
        stats = stats.map_values(str)
        stats = stats.map_values(
            partial(smart_truncate, max_length=32, trunc_loc=0.5))
        return ub.urepr(stats, compact=1, nl=0)

    @ub.memoize_property
    def warp_vid_from_wld(self):
        import kwimage
        return kwimage.Affine.coerce(self._proxy['warp_wld_to_vid'])

    @ub.memoize_property
    def warp_wld_from_vid(self):
        return self.warp_vid_from_wld.inv()

    @property
    def images(self):
        return self.dset.images(video_id=self['id'])

    def corners(self, space='video'):
        import kwimage
        if space == 'video':
            dsize = (self['width'], self['height'])
            vid_box = kwimage.Box.from_dsize(dsize)
            vid_poly = vid_box.to_polygon()
            return vid_poly
        elif space == 'wld':
            vid_poly = self.corners(space='video')
            wld_poly = vid_poly.warp(self.warp_wld_from_vid)
            return wld_poly

    @property
    def wld_crs(self):
        from geowatch.utils import util_gis
        wld_crs = util_gis.coerce_crs(self._proxy['wld_crs_info'])
        return wld_crs

    @property
    def wld_corners_gdf(self):
        import geopandas as gpd
        wld_crs = self.wld_crs
        wld_poly = self.corners(space='wld').to_shapely()
        subdict = ub.udict(self._proxy) & {'id', 'name'}
        gdf = gpd.GeoDataFrame([
            subdict | {'geometry': wld_poly},
        ], crs=wld_crs)
        return gdf
