#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class SiteModelAssociatorCLI(scfg.DataConfig):
    """
    For each site in sites1 find which site in site2 overlaps, and transfer its
    metadata over to it and write it to a new path.
    """
    sites1 = scfg.Value(None, help='directory containing original site models')
    sites2 = scfg.Value(None, help='directory containing site models with special UUIDs')
    out_dpath = scfg.Value(None, help='directory to write new site models')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> import sys, ubelt
            >>> cls = SiteModelAssociatorCLI
            >>> argv = 0
            >>> from site_model_associator import *  # NOQA
            >>> from geowatch.geoannots.geomodels import RegionModel
            >>> region1, sites1 = RegionModel.random(with_sites=True)
            >>> region2, sites2 = RegionModel.random(with_sites=True, region_poly=region1.geometry)
            >>> dpath = ub.Path.appdir('geowatch/demo/site_association').ensuredir()
            >>> dpath1 = (dpath / 'sites1').delete().ensuredir()
            >>> dpath2 = (dpath / 'sites2').delete().ensuredir()
            >>> out_dpath = (dpath / 'out_dpath').delete().ensuredir()
            >>> for site in sites1:
            >>>     fpath = dpath1 / (site.site_id + '.geojson')
            >>>     fpath.write_text(site.dumps())
            >>> for site in sites2:
            >>>     fpath = dpath2 / (site.site_id + '.geojson')
            >>>     fpath.write_text(site.dumps())
            >>> kwargs = {
            >>>     'sites1': dpath1,
            >>>     'sites2': dpath2,
            >>>     'out_dpath': out_dpath,
            >>> }
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        from geowatch.geoannots.geomodels import SiteModel
        from geowatch.geoannots.geomodels import SiteModelCollection
        sites1 = SiteModelCollection(SiteModel.coerce_multiple(config.sites1))
        sites2 = SiteModelCollection(SiteModel.coerce_multiple(config.sites2))

        region1 = sites1.as_region_model()
        region2 = sites2.as_region_model()

        region1_df = region1.pandas_summaries()
        region2_df = region2.pandas_summaries()

        from kwgis.utils import util_gis
        overlaps = util_gis.geopandas_pairwise_overlaps(region1_df, region2_df)

        missing = []
        ambiguous = []
        for idx1, idxs2 in overlaps.items():
            if len(idxs2) == 0:
                missing.append((idx1, idxs2))
            elif len(idxs2) > 1:
                ambiguous.append((idx1, idxs2))
            else:
                # Found the association, update site1 with props from site2
                idx2 = idxs2[0]
                site1 = sites1[idx1]
                site2 = sites2[idx2]
                # hack for single case
                cache1 = site1.header['properties'].get('cache', {})
                cache2 = site2.header['properties'].get('cache', {})
                cache1['smqtk_uuid'] = cache2.get('smqtk_uuid', None)

        out_dpath = ub.Path(config.out_dpath)
        out_dpath.ensuredir()

        for site in sites1:
            fpath = out_dpath / (site.site_id + '.geojson')
            fpath.write_text(site.dumps())


__cli__ = SiteModelAssociatorCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/oneoffs/site_model_associator.py
        python -m site_model_associator
    """
    __cli__.main()
