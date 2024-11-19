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
            >>> cls = SiteModelAssociatorCLI
            >>> argv = 0
            >>> from site_model_associator import *  # NOQA
            >>> from geowatch.geoannots.geomodels import RegionModel
            >>> region1, sites1 = RegionModel.random(with_sites=True)
            >>> region2, sites2 = RegionModel.random(with_sites=True, region_poly=region1.geometry, region_id=region1.region_id)
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

        Ignore:
            import sys, ubelt
            sys.path.append(ubelt.expandpath('~/code/geowatch/dev/oneoffs'))
            from site_model_associator import *  # NOQA
            kwargs = {}
            kwargs['sites1'] = [
                '/data/joncrall/dvc-repos/smart_phase3_data/annotations/drop8/site_models/NZ_R0*',
                '/data/joncrall/dvc-repos/smart_phase3_data/annotations/drop8/site_models/CH_R0*',
                '/data/joncrall/dvc-repos/smart_phase3_data/annotations/drop8/site_models/KR_R0*',
            ]
            kwargs['sites2'] = [
                '/home/joncrall/code/smqtk-repos/SMQTK-IQR/docs/tutorials/tutorial_003_geowatch_descriptors/workdir/processed/chips',
            ]
            kwargs['out_dpath'] = '/home/joncrall/code/smqtk-repos/SMQTK-IQR/docs/tutorials/tutorial_003_geowatch_descriptors/workdir/processed/fixed_sites'
            argv = 0
            cls = SiteModelAssociatorCLI
            config = cls(**kwargs)
            cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        out_dpath = ub.Path(config.out_dpath)
        out_dpath.ensuredir()

        import kwarray
        import numpy as np
        from geowatch.geoannots.geomodels import SiteModel
        from geowatch.geoannots.geomodels import SiteModelCollection
        from kwgis.utils import util_gis
        all_sites1 = SiteModelCollection(SiteModel.coerce_multiple(config.sites1))
        all_sites2 = SiteModelCollection(SiteModel.coerce_multiple(config.sites2))

        for site in all_sites2:
            site.fixup()

        region_to_sites1 = ub.group_items(all_sites1, lambda s: s.region_id)
        region_to_sites2 = ub.group_items(all_sites2, lambda s: s.region_id)

        regions1 = set(region_to_sites1)
        regions2 = set(region_to_sites2)

        region_relations = dict(
            missing1=regions1 - regions2,
            missing2=regions2 - regions1,
        )
        if any(region_relations.values()):
            print(f'region_relations = {ub.urepr(region_relations, nl=1)}')
            raise Exception('Regions do not agree')

        region_ids = list(region_to_sites1.keys())

        summaries = []
        for region_id in ub.ProgIter(region_ids, desc='compute assignments in regions'):
            # Consider all sites within each region
            sites1 = SiteModelCollection(region_to_sites1[region_id])
            sites2 = SiteModelCollection(region_to_sites2[region_id])

            # Convert to site summaries for quick spatial overlap checks
            region1 = sites1.as_region_model()
            region2 = sites2.as_region_model()
            region1_df = region1.pandas_summaries()
            region2_df = region2.pandas_summaries()
            overlaps = util_gis.geopandas_pairwise_overlaps(region1_df, region2_df)

            # To handle the case where more than one site have sptial overlap
            # we build an affinity matrix to determine the optimal matching.
            affinity = -np.ones((len(region1_df), len(region2_df)))

            # If there is any spatial overlap, compute a spatio-temporal score
            # between the two sites
            missing = []
            ambiguous = ub.ddict(list)
            for idx1, idxs2 in overlaps.items():
                site1 = sites1[idx1]
                if len(idxs2) == 0:
                    missing.append((idx1, idxs2))
                else:
                    for idx2 in idxs2:
                        site2 = sites2[idx2]
                        score = site_overlap_score(site1, site2)
                        if len(idxs2) > 1:
                            ambiguous[idx1].append((score, idx2))
                        affinity[idx1, idx2] = score

            # Compute the optimal assignment given our score function
            assignment, assignment_score = kwarray.maxvalue_assignment(affinity)

            # Compute some stats about how good the assignment is
            is_assigned1 = ub.boolmask([t[0] for t in assignment], len(region1_df))
            is_assigned2 = ub.boolmask([t[1] for t in assignment], len(region2_df))
            num_assigned1 = sum(is_assigned1)
            num_assigned2 = sum(is_assigned2)
            summary = {
                'region_id': region_id,
                'ratio1': num_assigned1 / len(is_assigned1),
                'ratio2': num_assigned2 / len(is_assigned2),
                'num_assigned1' : num_assigned1,
                'num_assigned2' : num_assigned2,
                'total1': len(is_assigned1),
                'total2': len(is_assigned2),
                'assignment_score': assignment_score,
            }
            summaries.append(summary)

            for idx1, idx2 in assignment:
                site1 = sites1[idx1]
                # Found the association, update site1 with props from site2
                site2 = sites2[idx2]
                # hack for single case
                cache1 = site1.header['properties'].get('cache', {})
                cache2 = site2.header['properties'].get('cache', {})
                cache1['smqtk_uuid'] = cache2.get('smqtk_uuid', None)

            # Write out the modified sites
            for site in sites1:
                fpath = out_dpath / (site.site_id + '.geojson')
                fpath.write_text(site.dumps())
        print(f'summaries = {ub.urepr(summaries, nl=1)}')


def site_overlap_score(site1, site2):
    space_isect = site1.geometry.intersection(site2.geometry)
    space_union = site1.geometry.union(site2.geometry)
    space_iou = space_isect.area / space_union.area
    time_iou = site_temporal_overlap(site1, site2)
    score = space_iou * time_iou
    return score


def site_temporal_overlap(site1, site2):
    import kwutil
    dummy_start = kwutil.datetime.coerce('2014-01-01')
    dummy_end = kwutil.datetime.coerce('2022-01-01')

    start1 = site1.start_date or dummy_start
    end1 = site1.end_date or dummy_end

    start2 = site2.start_date or dummy_start
    end2 = site2.end_date or dummy_end

    union_start = min(start1, start2)
    union_end = max(end1, end2)

    isect_start = max(start1, start2)
    isect_end = min(end1, end2)

    eps = 1.0
    isect_area = max((isect_end - isect_start).total_seconds(), 0)
    union_area = max((union_end - union_start).total_seconds(), eps)
    time_iou = isect_area / union_area
    return time_iou


__cli__ = SiteModelAssociatorCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/oneoffs/site_model_associator.py
        python -m site_model_associator
    """
    __cli__.main()
