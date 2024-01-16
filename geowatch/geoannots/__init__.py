"""
This submodule contains tools relating to:

    * the IARPA region and site model geojson formats.
    * geographic kwcoco extensions

At the time of creation we have many various tools scattered across the repo to
deal with these, but we should work to consolidate the ones pertaining strictly
to these specific types of geojsons into this submodule.


Location of existing tools that should likely be moved:

    * ../demo/metrics_demo/__init__.py :: various
    * ../cli/run_tracker.py :: various
    * ../cli/reproject_annotations.py :: various
    * ../cli/validate_annotation_schemas.py :: various
    * ../cli/extend_sc_sites.py :: various
    * ../cli/crop_sites_to_regions.py :: various
    * ../cli/cluster_sites.py :: various
    * ../utils/util_framework.py :: determine_region_id

    * ~/code/watch/dev/poc/make_region_from_sitemodel.py


Related tool that should NOT be moved are related to general geojson:

    * ../utils/util_gis.py :: coerce_geojson_datas
"""
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'geococo_objects',
        'geomodels',
    },
    submod_attrs={},
)

__all__ = ['geococo_objects', 'geomodels']
