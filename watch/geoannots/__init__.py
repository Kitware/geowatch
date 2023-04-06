"""
This submodule contains tools relating to the IARPA region and site model
geojson formats.

At the time of creation we have many various tools scattered across the repo to
deal with these, but we should work to consolidate the ones pertaining strictly
to these specific types of geojsons into this submodule.


Location of existing tools that should likely be moved:

    * ../demo/metrics_demo/__init__.py :: various
    * ../cli/kwcoco_to_geojson.py :: various
    * ../cli/reproject_annotations.py :: various
    * ../cli/validate_annotation_schemas.py :: various
    * ../cli/extend_sc_sites.py :: various
    * ../cli/crop_sites_to_regions.py :: various
    * ../cli/cluster_sites.py :: various
    * ../utils/util_framework.py :: determine_region_id


Related tool that should NOT be moved are related to general geojson:

    * ../utils/util_gis.py :: coerce_geojson_datas
"""
