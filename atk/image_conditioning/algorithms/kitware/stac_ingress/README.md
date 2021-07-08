# STAC Ingestion
Data ingestion script converting a STAC catalog to data on disk.
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
aoi_bounds|Spatial bounds of the Area of Interest (AOI), in WGS84 (EPSG4326).|Yes
date_range|Only retrieve images with timestamps within the specified start and end date.|Yes
stac_api_url|URL for the STAC API endpoint|Yes
output_dir|Directory where output images / data will be written.  Will be created if it doesn't already exist.|Yes
collections|List of STAC collections to retrieve results from|Yes
stac_api_key|API Key for accessing the STAC endpoint|
dry_run|Don't actually download the retrieved imagery|
max_results|Maximum number of results to ingress.|
max_cloud_cover|Maximum amount of cloud cover allowed before a STAC item is filtered out|
min_aoi_overlap|Minimum amount of overlap an image must have with the selected AOI to be kept|

## Outputs:
Name|Description
---|---
stac_catalog|STAC Catalog JSON string
output_dir|Directory where images / data have been written.
