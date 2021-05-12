# STAC Ingestion
Data ingestion script converting a STAC catalog to data on disk.
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
aoi-bounds|Spatial bounds of the Area of Interest (AOI).|Yes
date-range|Only retrieve images with timestamps within the specified start and end date.|Yes
stac-api-url|URL for the STAC API endpoint|Yes
output-dir|Directory where output images / data will be written.  Will be created if it doesn't already exist.|Yes
collections|List of STAC collections to retrieve results from|Yes
stac-api-key|API Key for accessing the STAC endpoint|
dry-run|Don't actually download the retrieved imagery|

## Outputs:
Name|Description
---|---
stac-catalog|STAC Catalog JSON string
output-dir|Directory where images / data have been written.
