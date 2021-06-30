# RGD Ingress
Query the WATCH RGD instance and pull down images from within a specified date range and bounding box.
Version: 0.0.1
License: Other
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
aoi_bounds|Spatial bounds of the Area of Interest (AOI).|Yes
date_range|Pull down images from within this date range|Yes
username|Username for RGD account|Yes
password|Password for RGD account|Yes
output_dir|Directory where output images / data will be written. Will be created if it doesn't already exist. |Yes
dry_run|Don't actually download the retrieved imagery|

## Outputs:
Name|Description
---|---
output_dir|Directory where output images / data have been written.
stac_catalog|STAC Catalog JSON string
