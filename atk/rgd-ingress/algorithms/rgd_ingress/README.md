# RGD Ingress
Query the WATCH RGD and pull down images within a specified date range and bounding box.
Version: 0.0.1
License: MIT
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
aoi_bounds|Spatial bounds of the Area of Interest (AOI).|Yes
date_range|Pull down images from within this date range|Yes

## Outputs:
Name|Description
---|---
out_dir|Directory where output images / data will be written.  Will be created if it doesn't already exist.
stac_catalog|STAC Catalog JSON string
