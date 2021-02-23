# Get Map Tiles In ROI
This algorithm will gather up map tiles at a given zoom level that intesect with the provided polygon. The source is the national map provided by USGS. All tiles will be written out to disk at a specified location. This location is also saved onto the chain ledger.
Version: 0.0.1
License: MIT
Homepage: [https://tiledriver.com/developer]

## Parameters:
Name|Description|Required
---|---|:---:
roi|Polygon WKT to obtain tiles that intersect.|Yes
zoom|Zoom level|Yes

## Outputs:
Name|Description
---|---
image_filenames|Absolute paths to each image collected separated by commas.
