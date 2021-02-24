# Output Image To Web
This algorithm will pull the path to an image (RGB png currently supported) and image bounds (as defined by leaflet) from the chain ledger and re-enter this information with the correct keys needed by the Algorithm Toolkit to send an image back to a chain endpoint request for placement onto a map.
Version: 0.0.1
License: MIT
Homepage: [https://tiledriver.com/developer]

## Parameters:
Name|Description|Required
---|---|:---:
image_path|Path of image to send to client|Yes
image_bounds|Array containing the leaflet bounds of image extent|Yes

## Outputs:
Name|Description
---|---
image_url|Path of image to send to client
image_extent|Array containing the leaflet bounds of image extent
