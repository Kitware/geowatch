# Stitch together tiles
This algorithm stitches a group of map tiles saved in a directory together. The resulting image is saved in png format with the image path and bounds saved onto the chain ledger.
Version: 0.0.1
License: MIT
Homepage: [https://tiledriver.com/developer]

## Parameters:
Name|Description|Required
---|---|:---:
image_filenames|List of image filenames to stitch together|Yes

## Outputs:
Name|Description
---|---
image_path|Path to image that was stitched together
image_bounds|Array containing the leaflet bounds of stiched image extent
