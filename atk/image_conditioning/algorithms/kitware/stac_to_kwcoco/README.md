# STAC to KWCOCO
Conversion tool to convert a STAC catalog to a KWCOCO dataset
Version: 0.0.1
License: MIT
Homepage: []

## Parameters:
Name|Description|Required
---|---|:---:
kwcoco|Path to location where KWCOCO file of converted STAC catalog will be written.|Yes
catalog|Path to the root of the STAC catalog to convert.|Yes

## Outputs:
Name|Description
---|---
output_path|Path to location where KWCOCO file was written
dataset|Output KWCOCO file as a JSON string
