Coding Oddities in the WATCH REPO
=================================


This document is an effort to list concepts and patterns that you will see if
you work in this repo that may cause confusion unless you have this prior
knowledge. In some cases we may try to make the code more clear in the future,
but in other cases these patterns bring enough of a benefit that we require
prerequisite knowledge.


NOTE: If you find a coding pattern or abbreviation that confused you at first,
please contribute it here!


Common abbreviations:

* GSD - ground sample distance


Variable abbreviations:

* ``aid`` - annotation id

* ``gid`` - imaGe id -- The first weird one!

* ``cid`` - category id 

* A suffix of ``x`` or ``idx`` - an index (e.g. ``cx`` for category index)

* ``tr`` often means "target", but that pattern has been deprecated and its usually just spelled out as ``target`` now. 

* The "xywh", "ltrb", and "cxywh" are codes indicating the format of bounding boxes for `kwimage.Boxes <https://kwimage.readthedocs.io/en/release/kwimage.structs.boxes.html#module-kwimage.structs.boxes>_`. They stand for "left-x, top-y, width, height", "left-x, top-y, right-x, bottom-y", and "center-x, center-y, width, height" respectively.

* ``dsize`` - This ALWAYS means a (width, height) pair, usually a ``Tuple[int, int]``, but not always. In VERY rare circumstances, an individual width or height may be None to represent that it is not known or needed to be specified. 

* ``shape`` - This is going to be the row-major shape of an array usually. Often (h, w, channel) or just (h, w)

* ``fpath`` - a "file" path. This is used to store a string of ``Path`` object representing a path that points to a file (i.e. not a directory).

* ``dpath`` - a "directory" path. This is used to store a string of ``Path`` object representing a path that points to a directory (i.e. not a file).


Notes on row-vs-column major coordinate axes:

Because numpy makes heavy use of row-major indexing and opencv uses
column-major indexing, it is worth developing a separate notation for when one
style of indexing is being used so we do not confuse them.

* Variables named ``dsize`` / ``size``  or used with ``cv2`` / ``warping``
  operations will use a column-major (i.e. [x, y]) indexing style. Think
  width/height when you see these patterns.


* Variables named ``dims``, ``shape`` or used in numpy / torch / array
  logic will use a row-major (i.e. [r, c]) indexing style. Think row /
  column when you see these patterns.



Short semi-ambiguous identifiers:

    * ``ub.udict`` - The extended ubelt dictionary with set operations and other nice methods
    * ``ub.ddict`` - A defaultdict alias


Module aliases
* ``import ubelt as ub``
* ``import numpy as np``
* ``import networkx as nx``
* ``import geopandas as gpd``
* ``import pytorch_lightning as pl``
* ``import geopandas as gpd``
* ``import pandas as pd``
* ``import seaborn as sns``
* ``import scriptconfig as scfg``


Best Practices:

* When you are working with a list of classes, try to make sure you have it wrapped in a ``kwcoco.CategoryTree`` and use that to shuffle around relevant metadata.

* When working with a set of channels wrt to a single sensor use: ``kwcoco.ChannelSpec`` or  ``kwcoco.FusedChannelSpec`` 

* When working with a set of channels wrt to known or unknown sensors use: ``kwcoco.SensorChanSpec`` or  ``kwcoco.FusedSensorChanSpec`` 

* DONT IMPORT PYPLOT AT THE MODULE LEVEL!!! Always do it in a function. If fact, do most everything inside a function. Reduce the amount of globally scoped code.


Spaces
------

See the section about SPACES in kwcoco docs


There are several 'spaces' here and that can get confusing.

**Native Space** - The space of the data on disk

**Video Space** - The space a sequence is geo-aligned in.  This is the space we generally want to be thinking in.
    It is hard coded wrt to the kwcoco dataset.

**Window Space** - GSD the sliding window is expressed in.
   Defaults to video space.
   Computes a integer-sized box as the 'space_slice' in video space.
   Effectively this space is only used to compute the size of the box
   in the underlying video space. It does nothing else.
   Alias: Grid Space

**Input Space** - GSD of the input to the network
   Computes a scale factor relative to video space.
   Alias: Sample Space
   Alias: Data Space

**Output Space** - GSD of the output of the network
   Scale factor is wrt to video space.
   Alias: Prediction Space
