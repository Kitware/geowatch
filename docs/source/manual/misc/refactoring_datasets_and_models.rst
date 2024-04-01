Refactoring Datasets and Models
===============================

This is spurred by conversation in https://gitlab.kitware.com/computer-vision/geowatch/-/merge_requests/14

----

I've been thinking about the chicken/egg problem with models/datasets.

Ultimately, it is important to be able to create one without the other, so
there should always be a mechanism for setting each up independently. (We want
to test one without the other).


The Model
---------

The **Model** needs to know about:

* Sensor / channels it is expected to produce.

* Class heads / outputs it should be expected to produce


The Dataset
-----------

The **Dataset** needs to know about:

* Sensor / channels it should be expect to take as input (train and inference).

* Truth it should be expected to produce (train time).


Ambiguous / Other
-----------------

Other important considerations are:

* Dataset statistics: Mean/Std -
  There is a design choice here, either the dataset must produce standardized / (
  i.e. mean/std normalized data), or it needs to provide that information to
  the model. I prefer to have the raw data produced by the dataset to be
  non-standardized. This means that if the original data was in a visual range,
  then the data loader will produce information still in that visual range.
  This makes visualization easier. We could also produce the inverse transform,
  in each data item.

* Dataset statistics: Class Frequency.
  The dataset is the natural place to compute this class frequency information,
  but the model is the natural place to store it, because then it can use it to
  derive default class weight.


* If a class head is requested, but the classes are unspecified, then the
  default should be to use all categories available in the training kwcoco
  dataset.


* The user should have the ability to override the per-input-stem mean/std.

* The user should have the ability to override the per-output-head class
  weights.
