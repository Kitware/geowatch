FAQ on CloudMasks
=================

Question
--------
There are a great deal of images that have clouds and sections where the images are completely black pixels.  Is there a way to deal with both of these images?


Answer
------

The ability to mask clouds depends on the availability of quality bands. The toydata doesn't have anything like that, so the MWE won't be able to demonstrate it, but if you load a non-median kwcoco file, then it will work (the median data should have already filtered out as many clouds as possible, so there are no qa bands for it).
 Look at the "DYNAMIC FILTER / MASKING OPTIONS" section in the config of geowatch/tasks/fusion/datamodules/kwcoco_dataset.py file.

You will see options like:

* observable_threshold - set to a non-zero value to filter out frames with unobservable pixels.

* mask_low_quality - set to True to force the network to nan-out cloud pixels marked by the QA bands. (there is a qa_bands.py that defines how this is done) 

* mask_samecolor_method - set to "histogram" to force the dataloader to nan-out large homogeneous regions. This is very effective when the QA band doesn't exist, but it can trigger incorrectly sometimes and remove valid data. However, this is rare. The "region" option is more computationally expensive, but also more accurate.

* resample_invalid_frames - set the number of times you will attempt to resample a frame before it gives up. 

Note: that in some cases it may not be possible to obtain a valid sample according to your constraints, thus you may have to manually drop batches yourself.
