FAQ: Model Prediction
=====================

Question
--------

I have a general question regarding model prediction. During the model training
phase, we utilized a specific input data dimension (e.g., 192). When our
trained model generates predictions, does it predict a cropped test image of
the same size as the input data (e.g., 192), subsequently stitching the
prediction image into the original size of the test image? Alternatively, does
the model directly predict the original size of the test image?


Answer
------

By default, yes. The input sizes: T H W (i.e. time, width, height) should be the same as the output.

The input resolution and output resolution are allowed to be different (e.g.
you can set ``--input_resolution=2GSD --output_resolution=8GSD``).
The data loader will always produce data at the specified input resolution, and
it encodes the desired output resolution in each batch item, but ultimately,
it's the underlying model's job to produce the output. I think the
MultimodalTransformer does respect it, but the feature is still experimental,
so it might need to be debuged / developed more.

By default, at predict time, the input window size and output window size will
be the same as whatever the train time parameters were (this information is
encoded in the torch package - and tracking metadata like this is one of most
important reasons that we use torch package). But all of the parameters of the
KWCocoVideoDatasetConfig can be overloaded by the user (E.g. we can request
that a model trained at 2GSD predict at 4GSD), so we can use different window
sizes at predict time. I have tested this, and the model still seems to do
well, but I didn't measure if there is a impact on the output quality or not -
it looks reasonable, but I imagine it might have a hard time extrapolating to
positional encodings that are very different than what it saw at train time
(although our continuous ad-hoc curriculum learning ensures the model has seen
a few different variations of the positional encodings).

Wrt to the stitcher, we know:

* input video size (e.g. 24 frames, 4,000 x 4,000 @ 2GSD)

* input time kernel (e.g. 5-consecutive-frames)

* input window size (e.g. 128x128@2GSD)

* input window resolution (e.g. 2GSD)

* output window resolution (e.g. 4GSD)

So, we grid up video spacetime into  a set of "targets", each target species a
spacetime window (e.g. 128x128 and the selected 5 frames).

Now we iterate through each of these targets. Whenever we load a target the
KWCocoVideoDataset also computes its expected output dimensions, and where the
expected output should be stitched into video space (e.g. this output will go
on image-ids x,y,z in bounding box location LTRB). The sampled input for each
target is fed to the model, and it's the model's job to produce data that
agrees with the computed output dimensions. Now, we have an output, and know
where it should go. We pass that to the stitcher, which "accumulates" the new
information into a pre-allocated memory buffer big enough to hold the entire
output (this actually needs optimization, sometimes videos are too long / bit).
Overlaping windows are smoothed out by having the stitcher accumulate running
weighted averages. The final image written to disk is the mean image of all
data that was stitched in each location.

I think this image illustrates the process well:

.. .. image:: https://i.imgur.com/exGv3uX.png

.. image:: https://data.kitware.com/api/v1/file/656fd3a8dfc0e5d60cffa244/download


I've been writing up slides like this with software-level descriptions here:
https://docs.google.com/presentation/d/125kMWZIwfS85lm7bvvCwGAlYZ2BevCfBLot7A72cDk8/edit#slide=id.g2947606d6c1_0_309

Lastly, I want to point out that a transformer with a proper decoder can
"encode" any sequence of images in any resolution and then "decode" those to a
completely different set of spacetime locations. I envision a future network
doing this. Given an input sequence we provide it with positional encodings
that seed the output locations we are interested in prediction (which could be
the same as the inputs, or it could ask a what if question by choosing a
positional encoding for a place it hasn't seen yet).
