Sensor Channel Spec
-------------------

We use the following notation to describe combinations of channels in a human readable way.



Say you are computing

If you want to write a raster that has 17 channels, you need to give them a
name. Let's call them mychan. The channel spec would refers to multiple bands
with the same name using a slice-like notation: mychan:17

.. code::

    from delayed_image import ChannelSpec
    spec = ChannelSpec('mychan:17')
    spec.normalize()



You can see that each "fused-band" is connected together by a pipe ``|``.

So when I want to train a model that fuses RGB and the new mychan features, I
would tell my network to run with --channels=red|green|blue|mychan:17 (which
actually gets expanded out as previously shown)


We also have the idea of "Late-fused" channels. These are channels that will be
input to the network via different stems, so they may be at different
resolutions. We use a comma , to separate these. So if I wanted to late fuse
RGB with the mychan features I might use red|green|blue,mychan:17.


Additionally, there is an extension to the channel spec called the sensorchan
spec, which allows for specification of what sensor the channels belong to.

We refer to sensors using the following codes: S2=sentinel 2, WV=worldview,
L8=landsat-8, PD=planet dove.

Say I have a feature that only belongs to worldview called `mywvfeat:4`, and I
want to early fuse that with worldview RGB channels, but I want to late fuse
sential2 RGB channels as well. I would specify that like:

`S2:(red|green|blue),WV:(red|green|blue|mywvfeat:4)`


The sensors are able to distribute over late fused sensors as well, so if I
wanted to include late fused `nir|swir16|swir22` with the S2 sensor I would write:


`S2:(red|green|blue,nir|swir16|swir22),WV:(red|green|blue|mywvfeat:4)`

That expands  like this

.. code::

    from delayed_image import SensorChanSpec
    spec = SensorChanSpec.coerce('S2:(red|green|blue,nir|swir16|swir22),WV:(red|green|blue|mywvfeat:4)')
    print(spec.normalize())


S2:red|green|blue,S2:nir|swir16|swir22,WV:red|green|blue|mywvfeat.0|mywvfeat.1|mywvfeat.2|mywvfeat.3


Its also possible to take an expanded code and shorten it.

.. code::


    from delayed_image import SensorChanSpec
    spec = SensorChanSpec.coerce('S2:red|green|blue,WV:red|green|blue,L8:red|green|blue')
    print(spec.concise())

Results in:

``(L8,S2,WV):red|green|blue``


So long story short, if you are writing hidden features as your final output,
make sure you specify ``"channels": "<featname>:<num>"`` in your auxiliary
dictionary.
