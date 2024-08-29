Supporting Projects
===================

These are a list of important projects that are used extensively in GeoWATCH. In
some cases these are being developed simultaneously with GeoWATCH.

First a graph overview of some dependencies

.. image:: https://i.imgur.com/xEea6kL.png

And now links and a description


* https://gitlab.kitware.com/computer-vision/kwcoco - Importance: CRITICAL - This is our data exchange

* https://gitlab.kitware.com/computer-vision/ndsampler - Importance: VERY HIGH - The engine behind the dataloader, the CocoSampler API should be understood if working with the KWCocoVideoDataset.

* https://gitlab.kitware.com/computer-vision/delayed_image - Importance: HIGH - This is what enables efficient sampling of COGs.

* https://gitlab.kitware.com/computer-vision/kwimage - Importance: HIGH - expressive but concise wrappers around cv2, gdal, PIL, scikit-image, rasterio, and some custom tooling

* https://gitlab.kitware.com/computer-vision/kwarray - Importance: HIGH - defines many numpy / torch functional utilities

* https://gitlab.kitware.com/utils/scriptconfig - Importance: HIGH - This library is how the CLI tools are written and configured, basic familiarly is required to modify anything that requires configuration.

* https://github.com/Erotemic/xdoctest - Importance: HIGH - The front line of our testing and documentation

* https://github.com/Erotemic/ubelt : Importance : HIGH - There are a lot of ubelt constructs used in this library. Path, cmd, Cacher, CacheStamp, download, grabdata, Executor, ProgIter, NiceRepr, etc...

* https://gitlab.kitware.com/computer-vision/cmd_queue - Importance: Medium - This is the tool we use for running DAGs of bash commands locally. It is the engine that powers MLOPs.

* https://gitlab.kitware.com/smart/metrics-and-test-framework - Importance: Medium-high - This is our internal fork of the IARPA metrics module / scoring scripts. This is important only for SMART performers.

* https://gitlab.kitware.com/computer-vision/torch_liberator - Importance: Medium - partial weight loading

* https://github.com/Erotemic/networkx_algo_common_subtree - Importance: Low - part of torch-liberator

* https://gitlab.kitware.com/computer-vision/kwplot - Importance: Low - matplotlib stuff. Useful, but usually not on the critical path.

* https://github.com/Erotemic/xdev - Importance: Low - mainly for debugging, xdev.embed, xdev.embed_on_exception_context, developed for and by Jon C, but has some nifty things in it. Using this in production must be avoided.
