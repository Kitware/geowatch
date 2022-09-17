Supporting Projects
===================

These are a list of important projects that are used extensively in WATCH. In
some cases these are being developed simultaniosly with WATCH.


* https://gitlab.kitware.com/computer-vision/kwcoco - Importance: CRITICAL - This is our data exchange 

* https://gitlab.kitware.com/computer-vision/ndsampler - Importance: VERY HIGH - The engine behind the dataloader

* https://github.com/Erotemic/xdoctest - Importance: HIGH - The front line of our testing and documentation

* https://github.com/Erotemic/ubelt : Importance : HIGH - There are a lot of ubelt constructs used in this library. Path, cmd, Cacher, CacheStamp, download, grabdata, Executor, ProgIter, NiceRepr, etc...

* https://gitlab.kitware.com/computer-vision/torch_liberator - Importance: Medium - partial weight loading

* https://github.com/Erotemic/networkx_algo_common_subtree - Importance: Low - part of torch-liberator

* https://gitlab.kitware.com/computer-vision/kwarray - Importance: HIGH - defines many numpy / torch functional utilities

* https://gitlab.kitware.com/computer-vision/kwimage - Importance: HIGH - expressive but concise wrappers around cv2, gdal, PIL, scikit-image, rasterio, and some custom tooling

* https://gitlab.kitware.com/computer-vision/kwplot - Importance: Low - matplotlib stuff. Useful, but usually not on the critical path.

* https://github.com/Erotemic/xdev - Importance: Low - mainly for debugging, xdev.embed, xdev.embed_on_exception_context, developed for and by Jon C, but has some nifty things in it. Using this in production must be avoided.

* https://gitlab.kitware.com/utils/scriptconfig - Importance: Medium - generally how the CLI tools are written and configured

* https://gitlab.kitware.com/computer-vision/cmd_queue - Importance: Medium - local scheduling
