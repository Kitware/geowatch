Roadmap
=======


Features
--------

- [ ] **Per-domain normalization**: keep track of mean/std per "domain", which can be user-defined.

- [ ] **Lineage tracking**: design the storage format and parser.

- [ ] **Bounding Box Head**: Add support for bounding box heads and associated loss functions.

- [ ] **Multi-head objectives**: Allow multiple non-weight-tied versions of the same head to have different losses

- [ ] **Balanced Sampling**: Better balanced API and code structure.

- [ ] **Augmentation**: Better augmentation API with more options (time symmetry).

- [ ] **Online Hard Negative Mining**: Track per-item loss and update sampling probabilities to target some degree of medium to hard difficulty.

- [ ] **OTF Models**: Off-the-shelf Model wrapper API.

- [ ] **Distilation**: Easy distilation by settting heatmaps from existing predictions as truth targets.

- [ ] **SMQTK**: Integrate with SMQTK by providing it feature descriptors derived from our network activations / heatmaps

- [ ] **Train Monitoring**: Log more weight statistics like rank, magnitude, etc. in tensorboard to better understand the training process.


Quality of Life
---------------

- [ ] Manual specification of input mean / std at train (or predict) time.

- [ ] Better checkpoint / package management CLI tools

- [ ] Remove old nomenclature (which may involve swapping scriptconfig aliases with the main variable).

- [ ] Refactor tracking API. It's the odd-duck, otherwise everything else follows very similar patterns.


Bugs
----

- [ ] Delayed Image #1 - bottom-left pixel bug

- [ ] Callbacks with DDP can cause system freeze; we can workaround by disabling our callbacks, but results in other limitations.


Performance
-----------

- [ ] Delayed Image #2 - memoize the optimization

- [ ] JIT The Network - Or otherwise build efficient inference structure

- [ ] Improve augmentation efficiency - Dataloaders can be bottlenecks depending on params

- [ ] NDsampler Zarr / HDF5 backend - Zarr is newer, HDF5 works similarly.

- [ ] On-disk stitching. Allow predictions to be stitched into context directly on disk (perhaps using an Zarr/HDF5 continer?) instead of always in memory (keep the in-memory option though).


Research
--------

- [ ] Design experiment to determine if continual learning helps in this context.

- [ ] Design experiment to compare heterogeneous network to divided-attention network.

- [ ] Reproduce and integrate ScaleMAE.

- [ ] Can we find a better way to use SAM as foundational model feature?

- [ ] Support "soft" targets for instance segmentation loss.

- [ ] Build new KWCoco datasets

   - [ ] QFabric

   - [ ] Black Marble

- [ ] Support for point-based annotations at train time. Build a loss function.


Compatibility
-------------

- [ ] Further subdivide and sequester dependencies.

- [ ] Upgrade pytorch lightning / jsonargparse to latest versions.


Documentation
-------------

- [ ] External review / revision.

- [ ] Document how to effectively use MLOps (and potentially improve on).


System Design
-------------

- [ ] Use the MLops directory structure in smartflow. This will ultimately allow us to gain the caching advantages of mlops with the horiztonal scaling of smartflow.

- [ ] Ensure smartflow output can be connected to mlops aggregate.

- [ ] Extend mlops to make it easier to test and evaluate ensembles.

- [ ] Extend mlops with teamfeats nodes.

- [ ] Smartflow tiling to split up regions, run prediction on smaller regions, and then consolidate stitching.

- [ ] Better support for training on AWS: https://www.reddit.com/r/MachineLearning/comments/18mfi70/p_kubernetes_plugin_for_mounting_datasets_to/


Algorithmic Exploration
-----------------------

- [ ] Improve High Resolution "Tracking" (Polygon Extraction / Classification).

- [ ] Measure uncertainty.

- [ ] Recurrent transformers that can look at previous predictions in a different context, and then update the predictions.

- [ ] Add decoder to predict unobserved events.


User Interface
--------------

- [ ] Lightning Extension that replaces the rich progress bar with a textual TUI, the idea is the engineer can manually tweak hyperparameters, or request status / visualizations on the fly.
