Roadmap
=======


Features
--------

- [ ] Per-domain normalization

- [ ] Lineage tracking: design the storage format and parser.

- [ ] Bounding box head + loss

- [ ] Multi-head objective support: Allow multiple non-weight-tied versions of the same head to have different losses

- [ ] Better balanced sampling API and code structure.

- [ ] Better augmentation API.

- [ ] Track per-item loss and update sampling probabilities to target some degree of medium to hard difficulty.

- [ ] Hook in off-the-shelf models via a wrapper API.


Quality of Life
---------------

- [ ] Manual specification of input mean / std

- [ ] Better checkpoint / package management CLI tools

- [ ] Remove old nomenclature (which may involve swapping scriptconfig aliases with the main variable).


Bugs
----

- [ ] Delayed Image #1 - bottom-left pixel bug

- [ ] Callbacks with DDP can cause system freeze



Performance
-----------

- [ ] Delayed Image #2 - memoize the optimization

- [ ] JIT The Network

- [ ] Improve augmentation efficiency

- [ ] NDsampler Zarr / hdf5 backend


Research
--------

- [ ] Determine if continual learning helps in this context

- [ ] Compare heterogeneous network to divided-attention network.

- [ ] Reproduce and integrate ScaleMAE


Compatibility
-------------

- [ ] Further subdivide and sequester dependencies

- [ ] Upgrade pytorch lightning / jsonargparse


Documentation
-------------

- [ ] External review / revision
