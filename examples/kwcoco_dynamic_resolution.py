"""
Demo how to efficiently load an image at a specified resolution.
"""
import watch
dset = watch.demo.coerce_kwcoco('watch-msi', geodata=True)
coco_img = dset.images().coco_images[0]

delayed = coco_img.delay(
    space='video', resolution='0.1 GSD', RESOLUTION_KEY='target_gsd')

image = delayed.finalize()
