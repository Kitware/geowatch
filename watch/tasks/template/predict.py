#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a Template for writing training logic.
"""
import kwcoco
import ndsampler
import scriptconfig as scfg
import ubelt as ub
import pathlib


class TemplatePredictConfig(scfg.Config):
    """
    Name this config object based on your prediction technique.

    Use this docstring to write a small blurb about it. It will be printed
    when someone runs ``python -m watch.tasks.template.predict --help``
    """

    default = {
        'dataset': scfg.Value(None, help=ub.paragraph(
            ''' A path to a kwcoco dataset ''')),

        'deployed': scfg.Value(None, help=ub.paragraph(
            ''' A path to a deployed (e.g. torch-liberator) model ''')),

        'out_dpath': scfg.Value('./pred_out', help=ub.paragraph(
            ''' A path to store results ''')),

        # Put other config options like batch_size, xpu, workers,
        # confidence thresholds, etc.. here
    }


def predict_on_dataset(cmdline=False, **kwargs):
    """
    Kwargs:
        See TemplatePredictConfig

    Example:
        >>> from watch.tasks.template.predict import *  # NOQA
        >>> kwargs = {
        >>>     'dataset': 'special:vidshapes8',
        >>> }
        >>> predict_on_dataset(**kwargs)

    """
    config = TemplatePredictConfig(default=kwargs, cmdline=cmdline)

    print('reading datasets')
    input_dset = kwcoco.CocoDataset.coerce(config['dataset'])

    # Make a copy of the dataset
    output_dset = input_dset.copy()
    # Remove all annotations
    output_dset.remove_annotations(output_dset.annots().aids)
    # Make all the paths absolute while we are modifying it
    output_dset.reroot(absolute=True)

    out_dpath = pathlib.Path(ub.ensuredir(config['out_dpath']))

    pred_fpath = out_dpath / 'predictions.kwcoco.json'

    output_dset.fpath = str(pred_fpath)
    # predict-time datasets are often, but not always written in different ways
    # than train-time datasets.

    # Do somethin to load data
    sampler = ndsampler.CocoSampler(input_dset)

    # do something to load model
    model = config['deployed']

    USE_RANDOM_OUTPUTS = True
    if USE_RANDOM_OUTPUTS:

        # Dummy random outputs
        import kwimage
        import kwarray

        # Ensure that we have at least one category for generating
        # random output
        if len(input_dset.index.cats) == 0:
            input_dset.add_category(id="1", name="site")
            output_dset.add_category(id="1", name="site")

        # Do somethin to load data
        sampler = ndsampler.CocoSampler(input_dset)

        rng = kwarray.ensure_rng(None)
        for gid, img in sampler.dset.imgs.items():
            rando_dets = kwimage.Detections.random(
                segmentations=True, classes=sampler.classes, rng=rng)
            rando_dets = rando_dets.scale((img['width'], img['height']))

        anns = list(rando_dets.to_coco(dset=output_dset))
        for ann in anns:
            # Should add a score for the annotation
            ann['score'] = rng.rand()
            output_dset.add_annotation(image_id=gid, **ann)

        # TODO: add random auxiliary channels

    else:
        loader = []

        # This code is not run in the template
        print('make predictions')
        for batch in loader:
            outputs = model(batch)

            # Convert outputs to coco
            # need to be able to associate gid with a prediction
            gid_to_coco_outputs = outputs.to_coco()
            for gid, coco_outputs in gid_to_coco_outputs.items():
                for ann in coco_outputs['anns']:
                    output_dset.add_annotation(image_id=gid, **ann)
                for aux in coco_outputs['auxs']:
                    # Handle writing the data to disk
                    output_dset.add_auxiliary_image(image_id=gid, **aux)

    # Save predictions to disk
    print('write output_dset.fpath = {!r}'.format(output_dset.fpath))
    output_dset.dump(output_dset.fpath, newlines=True)


def main(**kwargs):
    # Parse the config for this script and allow the command line (i.e. the
    # current value of sys.argv) to overwrite
    predict_on_dataset(cmdline=True, **kwargs)


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.fit --help

        python -m watch.tasks.template.predict \
            --dataset=special:vidshapes2-multispectral
    """
    main()
