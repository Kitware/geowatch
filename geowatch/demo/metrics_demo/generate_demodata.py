"""
Generate demodata for the metrics framework
"""
import click
import json
import kwarray
import ubelt as ub
from geowatch.demo.metrics_demo import demo_truth
from geowatch.demo.metrics_demo import demo_rendering
from geowatch.demo.metrics_demo import site_perterbing


def generate_demo_metrics_framework_data(
        roi="DR_R001",
        num_sites=3,
        num_observations=5,
        noise=1.0,
        p_observe=0.5,
        p_transition=0.15,
        seed=409060576688592,
        outdir=None,
        cache=True,
        **kwargs):
    """
    Example:
        >>> from geowatch.demo.metrics_demo.generate_demodata import *  # NOQA
        >>> demo_info = generate_demo_metrics_framework_data(
        >>>     num_sites=5, num_observations=10, noise=2, p_observe=0.5,
        >>>     p_transition=0.3, drop_noise=0.5, drop_limit=0.5)
        >>> print('demo_info = {}'.format(ub.urepr(demo_info, nl=1)))
        >>> # TODO: visualize

    Ignore:
        from geowatch.demo.metrics_demo.generate_demodata import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(generate_demo_metrics_framework_data))
    """

    if outdir is None:
        demo_dpath = ub.Path.appdir("iarpa/smart/ta2/demodata")
    else:
        demo_dpath = ub.Path(outdir)

    demo_dpath.ensuredir()

    image_dpath = demo_dpath / "images"
    true_dpath = demo_dpath / "truth"
    pred_dpath = demo_dpath / "predictions"
    output_dpath = demo_dpath / "demo_output"
    cache_dpath = demo_dpath / "_cache"

    output_dpath.ensuredir()
    true_region_dpath = (true_dpath / "region_models").ensuredir()
    true_site_dpath = (true_dpath / "site_models").ensuredir()
    pred_site_dpath = (pred_dpath / "pred_site_models").ensuredir()

    performer_id = 'alice'
    region_id = roi

    truekw = {
        'region_id': region_id,
        'num_sites': num_sites,
        'num_observations': num_observations,
        'p_transition': p_transition,
        'p_observe': p_observe,
    }
    predkw = {
        'noise': noise,
        'performer_id': performer_id,
        **kwargs,
    }

    rng = kwarray.ensure_rng(seed)
    # TODO: add more parameters here
    depends = [
        rng,
        truekw,
        predkw,
        "v2",
    ]

    stamp = ub.CacheStamp("demodata", dpath=cache_dpath, depends=depends,
                          enabled=cache)
    if stamp.expired():
        cache_dpath.ensuredir()
        # Generate a random truth region
        region, sites, renderables = demo_truth.random_region_model(
            rng=rng, **truekw)

        # Generate predictions that are close to the truth, but not quite.
        # TODO: parametarize the amount of perterbation
        pred_sites = site_perterbing.perterb_site_model(sites, rng=rng,
                                                        **predkw)

        # Dump region file
        region_id = region["features"][0]["properties"]["region_id"]
        region_fpath = true_region_dpath / (region_id + ".geojson")
        region_fpath.write_text(json.dumps(region, indent="  "))
        print(f"wrote true region_fpath={region_fpath}")

        # Dump true site file
        for site in sites:
            site_id = site["features"][0]["properties"]["site_id"]
            # site_fpath = (true_site_dpath / region_id).ensuredir() / (site_id + '.geojson')
            site_fpath = true_site_dpath / (site_id + ".geojson")
            site_fpath.write_text(json.dumps(site, indent="  "))
            print(f"wrote site_fpath={site_fpath}")

        # Dump pred site file
        for site in pred_sites:
            site_id = site["features"][0]["properties"]["site_id"]
            site_fpath = pred_site_dpath / (site_id + ".geojson")
            site_fpath.write_text(json.dumps(site, indent="  "))
            print(f"wrote predicted site_fpath={site_fpath}")

        # Render toy images
        # The image path currently needs to in a very specific format
        region_img_dpath = (image_dpath.joinpath(*region_id.split('_')) / 'images' / 'a' / 'b').ensuredir()
        # TODO: also need to output the crops path as well for visualize_stack_slices.
        # region_img_dpath = (image_dpath / region_id).ensuredir()
        for renderable in ub.ProgIter(renderables, desc="render"):
            demo_rendering.render_toy_georeferenced_image(
                region_img_dpath, renderable, rng=rng)
        stamp.renew()

    demo_info = {
        "region_id": region_id,
        "true_site_dpath": true_site_dpath,
        "true_region_dpath": true_region_dpath,
        "pred_site_dpath": pred_site_dpath,
        "image_dpath": image_dpath,
        "output_dpath": output_dpath,
    }
    return demo_info


@click.command()
@click.option('--roi', default="DR_R001", help="name of the demo region interest")
@click.option('--num_sites', default=3, help="number of demo sites to generate")
@click.option('--num_observations', default=5, help="number of observations of the demo region")
@click.option('--noise', default=1.0, help="Noise level of perterbations")
@click.option('--p_observe', default=0.5, help="The probability of annotating an observation")
@click.option('--p_transition', default=0.5, help="The probability of a site transitions phase on each frame")
@click.option('--seed', default=409060576688592, help="RNG seed")
@click.option('--outdir', default=None, help="Where to write the data. Defaults to ~/.cache/iarpa/smart/ta2/demodata")
@click.option('--cache/--no-cache', default=True, help="enable/disable the cache")
def main(*args, **kwargs):
    import os
    demo_info = generate_demo_metrics_framework_data(*args, **kwargs)
    # Build command to tell the user how to test this data with the eval
    # framework.
    test_args = [
        'python', '-m', 'iarpa_smart_metrics.run_evaluation',
        '--roi', demo_info['region_id'],
        '--gt_dir', os.fspath(demo_info['true_site_dpath']),
        '--rm_dir', os.fspath(demo_info['true_region_dpath']),
        '--sm_dir', os.fspath(demo_info['pred_site_dpath']),
        '--image_dir', os.fspath(demo_info['image_dpath']),
        '--output_dir', os.fspath(demo_info['output_dpath']),
        '--name', 'demoeval',
        '--cache_dir', 'None',
        '--loglevel', 'INFO',
        '--serial',
        '--no-db',
        '--no-viz',
    ]
    command = ' '.join(test_args)
    print(command)


if __name__ == "__main__":
    """
    CommandLine:
        python -m geowatch.demo.metrics_demo.generate_demodata --noise=0.0
    """
    main()
