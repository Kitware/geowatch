import argparse
import sys

# Import hack to fix somehow broken import of `from
# watch.utils.util_raster import GdalOpen` in run_mtra
from osgeo import gdal, osr  # noqa

from watch.cli.baseline_framework_ingress import baseline_framework_ingress
from watch.cli.baseline_framework_egress import baseline_framework_egress
from watch.cli.run_mtra import run_mtra


def main():
    parser = argparse.ArgumentParser(
        description="Run UConn's MTRA as baseline framework component")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument("-r", "--requester_pays",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with "
                             "`--requestor_payer requester` flag")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("-n", "--newline",
                        action='store_true',
                        default=False,
                        help="Output as simple newline separated STAC items")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        required=False,
                        help="Number of jobs to run in parallel")
    parser.add_argument("--num_pairs",
                        required=False,
                        type=int,
                        help="Number of best Landsat and S2 pairs to select "
                             "for building the harmonization model")

    run_mtra_for_baseline(**vars(parser.parse_args()))

    return 0


def run_mtra_for_baseline(input_path,
                          output_path,
                          outbucket,
                          aws_profile=None,
                          dryrun=False,
                          requester_pays=False,
                          newline=False,
                          jobs=1,
                          num_pairs=6):
    print("* Running ingress *")
    ingress_catalog = baseline_framework_ingress(
        input_path,
        '/tmp/ingress',
        aws_profile=aws_profile,
        dryrun=dryrun,
        requester_pays=requester_pays,
        relative=False,
        jobs=jobs)

    print("* Running MTRA harmonization *")
    mtra_catalog = run_mtra(
        ingress_catalog,
        '/tmp/mtra',
        num_pairs=num_pairs,
        remap_cloudmask_to_hls=True,
        jobs=jobs)

    print("* Running egress *")
    te_output = baseline_framework_egress(
        mtra_catalog,
        output_path,
        outbucket,
        aws_profile,
        dryrun,
        newline,
        jobs)

    return te_output


if __name__ == "__main__":
    sys.exit(main())
