#!/usr/bin/env python3
r"""

Usage:

    ANNOTATIONS_DPATH=$HOME/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6
    python ~/code/watch/dev/poc/compare_iarpa_metrics_versions.py \
        --roi CH_R001 \
        --sm_dir $HOME/data/dvc-repos/smart_expt_dvc/_airflow/preeval14_batch_v28/CH_R001/kit_fixups_v2/cropped_site_models_fixed \
        --gt_dir $ANNOTATIONS_DPATH/site_models \
        --rm_dir $ANNOTATIONS_DPATH/region_models


Requirements:
    pip install ubelt scriptconfig rich cmd_queue pandas
"""
import scriptconfig as scfg
import ubelt as ub
import cmd_queue
import itertools as it
import pandas as pd
import rich


class CompareIarpaMetricsVersionsCLI(scfg.DataConfig):
    """
    Install, run, and compare the results of multiple versions of the metrics
    code.

    This script will create virtual environments for different versions of the
    IARPA metrics and inspect the differences between them.
    """

    roi = scfg.Value(None, help='Region name')
    gt_dir = scfg.Value(None, help='Path to true sites directory')
    rm_dir = scfg.Value(None, help='Path to true regions directory')
    sm_dir = scfg.Value(None, help='Path to predicted site models directory')

    repo_dpath = scfg.Value(None, help='path to a cannonical checkout of the metrics code. Inferred if possible')
    performer = scfg.Value('kit', help='passed to metrics code')

    tmux_workers = scfg.Value(4, help='number of tmux workers')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from compare_iarpa_metrics_versions import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CompareIarpaMetricsVersionsCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        import iarpa_smart_metrics
        if config.repo_dpath is None:
            repo_dpath = ub.Path(iarpa_smart_metrics.__file__).parent.parent
        else:
            repo_dpath = ub.Path(config.repo_dpath)

        src_gitdir = repo_dpath / '.git'
        assert src_gitdir.exists()

        roi = config.roi
        performer = config.performer
        gt_dir = ub.Path(config.gt_dir)
        rm_dir = ub.Path(config.rm_dir)
        sm_dir = ub.Path(config.sm_dir)
        assert gt_dir.exists()
        assert rm_dir.exists()
        assert sm_dir.exists()

        # Install different versions of the code to different virtual
        # environments.
        versions = [
            'kitware-main',
            # 'main',
            'new_kit_speedups_2023-08-01',
            'make-database-reqs-optional',
        ]

        clone_base = ub.Path.appdir('cache', 'watch', 'tests', 'metrics', 'versions')

        queue = cmd_queue.Queue.create(backend='tmux',
                                       size=min(len(versions), config.tmux_workers))

        requests = []
        for version in versions:
            info = ub.cmd(f'git rev-parse {version}', cwd=repo_dpath, check=1)
            commit_hashid = info.stdout.strip()[0:8]
            requests.append({
                'version': version,
                'commit_hashid': commit_hashid,
            })

        for request in requests:
            commit_hashid = request['commit_hashid']
            clone_dpath = clone_base / f'metrics_{commit_hashid}'
            venv_dpath = clone_dpath / f'venv_{commit_hashid}'
            request['venv_dpath'] = venv_dpath
            request['clone_dpath'] = clone_dpath
            sm_hashid = ub.hash_data(sm_dir)[0:8]
            request['output_dir'] = (request['clone_dpath'] / f'output_{roi}_{sm_hashid}')
            clone_dpath.parent.ensuredir()
            if not clone_dpath.exists():
                ub.cmd(f'git clone {src_gitdir} {clone_dpath}', verbose=3)
                ub.cmd(f'git reset --hard {commit_hashid}', verbose=3, cwd=clone_dpath)
            if not venv_dpath.exists():
                ub.cmd(f'python -m venv {venv_dpath}', verbose=3)
            if 0 or  not list(clone_dpath.glob('*.egg-info')):
                # Hack to ensure specific versions
                ub.cmd(f"bash -c 'source {venv_dpath}/bin/activate && pip install pandas==1.5.3'", shell=True, cwd=clone_dpath, verbose=3, check=True)
                ub.cmd(f"bash -c 'source {venv_dpath}/bin/activate && pip install -e .[runtime-strict]'", shell=True, cwd=clone_dpath, verbose=3, check=True)

        # Add invocations of the evaluation script to a command queue.
        for request in requests:
            output_dir = request['output_dir']
            venv_dpath = request['venv_dpath']
            metrics_command = ub.codeblock(
                fr'''
                source {venv_dpath}/bin/activate && python -m iarpa_smart_metrics.run_evaluation \
                    --roi {roi} \
                    --gt_dir {gt_dir} \
                    --rm_dir {rm_dir} \
                    --sm_dir {sm_dir} \
                    --output_dir "{output_dir}" \
                    --activity overall \
                    --performer={performer} \
                    --eval_num=0 --eval_run_num=0 \
                    --sequestered_id '' \
                    --serial --no-viz-slices \
                    --no-viz-region \
                    --no-viz-slices \
                    --no-viz-detection-table --no-viz-comparison-table --no-viz-associate-metrics --no-viz-activity-metrics
                ''')
            request['metrics_command'] = metrics_command
            if not output_dir.exists():
                queue.submit(metrics_command)

        queue.print_commands()
        queue.run()

        # Load up all metric outputs and compare them.
        outputs = {}
        for request in requests:
            hashid = request['commit_hashid']
            output_dir = request['output_dir']
            print(f'Load results: output_dir={output_dir}')
            outputs[hashid] = load_metric_results(output_dir)

        # Compare each pair of outputs
        for out_hashid1, out_hashid2 in it.combinations(outputs, 2):
            outs1 = outputs[out_hashid1]
            outs2 = outputs[out_hashid2]
            status = compare_output_pair(outs1, outs2)
            if status['n_common'] == 0:
                print(f'{out_hashid1} and {out_hashid2} have NO COMMON FILES')
            elif status['has_difference']:
                print(f'{out_hashid1} and {out_hashid2} are NOT the same')
            else:
                print(f'{out_hashid1} and {out_hashid2} are the same')
            print('status = {}'.format(ub.urepr(status, nl=1)))


def compare_output_pair(outs1, outs2):
    status = {}

    fnames1 = set(outs1.keys())
    fnames2 = set(outs2.keys())
    common_fnames = fnames1 & fnames2
    unpaired_fnames = fnames1 ^ fnames2

    status['n_common'] = len(common_fnames)
    status['n_unpaired'] = len(unpaired_fnames)

    if fnames1 != fnames2:
        raise Exception('Versions did not produce the same exact output files')

    exactsame_fnames = []
    hasdiff_fnames = []
    for fname in common_fnames:
        if outs1[fname]['hash'] == outs2[fname]['hash']:
            exactsame_fnames.append(fname)
        else:
            hasdiff_fnames.append(fname)

    status['n_exact_same'] = len(exactsame_fnames)
    status['n_some_diff'] = len(hasdiff_fnames)

    errors = []

    total_major_differences = 0
    total_minor_differences = 0
    total_itemdiff_checks = 0

    bad_fpaths = [
        (fname, outs1[fname]['fpath'], outs2[fname]['fpath'])
        for fname in hasdiff_fnames
    ]
    # Take a closer look at cases where the hash is different
    # it might be a non-important difference
    for fname, fpath1, fpath2 in bad_fpaths:
        if fpath1.name == 'ac_phase_table.csv':
            compare_ac_phase_table(fpath1, fpath2)
        elif fpath1.name.endswith('.csv'):
            df1 = pd.read_csv(fpath1)
            df2 = pd.read_csv(fpath2)
            isna1 = df1.isna()
            isna2 = df1.isna()
            flags = ~((df1 == df2) | (isna1 & isna2))

            total_itemdiff_checks += df1.values.size

            if flags.values.any():
                bad_rows1 = df1[flags.any(axis=1)]
                bad_rows2 = df2[flags.any(axis=1)]

                # Check to see if the difference is a minor floating point
                # difference issue
                import numbers
                import numpy as np
                is_both_nan = (bad_rows1.isna() & bad_rows2.isna())
                is_bad_item = (bad_rows1 != bad_rows2) & ~is_both_nan
                bad_values1 = bad_rows1.values[is_bad_item]
                bad_values2 = bad_rows2.values[is_bad_item]
                is_major_difference = 0
                is_minor_difference = 0
                NUMBER = (numbers.Number, np.number)
                for v1, v2 in zip(bad_values1, bad_values2):
                    if isinstance(v1, NUMBER) and isinstance(v2, NUMBER):
                        if v1 != v2:
                            if np.isclose(v1, v2):
                                is_minor_difference += 1
                            else:
                                is_major_difference += 1
                    else:
                        is_major_difference += 1

                total_major_differences += is_major_difference
                total_minor_differences += is_minor_difference

                if is_major_difference:
                    msg = f'{fname} has {len(bad_rows2)} / {len(df2)} rows with major differences'
                    print('=========')
                    print('BAD ROWS:', msg)
                    print('=========')
                    print(f'{fpath1}')
                    print(f'{fpath2}')
                    print(bad_rows1)
                    print('---')
                    print(bad_rows2)
                    print('---')
                    a = bad_rows1.to_string()
                    b = bad_rows2.to_string()
                    print('Diff:')
                    print(difftext(a, b, colored=True))
                    errors.append(msg)
        else:
            print(f'{fpath1=}')
            t1 = fpath1.read_text()
            t2 = fpath2.read_text()
            print(difftext(t1, t2, colored=True))
            raise Exception

    status['total_major_differences'] = total_major_differences
    status['total_minor_differences'] = total_minor_differences
    status['total_itemdiff_checks'] = total_itemdiff_checks

    has_difference = False
    if status['n_unpaired'] > 0:
        has_difference = True

    if status['n_some_diff'] > 0:
        has_difference = True

    if len(errors):
        has_difference = True

    status['has_difference'] = has_difference
    status['errors'] = errors
    return status


def compare_ac_phase_table(fpath1, fpath2):
    df1 = pd.read_csv(fpath1)
    df2 = pd.read_csv(fpath2)
    assert (df1.columns == df2.columns).all()
    assert (df1['date'] == df2['date']).all()

    def normalize_row(row):
        new_row = {}
        for k, v in row.items():
            if isinstance(v, str) and 'vs.' in v:
                parts = v.split('vs.')
                import ast
                sets = [ast.literal_eval(p) for p in parts]
                v = sets
            new_row[k] = v
        return new_row

    records1 = df1.to_dict('records')
    records2 = df2.to_dict('records')
    for row1, row2 in zip(records1, records2):
        if row1 != row2:
            norm_row1 = normalize_row(row1)
            norm_row2 = normalize_row(row2)
            assert norm_row1 == norm_row2


def load_metric_results(output_dir):
    fpaths = {}
    for r, ds, fs in output_dir.walk():
        for f in fs:
            # Skip images
            if not f.endswith('.png'):
                fpath = r / f
                rel_fpath = fpath.relative_to(output_dir)
                fpaths[rel_fpath] = {
                    'fpath': fpath,
                    'hash': ub.hash_file(fpath),
                }
    return fpaths


def difftext(text1, text2, context_lines=0, ignore_whitespace=False,
             colored=False):
    r"""
    Uses difflib to return a difference string between two similar texts

    Args:
        text1 (str): old text
        text2 (str): new text
        context_lines (int): number of lines of unchanged context
        ignore_whitespace (bool):
        colored (bool): if true highlight the diff

    Returns:
        str: formatted difference text message

    References:
        http://www.java2s.com/Code/Python/Utility/IntelligentdiffbetweentextfilesTimPeters.htm

    Example:
        >>> # build test data
        >>> text1 = 'one\ntwo\nthree'
        >>> text2 = 'one\ntwo\nfive'
        >>> # execute function
        >>> result = difftext(text1, text2)
        >>> # verify results
        >>> print(result)
        - three
        + five

    Example:
        >>> # build test data
        >>> text1 = 'one\ntwo\nthree\n3.1\n3.14\n3.1415\npi\n3.4\n3.5\n4'
        >>> text2 = 'one\ntwo\nfive\n3.1\n3.14\n3.1415\npi\n3.4\n4'
        >>> # execute function
        >>> context_lines = 1
        >>> result = difftext(text1, text2, context_lines, colored=True)
        >>> # verify results
        >>> print(result)
    """
    import ubelt as ub
    import difflib
    text1 = ub.ensure_unicode(text1)
    text2 = ub.ensure_unicode(text2)
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    if ignore_whitespace:
        text1_lines = [t.rstrip() for t in text1_lines]
        text2_lines = [t.rstrip() for t in text2_lines]
        ndiff_kw = dict(linejunk=difflib.IS_LINE_JUNK,
                        charjunk=difflib.IS_CHARACTER_JUNK)
    else:
        ndiff_kw = {}
    all_diff_lines = list(difflib.ndiff(text1_lines, text2_lines, **ndiff_kw))

    if context_lines is None:
        diff_lines = all_diff_lines
    else:
        # boolean for every line if it is marked or not
        ismarked_list = [len(line) > 0 and line[0] in '+-?'
                         for line in all_diff_lines]
        # flag lines that are within context_lines away from a diff line
        isvalid_list = ismarked_list[:]
        for i in range(1, context_lines + 1):
            isvalid_list[:-i] = list(map(any, zip(
                isvalid_list[:-i], ismarked_list[i:])))
            isvalid_list[i:] = list(map(any, zip(
                isvalid_list[i:], ismarked_list[:-i])))

        USE_BREAK_LINE = True
        if USE_BREAK_LINE:
            # insert a visual break when there is a break in context
            diff_lines = []
            prev = False
            visual_break = '\n <... FILTERED CONTEXT ...> \n'
            #print(isvalid_list)
            for line, valid in zip(all_diff_lines, isvalid_list):
                if valid:
                    diff_lines.append(line)
                elif prev:
                    if False:
                        diff_lines.append(visual_break)
                prev = valid
        else:
            diff_lines = list(ub.compress(all_diff_lines, isvalid_list))
    text = '\n'.join(diff_lines)
    if colored:
        text = ub.highlight_code(text, lexer_name='diff')
    return text


__cli__ = CompareIarpaMetricsVersionsCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/compare_iarpa_metrics_versions.py
        python -m compare_iarpa_metrics_versions
    """
    main()
