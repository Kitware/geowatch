from dateutil.parser import parse
import ubelt as ub
import watch
import kwcoco
import pandas as pd
import itertools as it
from watch import heuristics


def build_emperical_transition_probs():

    import numpy as np

    dvc_dpath = watch.find_smart_dvc_dpath()
    # coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data_train.kwcoco.json'
    coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
    dset = kwcoco.CocoDataset(coco_fpath)

    force_sequence = ['No Activity', 'Site Preparation', 'Active Construction', 'Post Construction']

    transitions = []

    deltas = ub.ddict(list)

    import kwarray
    for tid, aids in ub.ProgIter(list(dset.index.trackid_to_aids.items()), desc='build transition probs'):
        _track_annots = dset.annots(list(aids))
        times = [parse(d) for d in _track_annots.images.lookup('date_captured')]
        sorted_idx = ub.argsort(times)
        track_annots = _track_annots.take(sorted_idx)

        frame_indexes = track_annots.images.lookup('frame_index')
        frame_diffs = np.diff(frame_indexes)
        assert np.all(frame_diffs >= 0), 'check'

        times = [parse(d) for d in track_annots.images.lookup('date_captured')]
        states = track_annots.cnames

        seq = kwarray.DataFrameLight({
            'state': states,
            'time': times,
            'frame_indexes': frame_indexes,
        })

        prev_states = None
        for idx, curr_states in seq.groupby('frame_indexes'):
            if prev_states is not None:
                curr_rows = [r for _, r in curr_states.iterrows()]
                prev_rows = [r for _, r in prev_states.iterrows()]

                for src, dst in it.product(prev_rows, curr_rows):
                    delta = dst['time'] - src['time']
                    assert delta.total_seconds() >= 0
                    transitions.append({
                        'src': src['state'],
                        'dst': dst['state'],
                        'delta': delta
                    })

            prev_states = curr_states

        for catname in force_sequence:
            flags = [s == catname for s in states]
            if any(flags):
                _times = list(ub.take(times, flags))
                _delta = max(_times) - min(_times)
                deltas[catname].append(_delta)

        # for i in range(1, len(states)):
        #     src = states[i - 1]
        #     dst = states[i]
        #     src_time = times[i - 1]
        #     dst_time = times[i]
        #     assert dst_time >= src_time
        #     delta = (dst_time - src_time)
        #     transitions.append({'src': src, 'dst': dst, 'delta': delta})

    for k, ds in deltas.items():
        secs = np.array([d.total_seconds() for d in ds])
        secs = secs[secs > 60 * 60 * 24]
        average_days = np.mean(secs) / (60 * 60 * 24)
        print(f'{k} {average_days=!r}')

    valid_states = {c['name'] for c in heuristics.CATEGORIES_SCORED}
    valid_states.add('No Activity')  # hard coded
    valid_transitions = [t for t in transitions if t['src'] in valid_states and t['dst'] in valid_states]

    accum = ub.ddict(lambda: ub.ddict(list))
    for t in valid_transitions:
        accum[t['src']][t['dst']].append(t['delta'])

    final = []
    for src, dst_accum in accum.items():
        for dst, deltas in dst_accum.items():
            final.append({'src': src, 'dst': dst, 'count': len(deltas), 'total_seconds': sum([d.total_seconds() for d in deltas])})

    final_df = pd.DataFrame(final)

    count_piv = final_df.pivot(['src'], ['dst'], ['count']).droplevel(axis=1, level=0)
    seconds_piv = final_df.pivot(['src'], ['dst'], ['total_seconds']).droplevel(axis=1, level=0)
    count_piv = count_piv.fillna(0)
    seconds_piv = seconds_piv.fillna(0)

    # Fix impossible transitions (not sure why they exist in the data, is the
    # way I looked them up bugged?)
    force_sequence = ['No Activity', 'Site Preparation', 'Active Construction', 'Post Construction']

    all_transitions = set(it.product(force_sequence, force_sequence))
    impossible_transitions = set(it.combinations(force_sequence[::-1], 2))
    possible_transitions = all_transitions - impossible_transitions

    for src, dst in impossible_transitions:
        count_piv.loc[src, dst] = 0
        seconds_piv.loc[src, dst] = 0

    # Give all possible transitions a heuristic bump (maybe do an posterior
    # update given a hard-coded prior?)
    for src, dst in possible_transitions:
        count_piv.loc[src, dst] += 1
        seconds_piv.loc[src, dst] += 1

    # reorder DF for nice printing
    count_piv = count_piv[force_sequence].loc[force_sequence]
    seconds_piv = seconds_piv[force_sequence].loc[force_sequence]

    count_probs = count_piv / count_piv.sum(axis=1)
    seconds_probs = seconds_piv / seconds_piv.sum(axis=1)

    # Hueristic linear combination of count and time-based methods
    # for estimating this data.
    hybrid_probs = 0.5 * count_probs + 0.5 * seconds_probs
    hybrid_probs = hybrid_probs / hybrid_probs.sum(axis=1)

    print('\n\ncount_probs =\n{}'.format(ub.repr2(count_probs, nl=1)))
    print('\n\nseconds_probs =\n{}'.format(ub.repr2(seconds_probs, nl=1)))
    print('\n\nhybrid_probs =\n{}'.format(ub.repr2(hybrid_probs, nl=1)))
