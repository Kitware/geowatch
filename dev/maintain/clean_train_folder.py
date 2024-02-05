"""
Limited, but similar functionality to ~/code/netharn/netharn/cli/manage_runs.py
to remove batch visualizations that take up a lot of space.
"""
import pint
import ubelt as ub

ureg = pint.UnitRegistry()


class ImageDirectory(ub.NiceRepr):
    def __init__(self, dpath):
        self.dpath = dpath
        self.fpaths = None
        self.stats = None
        self.sizes = None
        self.total_mb = None
        self.total_bytes = None

    def __nice__(self):
        return f'{self.total_mb} - len(self.fpaths)'

    def build(self):
        fpaths = sorted(self.dpath.glob('*.jpg'))
        self.fpaths = fpaths
        self.stats = []
        sizes = []
        for fpath in fpaths:
            stat = fpath.stat()
            self.stats.append(stat)
            sizes.append(stat.st_size * ureg.bytes)

        self.sizes = sizes
        total_bytes = sum(sizes)
        total_mb = total_bytes.to('megabytes')
        self.total_mb = total_mb
        self.total_bytes = total_bytes
        return self

    def select_removers(self, max_size, keep_atleast=0):
        total = 0
        all_indexes = set(range(len(self.fpaths)))
        keep_idxs = []
        for idx in generate_indexes(len(self.fpaths)):
            total += self.sizes[idx]
            if total > max_size and len(keep_idxs) >= keep_atleast:
                break
            keep_idxs.append(idx)

        remove_idxs = sorted(all_indexes - set(keep_idxs))
        remove_fpaths = list(ub.take(self.fpaths, remove_idxs))
        remove_sizes = list(ub.take(self.sizes, remove_idxs))
        keep_sizes = list(ub.take(self.sizes, keep_idxs))

        info = {}

        zero = 0 * ureg.bytes
        info['remove_size'] = sum(remove_sizes, start=zero).to('megabytes')
        info['keep_size'] = sum(keep_sizes, start=zero).to('megabytes')
        info['keep_num'] = len(keep_idxs)
        info['remove_num'] = len(remove_idxs)
        return remove_fpaths, info


def main():
    # dpath = '/data/projects/smart/smart_watch_dvc/training/horologic/jon.crall/Drop1-20201117/runs/SC_smt_it_stm_p8_newanns_weighted_mat6raw6_v41/lightning_logs/version_0/monitor/train/batch'
    dpath = '/data/projects/smart/smart_watch_dvc/training/horologic/jon.crall/Drop1-20201117/runs/SC_smt_it_stm_p8_newanns_weighted_mat6raw6_v41/lightning_logs/version_0/monitor/train/batch'

    training_dpath = ub.Path('/data/projects/smart/smart_watch_dvc/training')

    batch_dpaths = list(training_dpath.glob('*/*/*/runs/*/lightning_logs/version_*/monitor/train/batch'))
    # batch_dpaths = list(training_dpath.glob('*/*/*/runs/*/lightning_logs/version_*/monitor/validate/batch'))
    # batch_dpaths = list(training_dpath.glob('*/*/*/runs/*/lightning_logs/version_*/monitor/validate/sanity_check'))

    image_dirs = []
    for dpath in ub.ProgIter(batch_dpaths, verbose=3):
        imgdir = ImageDirectory(dpath).build()
        print(f'imgdir={imgdir}')
        image_dirs.append(imgdir)

    total_mb = 0 * ureg.bytes
    for imgdir in image_dirs:
        total_mb += imgdir.total_mb
    total_gb = total_mb.to('gigabytes')
    print(f'total_gb={total_gb}')

    max_size = 64 * ureg.megabytes

    remove_infos = []
    all_remove_fpaths = []

    for imgdir in image_dirs:
        self = imgdir
        remove_fpaths, info = self.select_removers(max_size, keep_atleast=4)
        all_remove_fpaths.extend(remove_fpaths)
        remove_infos.append(info)

    total_keep = sum([g['keep_size'] for g in remove_infos], start=0 * ureg.bytes).to('gigabytes')
    total_remove = sum([g['remove_size'] for g in remove_infos], start=0 * ureg.bytes).to('gigabytes')
    print(f'total_keep={total_keep}')
    print(f'total_remove={total_remove}')

    for fpath in ub.ProgIter(all_remove_fpaths):
        fpath.delete()


def generate_indexes(total):
    """
    E.g. if total is 10, should generate something like:

    0, 9, 4, 2, 6, 1, 5, 3, 7, 8

    To visualize the pattern

      |0|1|2|3|4|5|6|7|8|9|
    0 |x| | | | | | | | | |
    1 |.| | | | | | | | |x|
    2 |.| | | |x| | | | |.|
    3 |.| |x| |.| | | | |.|
    4 |.| |.| |.| |x| | |.|
    6 |.|x|.| |.| |.| | |.|
    8 |.|.|.| |.|x|.| | |.|
    7 |.|.|.|x|.|.|.| | |.|
    5 |.|.|.|.|.|.|.|x| |.|
    9 |.|.|.|.|.|.|.|.|x|.|

    Example:
        >>> total = 10
        >>> gen = generate_indexes(total)
        >>> result = list(gen)
        >>> assert set(result) == set(range(total))
        >>> print(result)
        [0, 9, 4, 2, 6, 1, 5, 3, 7, 8]
    """
    start = 0
    stop = total
    yield stop - 1
    yield start
    yield from generate_midpoints(start, stop - 1)


def generate_midpoints(start, stop):
    import itertools as it
    mid = (start + stop) // 2

    if start == stop or start == mid:
        return

    yield mid

    left_start = start
    left_stop = mid
    left_gen = generate_midpoints(left_start, left_stop)

    right_start = mid
    right_stop = stop
    right_gen = generate_midpoints(right_start, right_stop)

    for a, b in it.zip_longest(left_gen, right_gen):
        if a is not None:
            yield a
        if b is not None:
            yield b


def alt(total):
    import math
    digits = math.ceil(math.log2(total))
    ordered = list(range(total))
    fmtstr = '{:0' + str(digits) + 'b}'
    binary = [fmtstr.format(x) for x in ordered]
    munged = sorted([b[::-1] for b in binary])
    new_order = [int(b[::-1], 2) for b in munged]
    return new_order


def generate_all(start, stop):
    yield stop - 1
    yield from generate_from_starts(start, stop - 1)


def generate_from_starts(start, stop):
    import itertools as it
    mid = (start + stop) // 2
    if start == mid:
        yield start
    else:
        for a, b in it.zip_longest(
                generate_from_starts(start, mid),
                generate_from_starts(mid, stop)
        ):
            if a is not None:
                yield a
            if b is not None:
                yield b

from typing import Generator  # NOQA


def farthest_from_previous(start: int, stop: int) -> Generator[int, None, None]:
    """

    Use case:

        I have a directory of ordered images images that were generated to
        visualize neural network training iterations. I create one of these
        directories every time I train a network.

        These visualizations can start to take up too much disk space, and
        removing some percent of them would free up a lof of space, but still
        leave some of the visualizations in case I wanted to go back and
        inspect an old run. So the question is: which of these images do I
        keep?

        To formally talk about this problem we will refer to files as "items",
        and the file size will be the "weight" of each "item".

        In general, if we can take N items, we should pick items equally spaced
        (or as close to it as possible). But imagine we can choose items within
        a total weight constraint.
        I.e. sum(item['weight'] for item in items) < W

        We can formulate this as an optimization problem:

            # this is not complete...

            Let N = the number of candidate items
            Let W = the maximum weight allowed
            Let x[i] = be an indicator variable if the i-th item is taken
            Let w[i] = be the weight of the i-th item

            objective:

                Consider each pair of nodes with indexes i < j where
                both x[i] > 0 and x[j] > 0 and not
                any(x[k] for k in range(i, j + 1)).
                These are chosen neighbors. Call them neighbs

                Take:
                    max_dist = max((j - i) for i, j in neighbs)
                    min_dist = min((j - i) for i, j in neighbs)
                    total_chosen = sum(x[i] for i in range(N)
                    diff_delta = max_dist - min_dist

                Minimize:
                    # We want to choose as many points as possible such that
                    # the difference between the furthest pair of neighbors and
                    # closest pair of neighbors is minimized
                    # q: does total_chosen need a multiplier
                    #    such that its always the secondary objective?
                    diff_delta - total_chosen

                # todo: nicer formulation of objective.
                # basic idea: distribute chosen points uniformly

            constraint:

                # Total weight is within allowance
                sum(w[i] * x[i] for in range(N)) <= W

        #### The idea of these paragraphs is to motivate the restricted
        #### version of the problem where this heuristic is optimal.

        A specific variant of the above problem is the case where you only have
        one shot to decide if you want to remove an item. We can keep as many
        items as we want, but we can only query the weight of the item once,
        and after you do you have to decide if you keep or delete everything
        else.

        Such a constraint minimizes the number of filesystem operations you
        have to perform, which greatly speeds up the procedure.

        If we keep any of the images, we probably want to see what the network
        was doing at different points in the training process. If we can only
        keep one image, it should be the last one: see what the end state was
        like. If we can keep two, then we want to take the first one as well,
        so we can see what the network looked like at the start of training. If
        we can take three, then perhaps we should take the previous two and
        then one as far away from either of them as possible, so take the
        middle one if the number of items is odd, otherwise pick one of the two
        equidistant items.

        This motivates a heuristic to obtain a feasible solution to the above
        objective. It will not optimize the original objective in all cases,
        but in many cases it will.

    Example:
        >>> total = 10
        >>> start, stop = 0, 10
        >>> gen = farthest_from_previous(start, stop)
        >>> result = list(gen)
        >>> assert set(result) == set(range(start, stop))
        >>> print(result)
    """
    import itertools as it

    def from_starts(start: int, stop: int) -> Generator[int, None, None]:
        if start < stop:
            low_mid: int = (start + stop) // 2
            high_mid: int = (start + stop + 1) // 2

            left_gen = from_starts(start, low_mid)
            right_gen = from_starts(high_mid, stop)

            pairgen = it.zip_longest(left_gen, right_gen)
            flatgen = it.chain.from_iterable(pairgen)
            filtgen = filter(lambda x: x is not None, flatgen)
            yield from filtgen
            if low_mid < high_mid:
                yield low_mid
    if start < stop:
        yield stop - 1
        yield from from_starts(start, stop - 1)
