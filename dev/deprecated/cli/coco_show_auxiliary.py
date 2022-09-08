"""
Visualize kwcoco auxiliary channels to spot-inspect if they are aligned nicely

CommandLine:
    python -m watch.cli.coco_show_auxiliary ~/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/landcover.kwcoco.json
    python -m watch coco_show_auxiliary ~/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/landcover.kwcoco.json
"""
import kwimage
import kwcoco
import ubelt as ub
import scriptconfig as scfg
# import numpy as np


class ShowAuxiliaryConfig(scfg.Config):
    """
    Visualize kwcoco auxiliary channels to spot-inspect if they are aligned
    nicely
    """
    default = {
        'src': scfg.Value(None, position=1, help='source kwcoco file'),

        'gid': scfg.Value(None, type=int, help='image id to visualize, defaults to the first one'),

        'channel1': scfg.Value(None, type=str, help='base auxiliary channel to compare against. If None, the script chooses one.'),

        'channel2': scfg.Value(None, type=str, help='auxiliary channel to overlay on the base. If None, the script chooses one.'),
    }


def main(cmdline=True, **kwargs):
    import kwplot
    plt = kwplot.autoplt()

    config = ShowAuxiliaryConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(config, nl=1)))
    src = config['src']
    print('Read src = {!r}'.format(src))
    dset = kwcoco.CocoDataset.coerce(src)

    gid = config['gid']
    if gid is None:
        gid = ub.peek(dset.index.imgs.keys())

    try:
        import xdev
    except Exception:
        do_plots_for_gid(dset, gid, config)
    else:
        gids = [gid] + list(set(dset.imgs.keys()) - {gid})
        for gid in xdev.InteractiveIter(gids):
            do_plots_for_gid(dset, gid=gid, config=config)
            xdev.InteractiveIter.draw()
            plt.show(block=False)

    plt.show()


def do_plots_for_gid(dset, gid, config):
    print('Plotting gid = {!r}'.format(gid))
    img = dset.index.imgs[gid]
    print('Plot Auxiliary')
    plot_auxiliary_images(dset, img)
    print('Plot Overlay')
    overlay_auxiliary_images(dset, gid, config)


def test_auxiliary_values(dset):
    """
    import kwcoco
    dset = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/rutgers_material_seg.kwcoco.json')
    """
    import numpy as np
    import torch.utils.data

    class SimpleCocoLoader(torch.utils.data.Dataset):
        def __init__(self, dset):
            self.dset = dset
            self.gids = list(self.dset.imgs.keys())

        def __len__(self):
            return len(self.gids)

        def __getitem__(self, idx):
            gid = self.gids[idx]

            img = self.dset.index.imgs[gid]
            coco_img = self.dset.coco_image(img['id'])

            channels = coco_img.channels.fuse()
            channels = 'matseg_0|matseg_1'
            delayed = coco_img.delay(channels=channels)
            data = delayed.finalize()

            # cropped = delayed.delayed_crop((slice(-256, None), slice(-256, None)))
            # data = cropped.finalize()

            item = {
                'data': data,
                'gid': gid,
                'idx': idx,
            }
            return item

    self = SimpleCocoLoader(dset)
    loader = torch.utils.data.DataLoader(self, num_workers=8, shuffle=False,
                                         collate_fn=ub.identity)
    data_iter = iter(loader)
    import kwarray
    import pandas as pd
    # item = next(data_iter)
    for batch in data_iter:
        for item in batch:
            data = item['data']
            gid = item['gid']
            stats = ub.dict_diff(kwarray.stats_dict(data, axis=(0, 1)), {'shape'})
            stats = pd.DataFrame(stats)
            print(stats.T)
            nans = np.isnan(data)
            nans_per_c = nans.sum(axis=(0, 1))
            num_pixels = np.prod(nans.shape[0:2])
            percent_nans = nans_per_c / num_pixels
            print('data.shape = {!r}'.format(data.shape))
            # nans_per_hw = nans.sum(axis=2)
            # print('job.gid = {!r}'.format(job.gid))
            print('gid = {!r}'.format(gid))
            print('percent_nans = {!r}'.format(percent_nans))


def plot_auxiliary_images(dset, img, fnum=1):
    """
    Args: coco image
    """
    import kwplot
    canvas_dsize = (img['width'], img['height'])

    auxiliary_corners = []

    for aux in img.get('auxiliary', []):
        aux_to_img = kwimage.Affine.coerce(aux['warp_aux_to_img'])
        aux_box = kwimage.Boxes([[0, 0, aux['width'], aux['height']]], 'xywh')
        corners_in_aux = aux_box.to_polygons()[0]
        corners_in_img = corners_in_aux.warp(aux_to_img)
        auxiliary_corners.append((corners_in_img, aux['channels'], aux_to_img))

    # canvas = np.zeros(tuple(canvas_dsize[::-1]) + (3,))
    pnum_ = kwplot.PlotNums(nSubplots=len(auxiliary_corners))
    fig = kwplot.figure(fnum=fnum, doclf=True)

    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0, bottom=0)

    colors = kwimage.Color.distinct(len(auxiliary_corners))
    for color, (poly, channels, aux_to_img) in zip(colors, auxiliary_corners):
        # label = channels[0:32]

        text = ub.repr2(aux_to_img.concise(), nl=1, precision=3)

        delayed = dset.delayed_load(img['id'], channels=channels)

        on_img_canvas = delayed.take_channels([0]).finalize()
        on_img_canvas = kwimage.normalize_intensity(on_img_canvas)

        canvas2 = kwimage.draw_text_on_image(
            on_img_canvas, text,
            org=(10, canvas_dsize[1] // 2),
            valign='center', halign='left', fontScale=2, thickness=3,
            color='white', border={'thickness': 6})

        ax = kwplot.imshow(canvas2, fnum=1, pnum=pnum_())[1]
        poly.draw(color=color, alpha=0.3, border=True)
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ax.set_xlim(minx - 100, maxx + 100)
        ax.set_ylim(miny + 200, maxy - 200)
        ax.set_title(channels[0:16] + '...')

    fig.suptitle(img['name'])


def overlay_auxiliary_images(dset, gid, config, fnum=2):
    import kwplot
    import numpy as np

    img = dset.index.imgs[gid]

    available_channels = []
    if img.get('file_name', None) is not None:
        available_channels.append(img.get('channels', None))
    for aux in img.get('auxiliary', []):
        available_channels.append(aux.get('channels', None))

    heuristic_channel_candidates1 = ub.oset([
        'red',
    ])
    heuristic_channel_candidates2 = ub.oset([
        'road', 'inv_overlap1', 'landcover'
    ])

    channel2 = config['channel2']
    if channel2 is None:
        candidates2 = heuristic_channel_candidates2 & ub.oset(available_channels)
        channel2 = candidates2[0] if len(candidates2) else available_channels[-1]

    channel1 = config['channel1']
    if channel1 is None:
        candidates1 = heuristic_channel_candidates1 & ub.oset(available_channels)
        channel1 = candidates1[0] if len(candidates1) else available_channels[0]

    delayed_frame = dset.delayed_load(gid, channels=channel1)
    raw1 = delayed_frame.finalize()
    raw1 = np.nan_to_num(raw1)
    norm1 = kwimage.normalize_intensity(raw1)

    delayed_frame = dset.delayed_load(gid, channels=channel2)
    raw2 = delayed_frame.finalize()
    raw2 = np.nan_to_num(raw2)

    # import matplotlib.cm  # NOQA
    # import matplotlib as mpl
    # cmap_ = mpl.cm.get_cmap('plasma')
    # norm2 = cmap_(raw2[:, :, 0])
    norm2 = kwimage.normalize_intensity(raw2[:, :, 0])
    norm2 = kwimage.make_heatmask(norm2, cmap='plasma', with_alpha=False)
    norm2 = kwimage.ensure_alpha_channel(norm2)

    # norm2[..., 3] = (raw2[..., 0] != 0) * norm2[..., 3]
    norm2[..., 3] = 0.4

    overlaid = kwimage.overlay_alpha_images(norm2, norm1)
    overlaid = overlaid.clip(0, 1)

    kwplot.figure(fnum=fnum)
    _, ax1 = kwplot.imshow(norm1, pnum=(1, 2, 1))
    _, ax2 = kwplot.imshow(overlaid, pnum=(1, 2, 2))
    ax1.set_title(f'base {channel1}')
    ax2.set_title(f'overlaid {channel2}')
    # kwplot.imshow(norm1)


_SubConfig = ShowAuxiliaryConfig

if __name__ == '__main__':
    main()
