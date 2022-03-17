import ubelt as ub
import shutil
import watch


model_candidates = [
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=11-step=62759-v2.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=11-step=62759-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=11-step=62759.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=12-step=67989-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=14-step=78449-v2.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=4-step=26149-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=4-step=26149-v4.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=5-step=31379.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=6-step=36609-v2.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=6-step=36609-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=8-step=47069.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=9-step=52299.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p1_v0100/BOTH_TA1_COMBO_TINY_p1_v0100_epoch=107-step=110591.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p1_v0100/BOTH_TA1_COMBO_TINY_p1_v0100_epoch=4-step=5119-v2.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p1_v0100/BOTH_TA1_COMBO_TINY_p1_v0100_epoch=41-step=43007.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104_epoch=12-step=13311.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104_epoch=17-step=18431.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v0104_epoch=79-step=81919.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106_epoch=30-step=31743.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106_epoch=60-step=62463.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106/BOTH_TA1_COMBO_TINY_p2w_raw_scratch_v2_v0106_epoch=98-step=101375.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=100-step=103423.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=13-step=14335.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=14-step=15359.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=15-step=16383.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=33-step=34815.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v0103_epoch=41-step=43007.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=10-step=11263.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=14-step=15359.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=18-step=19455.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=3-step=4095.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=55-step=57343.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=56-step=58367.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105/BOTH_TA1_COMBO_TINY_p2w_raw_xfer_v2_v0105_epoch=57-step=59391.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=0-step=1023-v1.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=0-step=1023-v4.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=10-step=11263.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=15-step=16383.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=2-step=3071-v2.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=2-step=3071-v4.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=3-step=4095-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=3-step=4095.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=4-step=5119-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=6-step=7167.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=60-step=62463.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=8-step=9215.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/BOTH_TA1_COMBO_TINY_p2w_v0101/BOTH_TA1_COMBO_TINY_p2w_v0101_epoch=9-step=10239-v3.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/FUSION_EXPERIMENT_ML_only_nowv_p8_V130/FUSION_EXPERIMENT_ML_only_nowv_p8_V130_epoch=12-step=153022.pt',

    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_V125/FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_V125_epoch=10-step=2815.pt',
    '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/Drop2-Aligned-TA1-2022-02-15/FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_V125/FUSION_EXPERIMENT_M_late_nowv_p8_scratch_shorter_V125_epoch=3-step=1023.pt',
]


dvc_dpath = watch.find_smart_dvc_dpath()

model_paths = []
for sc in model_candidates:
    model_fpath = dvc_dpath / sc.split('smart_watch_dvc/')[1]
    model_paths.append(model_fpath)
    pass

print(len(model_paths))
model_paths = sorted(set(model_paths))
print(len(model_paths))

print(ub.repr2(list(map(str, model_paths))))

contenders_dpath = dvc_dpath / 'models/fusion/eval3_candidates/packages'

move_jobs = []
for fpath in map(ub.Path, model_paths):
    rel_fpath = fpath.relative_to(dvc_dpath)
    contenders_dpath
    expt_suffix = '/'.join(rel_fpath.parts[-2:])
    dst_fpath = contenders_dpath / expt_suffix
    move_jobs.append({'src': fpath, 'dst': dst_fpath })

dpaths = sorted(set([j['dst'].parent for j in move_jobs]))
for dpath in dpaths:
    dpath.ensuredir()

for job in ub.ProgIter(move_jobs):
    shutil.copy(job['src'], job['dst'])
