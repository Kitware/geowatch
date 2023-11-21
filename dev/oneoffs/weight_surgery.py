# from torch_liberator.xpu_device import XPU
# xpu = XPU.coerce('cpu')

# state = torch.load(fpath)

import torch
import ubelt as ub
fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_4/checkpoints/epoch=56-step=3648-val_loss=3.596.ckpt.ckpt'

fpath = '/home/joncrall/remote/namek/data/dvc-repos/xview2/training/namek/joncrall/xview2/runs/XView2Baseline_v2/lightning_logs/version_0/checkpoints/last.ckpt'

state = torch.load(fpath)
state_dict = state['state_dict']

bad_idxs = [
    state['hyper_parameters']['classes'].index('ignore'),
    state['hyper_parameters']['classes'].index('un-classified'),
]

# pos_idx = state['hyper_parameters']['classes'].index('positive')
# ig_idx = state['hyper_parameters']['classes'].index('ignore')
# unkn_idx = state['hyper_parameters']['classes'].index('Unknown')
# bad_idxs = [pos_idx, ig_idx, unkn_idx]

# Set the bias for the classes we dont want to predict to be very negative
state_dict['heads.class.hidden.output.bias'][bad_idxs] = -200


new_fpath = ub.Path(fpath).augment(stemsuffix='_weight_hacked')
torch.save(state, new_fpath)

# state_dict['heads.class.hidden.output.weight'].shape
# print(state['hyper_parameters']['classes'])
# print('pos_idx = {}'.format(ub.urepr(pos_idx, nl=1)))
# print('ig_idx = {}'.format(ub.urepr(ig_idx, nl=1)))
# print('unkn_idx = {}'.format(ub.urepr(unkn_idx, nl=1)))
# act_con = state['hyper_parameters']['classes'].index('Active Construction')
# sp_con = state['hyper_parameters']['classes'].index('Site Preparation')
# bg = state['hyper_parameters']['classes'].index('background')
# print('act_con = {}'.format(ub.urepr(act_con, nl=1)))
# state_dict['heads.class.hidden.output.weight'][unkn_idx]
# state_dict['heads.class.hidden.output.weight'][act_con]

"""
HACK_SAVE_ANYWAY=1 python -m geowatch.mlops.repackager \
    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/lightning_logs/version_4/checkpoints/epoch=56-step=3648-val_loss=3.596.ckpt_weight_hacked.ckpt

        /home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/lightning_logs/version_1/checkpoints/epoch=44-step=3870-val_loss=5.761.ckpt_weight_hacked.ckpt

"""

# ls /home/joncrall/remote/namek/data/dvc-repos/xview2/training/namek/joncrall/xview2/runs/XView2Baseline_v2/lightning_logs/version_0/checkpoints/last.ckpt
