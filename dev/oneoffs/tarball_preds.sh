cd ~/data/dvc-repos/smart_watch_dvc-hdd/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976

7z a models---fusion---eval3_candidates---pred-Drop3_SpotCheck_V323---pred_Drop3_SpotCheck_V323_epoch=18-step=12976---Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco.7z Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco

eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976


#for c in method.children():
#    print(c.__class__.__name__)
#    try:
#        repr(c)
#    except Exception:
#        print("Error here")
#        pass


#        python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation             --gpus="$TMUX_GPUS"             --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt"             --test_dataset="$VALI_FPATH"             --run=1 --skip_existing=False --backend=tmux --enable_track=True --enable_iarpa_eval=True --virtualenv_cmd="conda activate watch"


TMUX_GPUS=0,1
DVC_DPATH=/data/projects/smart/smart_watch_dvc
VALI_FPATH=/data/projects/smart/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_train.kwcoco.json
python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation             --gpus="$TMUX_GPUS"             --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt"             --test_dataset="$VALI_FPATH"             --run=1 --skip_existing=False --backend=tmux --enable_track=True --enable_iarpa_eval=True --virtualenv_cmd="conda activate watch"                                                                        


DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
PRED_KWCOCO_DPATH=$DVC_DPATH/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_train.kwcoco/predcfg_1c530993/pred.kwcoco.json

smartwatch visualize "$PRED_KWCOCO_DPATH" --channels="salient" --animate=True --workers=8 --draw_imgs=False --draw_anns=True --only_boxes=True
