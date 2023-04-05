### START HACKS ###

# WE CAN ALSO HACK THIS SYSTEM TO AGREE WITH PREVIOUS IMAGE STRUCTURES

# Put watch where it was in the old container
docker exec -t temp_container \
   ln -s /root/code/watch /watch

# Put models where it was in the old container
docker exec -t temp_container mkdir -p /models
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt /models/
docker exec -t temp_container \
   ln -s /root/data/smart_expt_dvc/models/uky/uky_invariants_2022_03_21/pretext_model/pretext_pca_104.pt /models/

# Make a fake conda command that forwards to pyenv
xdev codeblock '
#!/bin/bash
# A fake conda script that will just forward to pyenv
# shift past all arguments until you find python
# thus if you run conda -n blah blah blah python
# it only runs the python part
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    python)
    break
    ;;
    *)    # unknown option
    shift # past argument
    ;;
esac
done
$@
' > fake_conda
chmod +x fake_conda
docker cp fake_conda temp_container:/bin/conda

### END HACKS ###
