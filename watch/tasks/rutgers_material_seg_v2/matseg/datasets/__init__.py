from watch.tasks.rutgers_material_seg_v2.matseg.datasets.lb_materials import Material_LB_Dataset
from watch.tasks.rutgers_material_seg_v2.matseg.datasets.lb_materials_multi_sensor import Material_Multi_Sensor_Dataset

DATASETS = {'lb_materials': Material_LB_Dataset, 'lb_ms_materials': Material_Multi_Sensor_Dataset}


def build_dataset(dset_name, mat_labels, split, slice_params, sensors, **kwargs):

    try:
        dataset = DATASETS[dset_name](mat_labels, split, slice_params, sensors=sensors, **kwargs)
    except KeyError:
        raise KeyError(
            f'DATASETS dictionary does not contain a dataset class for dataset name "{dset_name}"')
    return dataset