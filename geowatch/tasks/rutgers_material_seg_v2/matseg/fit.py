import os
import json

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from geowatch.tasks.rutgers_material_seg_v2.matseg.models import build_model
from geowatch.tasks.rutgers_material_seg_v2.matseg.datasets import build_dataset
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import generate_image_slice_object
from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import get_labelbox_material_labels, MATERIAL_TO_MATID


def fit_model(cfg: DictConfig, overwrite_exp_dir=None) -> str:

    # Get experiment directory.
    if overwrite_exp_dir is None:
        exp_dir = os.getcwd()
    else:
        exp_dir = overwrite_exp_dir

    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width, cfg.crop_stride)

    if cfg.dataset.kwargs is None:
        cfg.dataset.kwargs = {}

    # Get material labels.
    mat_labels, mat_dist = get_labelbox_material_labels(cfg.refresh_labels, cfg.lb_project_id)

    train_dataset = build_dataset(cfg.dataset.name,
                                  mat_labels,
                                  slice_params,
                                  'all',
                                  sensors=cfg.dataset.sensors,
                                  channels=cfg.dataset.channels,
                                  resize_factor=cfg.resize_factor,
                                  refresh_labels=cfg.refresh_labels,
                                  **cfg.dataset.kwargs)
    # valid_dataset = build_dataset(cfg.dataset.name,
    #                               mat_labels,
    #                               slice_params,
    #                               'all',
    #                               sensors=cfg.dataset.sensors,
    #                               channels=cfg.dataset.channels,
    #                               resize_factor=cfg.resize_factor,
    #                               **cfg.dataset.kwargs)

    # Create dataloaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.n_workers)
    # valid_loader = DataLoader(valid_dataset,
    #                           batch_size=cfg.batch_size,
    #                           shuffle=False,
    #                           num_workers=cfg.n_workers)

    # Create model.
    if cfg.model.kwargs is None:
        cfg.model.kwargs = {}
    model = build_model(
        mat_dist,
        network_name=cfg.model.architecture,
        encoder_name=cfg.model.encoder,
        in_channels=train_dataset.n_channels,
        out_channels=len(MATERIAL_TO_MATID.keys()),
        loss_mode=cfg.model.loss_mode,
        optimizer_mode=cfg.model.optimizer_mode,
        class_weight_mode=cfg.model.class_weight_mode,
        lr=cfg.model.lr * cfg.batch_size * cfg.batch_size_lr_scaling,
        wd=cfg.model.wd,
        pretrain=cfg.model.pretrain,
        lr_scheduler_mode=cfg.model.lr_scheduler_mode,
        # log_image_iter=cfg.log_image_iter,
        to_rgb_fcn=train_dataset.to_RGB,
        **cfg.model.kwargs)

    # Create logger.
    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(exp_dir, 'tensorboard_logs'))

    # Train model.
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(exp_dir, 'checkpoints'),
                                          save_top_k=cfg.save_topk_models,
                                          mode='max',
                                          monitor="train_F1Score",
                                          filename="model-{epoch:02d}-{train_F1Score:.5f}")
    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        accelerator="gpu",
        devices=1,
        default_root_dir=exp_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        profiler=cfg.profiler,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        #  auto_lr_find=False,
        strategy=cfg.strategy,
        precision=16)
    # trainer.tune(model, train_loader, train_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=None)

    if cfg.test_model:
        trainer.test(model, train_loader, trainer.checkpoint_callback.best_model_path)

        # Save test results to experiment folder.
        test_metrics = model.test_metrics.compute()
        save_test_results = {
            'acc': test_metrics['test_Accuracy'].item(),
            'iou': test_metrics['test_JaccardIndex'].item(),
            'f1_score': test_metrics['test_F1Score'].item()
        }
        test_save_path = os.path.join(os.getcwd(), 'test_metrics.json')
        with open(test_save_path, 'w') as f:
            json.dump(save_test_results, f, indent=4)

        # Save region analysis as well.
        analysis_dpath = os.path.join(os.getcwd(), 'analysis')
        os.makedirs(analysis_dpath, exist_ok=True)
        with open(os.path.join(analysis_dpath, 'region_metrics.json'), 'w') as f:
            json.dump(model.region_cm_metrics, f, indent=4)

        # Save the CSV file for each confusion matrix.
        for region_id, region_cm in model.region_cms.items():
            region_cm.save_csv(os.path.join(analysis_dpath, f'region_{region_id}_cm.csv'))

    # Return best model path.
    return trainer.checkpoint_callback.best_model_path


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def config_fit_func(cfg: DictConfig):
    fit_model(cfg)


if __name__ == '__main__':
    config_fit_func()
