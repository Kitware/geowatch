# from material_seg.datasets import build_dataset
import os
import sys
import comet_ml
import kwcoco
import ndsampler
import ubelt as ub
import watch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pdb
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.optim as optim
import datetime
import random

from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.models import build_model
import watch.tasks.rutgers_material_seg.utils.utils as utils


class Clusterer(object):
    def __init__(self, loader: torch.utils.data.DataLoader,
                 optimizer: object, scheduler: object,
                 clusterer: str="kmeans", k: int=20, **clusterer_kwargs) -> None:
        """clustering object

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            data to cluster
        optimizer : object
            optimizer used to train pre-trained model
        scheduler : object
            scheduler of pre-trained model
        clusterer : str, optional
            clustering algorithm, by default "kmeans"
        k : int, optional
            number of clusters, by default 20
        """
        self.models = {'kmeans':KMeans, 'spectralcluster':SpectralClustering}
        self.clustering_model = self.models[clusterer](n_clusters=k, **clusterer_kwargs)
        self.clustering_model_tsne = self.models[clusterer](n_clusters=k, **clusterer_kwargs)
        self.use_crf = config['evaluation']['use_crf']
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.cmap = visualization.rand_cmap(nlabels=config['data']['num_classes'] +1, type='bright',
                                            first_color_black=True, last_color_black=True,
                                            bg_alpha=config['visualization']['bg_alpha'],
                                            fg_alpha=config['visualization']['fg_alpha'])

    def multitemporal_cluster(self):
        """Clustering algorithm considers features from multiple timesteps.
        """
        for batch in loader:
            # pdb.set_trace()
            image_data = batch['inputs']['im'].data[0] # [b,c,t,h,w]
            b, c, t, h, w = image_data.shape
            mask_data = batch['label']['class_masks'].data[0] #len(mask_data) = b
            mask_data = torch.stack(mask_data)
            outputs = torch.zeros((b,num_classes,t,h,w))
            for sub_batch in range(b):
                for timestep in range(t):
                    b_t_image_input = image_data[sub_batch,:,timestep,:,:].unsqueeze(0)
                    output = model(b_t_image_input) # [1,num_classes + 1, h, w]
                    output.cpu().detach()
                    outputs[sub_batch,:,timestep,:,:] = output

            image_show = np.array(image_data).transpose(0, 2, 3, 4, 1) /500 # visualize 0 indexed in batch
            # mask_show = np.array(mask_data) # [b,t,h,w]

            b, c, t, h, w = outputs.shape
            image_data = outputs.view(b, c *t, h *w).detach()
            image_data = torch.transpose(image_data,1,2)
            image_data = torch.flatten(image_data,start_dim=0, end_dim=1)

            out_feat_embed = TSNE(n_components=2).fit_transform(image_data)

            self.clustering_model.fit(image_data)
            self.clustering_model_tsne.fit(out_feat_embed)
            # cluster_centers = clustering_model.cluster_centers_
            cluster_labels = self.clustering_model.labels_
            y_kmeans = self.clustering_model_tsne.labels_
            prediction = cluster_labels.reshape(h,w)
            prediction_no_bg = np.ma.masked_where(prediction == 0,prediction)
            # print(f"image_data: {image_data.shape}, mask: {mask_data.shape}")
            # print(f"image min: {image_show.min()}, image max: {image_show.max()}")
            plt.scatter(out_feat_embed[:,0], out_feat_embed[:,1], c=y_kmeans, marker='.', cmap='tab20c')
            # plt.scatter(clustering_model_tsne.cluster_centers_[:, 0], clustering_model_tsne.cluster_centers_[:, 1], c='black', s=200, alpha=0.5);
            plt.show()

            figure = plt.figure(figsize=(15,15))
            ax1 = figure.add_subplot(1,5,1)
            ax2 = figure.add_subplot(1,5,2)
            ax3 = figure.add_subplot(1,5,3)
            ax4 = figure.add_subplot(1,5,4)
            ax5 = figure.add_subplot(1,5,5)
            ax1.imshow(image_show[0,0,:,:,:])
            ax2.imshow(image_show[0,1,:,:,:])
            ax3.imshow(image_show[0,2,:,:,:])
            ax4.imshow(image_show[0,3,:,:,:])
            ax5.imshow(prediction, vmin=0, vmax=k,cmap='tab20c')
            plt.show()

            if visualize_images:
                mask_show = mask_show[0] # [b,t,h,w]
                image_show = image_show[0]
                figure = plt.figure(figsize=(10,10))
                axes = {}
                for i in range(1,2 *t +1):
                    axes[i] = figure.add_subplot(2,t,i)
                for key in axes.keys():
                    if key <= t:
                        axes[key].imshow(image_show[key -1,:,:,:])
                    else:
                        axes[key].imshow(mask_show[key -t -1,:,:],vmin=-1, vmax=7)
                figure.tight_layout()
                plt.show()

if __name__ == "__main__":

    project_root = "/home/native/projects/watch/watch/tasks/rutgers_material_seg/"
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    # main_config_path = f"{os.getcwd()}/configs/main.yaml"
    main_config_path = f"{project_root}/configs/main.yaml"
    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"

    experiment_config = utils.config_parser(experiment_config_path,experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    project_name = f"{project_root[-3]}_{project_root[-1]}"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
    experiment_name = f"SMART_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
    experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
                                     project_name=project_name,
                                     workspace=config['cometml']['workspace'],
                                     display_summary_level=0)
    experiment.set_name(experiment_name)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.set_default_dtype(torch.float32)

    # device_cpu = torch.device('cpu')
    # print(config['data']['image_size'])
    device_ids = list(range(torch.cuda.device_count()))
    config['device_ids'] = device_ids
    gpu_devices = ','.join([str(id) for id in device_ids])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = torch.device('cuda')

    config['devices_used'] = gpu_devices
    experiment.log_asset_data(config)
    experiment.log_text(config)
    experiment.log_parameters(config)
    experiment.log_parameters(config['training'])
    experiment.log_parameters(config['evaluation'])
    experiment.log_parameters(config['visualization'])

    channels = 'r|g|b'
    number_of_timestamps, h, w = 4, 128, 128
    window_dims = (number_of_timestamps, h, w) #[t,h,w]
    input_dims = (h, w)

    coco_fpath = ub.expandpath(config['data'][config['location']]['coco_json'])
    dset = kwcoco.CocoDataset(coco_fpath)

    sampler = ndsampler.CocoSampler(dset)
    # # channels = 'r|g|b|gray|wv1'
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    train_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

    model = build_model(model_name = config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'] +1,
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'])

    model = build_model(model_name = config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'] +1,
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)
    model.to(device)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    optimizer = optim.SGD(model.parameters(),
                          lr=config['training']['learning_rate'],
                          momentum=config['training']['momentum'],
                          weight_decay=config['training']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_dataloader),
                                                     eta_min = config['training']['learning_rate'])

    if config['training']['resume'] != False:

        if os.path.isfile(config['training']['resume']):
            checkpoint = torch.load(config['training']['resume'])
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"loaded model from {config['training']['resume']}")
        else:
            print("no checkpoint found at {}".format(config['training']['resume']))
            exit()

    clusterer = Clusterer(model,
                      train_dataloader,
                      train_dataloader,
                      config['training']['epochs'],
                      optimizer,
                      scheduler,
                      test_loader=train_dataloader
                      )
    train_losses, val_losses, mean_ious_val = clusterer.multitemporal_cluster(experiment)
