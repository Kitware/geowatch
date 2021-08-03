
import kwcoco
import ndsampler
import ubelt as ub
import watch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pdb
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset

visualize_images=False
coco_fpath = ub.expandpath('/media/native/data/data/smart_watch_dvc/drop0_aligned_msi/material_labels.kwcoco.json')
dset = kwcoco.CocoDataset(coco_fpath)

sampler = ndsampler.CocoSampler(dset)

# # print(sampler)
number_of_timestamps, h, w = 5, 128, 128
window_dims = (number_of_timestamps, h, w) #[t,h,w]
input_dims = (h, w)

# # channels = 'r|g|b|gray|wv1'
# channels = 'r|g|b'
channels = 'red|green|blue|nir|swir16|swir22'
# channels = 'nir'``
dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
loader = dataset.make_loader(batch_size=1)

k = 80
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans_tsne = KMeans(n_clusters=k, random_state=0)

for batch_index, batch in enumerate(loader):
    # pdb.set_trace()
    image_data = batch['inputs']['im'].data[0] # [b,c,t,h,w]
    print(f"image_data min: {image_data.min()}, image_data max: {image_data.max()}")
    b, c, t, h, w = image_data.shape
    mask_data = batch['label']['class_masks'].data[0] #len(mask_data) = b
    mask_data = torch.stack(mask_data)

    image_show = np.array(image_data).transpose(0, 2, 3, 4, 1)#/50000 # visualize 0 indexed in batch
    image_show = image_show/image_show.max()
    print(f"image min: {image_show.min()}, image max: {image_show.max()}")
    # print(image_show.shape)
    # image_show = image_show[0,0,:,:,:]
    # plt.imshow(image_show[0,0,:,:,:3])
    # plt.show()
    # mask_show = np.array(mask_data) # [b,t,h,w]

    image_data = image_data.view(b, c*t, h*w)
    # print(image_data.shape)
    image_data = torch.transpose(image_data,1,2)
    # print(image_data.shape)
    image_data = torch.flatten(image_data,start_dim=0, end_dim=1)
    # print(image_data.shape)
    # image_data = torch.transpose(image_data,0,1)
    # print(image_data.shape)
    out_feat_embed = TSNE(n_components=2).fit_transform(image_data)
    kmeans_tsne.fit(out_feat_embed)
    # data = image_data
    # data = out_feat_embed
    kmeans.fit(image_data)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    # y_kmeans = kmeans.predict(data)
    y_kmeans_tse = kmeans_tsne.predict(out_feat_embed)
    # print(cluster_centers)
    # print(cluster_labels)
    prediction = cluster_labels.reshape(h,w)
    prediction_no_bg = np.ma.masked_where(prediction==0,prediction)
    # print(f"image_data: {image_data.shape}, mask: {mask_data.shape}")
    # print(f"image min: {image_show.min()}, image max: {image_show.max()}")
    plt.scatter(out_feat_embed[:,0], out_feat_embed[:,1], c=y_kmeans_tse, marker='.', cmap='tab20c')
    plt.scatter(kmeans_tsne.cluster_centers_[:, 0], kmeans_tsne.cluster_centers_[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

    figure = plt.figure(figsize=(15,15))
    ax1 = figure.add_subplot(1,5,1)
    ax2 = figure.add_subplot(1,5,2)
    ax3 = figure.add_subplot(1,5,3)
    ax4 = figure.add_subplot(1,5,4)
    ax5 = figure.add_subplot(1,5,5)
    # ax6 = figure.add_subplot(2,4,2)
    # ax7 = figure.add_subplot(2,4,1)
    # ax8 = figure.add_subplot(2,4,2)

    ax1.imshow(image_show[0,0,:,:,:3])
    ax2.imshow(image_show[0,1,:,:,:3])
    ax3.imshow(image_show[0,2,:,:,:3])
    ax4.imshow(image_show[0,3,:,:,:3])
    # ax1.imshow(image_show[0,0,:,:,:])
    ax5.imshow(prediction, vmin=0, vmax=k, cmap='Set1', interpolation='nearest')
    # ax3.imshow(image_show[0,0,:,:,:])
    # ax3.imshow(prediction_no_bg, alpha=0.6, cmap='Set1')
#     ax4.scatter(x_clusters_scatters, y_clusters_scatters, color=(len(x_clusters_scatters)//channels)*['red','green','blue','yellow','black'])

    plt.show()

    # if visualize_images:
    #     mask_show = mask_show[0] # [b,t,h,w]
    #     image_show = image_show[0]
    #     figure = plt.figure(figsize=(10,10))
    #     axes = {}
    #     for i in range(1,2*t+1):
    #         axes[i] = figure.add_subplot(2,t,i)
    #     for key in axes.keys():
    #         if key <= t:
    #             axes[key].imshow(image_show[key-1,:,:,:])
    #         else:
    #             axes[key].imshow(mask_show[key-t-1,:,:],vmin=-1, vmax=7)
    #     figure.tight_layout()
    #     plt.show()
