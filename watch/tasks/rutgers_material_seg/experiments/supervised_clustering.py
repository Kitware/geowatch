# from material_seg.datasets import build_dataset
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
from metric_learn import NCA, LMNN, MMC_Supervised, LSML_Supervised
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset

visualize_images = False
coco_fpath = ub.expandpath('/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/material_labels.kwcoco.json')
dset = kwcoco.CocoDataset(coco_fpath)

sampler = ndsampler.CocoSampler(dset)

# # print(sampler)
number_of_timestamps, h, w = 4, 128, 128
window_dims = (number_of_timestamps, h, w) #[t,h,w]
input_dims = (h, w)

# # channels = 'r|g|b|gray|wv1'
channels = 'r|g|b'
# channels = 'gray'
dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
print(dataset.__len__())
loader = dataset.make_loader(batch_size=1)

k = 40
kmeans = KMeans(n_clusters=k, random_state=0, verbose=True)
nca = NCA(max_iter=10, verbose=True, random_state=0)
lmnn = LMNN(k=3, verbose=True, random_state=0, learn_rate=0.01, max_iter=1000)
mmc = MMC_Supervised(max_iter=100, max_proj=10000, verbose=True, random_state=0)
lsml = LSML_Supervised(verbose=True, random_state=0)
n_points = 300

for batch in loader:
    # pdb.set_trace()
    image_data = batch['inputs']['im'].data[0] # [b,c,t,h,w]
    b, c, t, h, w = image_data.shape
    mask_data = batch['label']['class_masks'].data[0] #len(mask_data) = b
    mask_data = torch.stack(mask_data)#.numpy()

    image_show = np.array(image_data).transpose(0, 2, 3, 4, 1) /500 # visualize 0 indexed in batch
    # plt.imshow(image_show)
    # plt.show()
    # image_show = image_show[0,]
    # mask_show = np.array(mask_data) # [b,t,h,w]

    image_data = image_data.view(b, c *t, h *w)
    mask_data = mask_data.view(b, t, h *w).squeeze(0)
    image_data = torch.transpose(image_data,1,2)
    image_data = torch.flatten(image_data,start_dim=0, end_dim=1)

    mask_data = mask_data[0,:]
    # mask_data = mask_data.reshape(h,w)

    # mask_data = mask_data.view(-1,1)
    print(torch.unique(mask_data))
    print(mask_data.shape)
    print(image_data.shape)

    non_bg_indices = np.where(mask_data > 0)[0]
    concrete_indices = np.where(mask_data == 1)[0][:n_points]
    veg_indices = np.where(mask_data == 2)[0][:n_points]
    soil_indices = np.where(mask_data == 3)[0][:n_points]
    water_indices = np.where(mask_data == 4)[0][:n_points]
    material_indices = np.concatenate((concrete_indices, veg_indices, soil_indices, water_indices),axis=0)

    # print(f"concrete: {concrete_indices.shape}, veg: {veg_indices.shape}, soil: {soil_indices.shape}, water: {water_indices.shape}")
    # print(f"concrete: {concrete_indices}, veg: {veg_indices}, soil: {soil_indices}, water: {water_indices}")
    # print(f"concrete: {type(concrete_indices)}, veg: {type(veg_indices)}, soil: {type(soil_indices)}, water: {type(water_indices)}")
    mask_data_filtered = mask_data[material_indices]
    image_data_filtered = image_data[material_indices]

    # nca.fit(image_data[non_bg_indices,:], mask_data[non_bg_indices])
    lmnn.fit(image_data_filtered, mask_data_filtered)
    # mmc.fit(image_data[non_bg_indices,:], mask_data[non_bg_indices])
    # lsml.fit(image_data[non_bg_indices,:], mask_data[non_bg_indices])

    # X_nca = nca.transform(image_data)
    X_lmnn = lmnn.transform(image_data)[material_indices]
    # X_mmc = mmc.transform(image_data)
    # X_lsml = lsml.transform(image_data)
    # print(np.unique(X_lsml))
    # X_lsml_show = X_lsml.reshape(t,h,w,c)/255
    print(X_lmnn.shape)
    # plt.imshow(X_lsml_show[0,:,:,:])
    # plt.show()

    out_feat_embed = TSNE(n_components=2, verbose=True, random_state=0).fit_transform(image_data_filtered)
    # out_feat_embed_nca = TSNE(n_components=2, verbose=True).fit_transform(X_nca)
    out_feat_embed_lmnn = TSNE(n_components=2, verbose=True, random_state=0).fit_transform(X_lmnn)
    # out_feat_embed_mmc = TSNE(n_components=2, verbose=True).fit_transform(X_mmc)
    # out_feat_embed_lsml = TSNE(n_components=2, verbose=True).fit_transform(X_lsml)

    # kmeans.fit(image_data)
    # cluster_centers = kmeans.cluster_centers_
    # cluster_labels = kmeans.labels_
    # y_kmeans = kmeans.predict(image_data)
    # prediction = cluster_labels.reshape(h,w)
    # prediction_no_bg = np.ma.masked_where(prediction==0,prediction)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    # ax3 = fig.add_subplot(2,2,3)
    # ax4 = fig.add_subplot(2,2,4)
    scatter1 = ax1.scatter(out_feat_embed[:,0], out_feat_embed[:,1], c=mask_data_filtered, marker='.', cmap='Set1')
    legend1 = ax1.legend(*scatter1.legend_elements(), loc="lower left", title="Classes")
    ax1.set_title("TSNE of Image Raw Data")
    # ax2.scatter(out_feat_embed_mmc[:,0], out_feat_embed_mmc[:,1], c=mask_data, marker='.', cmap='Set1')
    # ax3.scatter(out_feat_embed_lsml[:,0], out_feat_embed_lsml[:,1], c=mask_data, marker='.', cmap='Set1')
    scatter2 = ax2.scatter(out_feat_embed_lmnn[:,0], out_feat_embed_lmnn[:,1], c=mask_data_filtered, marker='.', cmap='Set1')
    legend2 = ax2.legend(*scatter2.legend_elements(), loc="lower left", title="Classes")
    ax2.set_title("TSNE of Clustered Features")
    plt.show()

    # figure = plt.figure(figsize=(15,15))
    # ax1 = figure.add_subplot(2,5,1)
    # ax2 = figure.add_subplot(2,5,2)
    # ax3 = figure.add_subplot(2,5,3)
    # ax4 = figure.add_subplot(2,5,4)
    # ax5 = figure.add_subplot(2,5,5)
    # ax6 = figure.add_subplot(2,5,6)
    # ax7 = figure.add_subplot(2,5,7)
    # ax8 = figure.add_subplot(2,5,8)
    # ax9 = figure.add_subplot(2,5,9)
    # # ax10 = figure.add_subplot(2,5,10)


    # ax1.imshow(image_show[0,0,:,:,:])
    # ax2.imshow(image_show[0,1,:,:,:])
    # ax3.imshow(image_show[0,2,:,:,:])
    # ax4.imshow(image_show[0,3,:,:,:])
    # # ax1.imshow(image_show[0,0,:,:,:])
    # ax5.imshow(prediction, vmin=0, vmax=k, cmap='Set1')

    # ax6.imshow(mask_data[0,0,:,:])
    # ax7.imshow(mask_data[0,1,:,:])
    # ax8.imshow(mask_data[0,2,:,:])
    # ax9.imshow(mask_data[0,3,:,:])
#     ax4.scatter(x_clusters_scatters, y_clusters_scatters, color=(len(x_clusters_scatters)//channels)*['red','green','blue','yellow','black'])

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
