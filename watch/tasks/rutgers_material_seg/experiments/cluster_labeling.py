# flake8: noqa
"""
This method uses K-Means clusters as region proposals,
which this tool allows you to label. Note that this tool:
    - Does not have "free brush" tool. It can only label clusters.
    - The processing time of each region proposal depends on the
    number of clusters, timesteps, and channels. It may take
    up to 2 minutes with some configurations.
"""

import sys
from PyQt5.QtGui import QPixmap, QImage  # NOQA
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QSizePolicy,   # NOQA
                             QDoubleSpinBox, QLabel, QCheckBox, QMainWindow,     # NOQA
                             QGridLayout, QComboBox, QTextBrowser, QLineEdit, QInputDialog)    # NOQA
from PyQt5.QtCore import Qt  # NOQA
from PyQt5 import QtCore
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ubelt as ub
import kwcoco
import ndsampler
# import numpy as np
# import watch
# from watch.utils.util_norm import *
# import matplotlib.pyplot as plt
# from tqdm import tqdm
import torch
# import pdb
from sklearn.cluster import KMeans
# from skimage.transform import resize
# from sklearn.manifold import TSNE
# import scipy
# from PIL import Image
# from PIL.ImageQt import ImageQt
import qimage2ndarray
import cmapy
# import random
import kwimage
from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset


class Window(QMainWindow):
    def __init__(self, dataset, dset, resume='', save_path='', k=255):
        """Main interactive window

        Parameters
        ----------
        dataset : kwcoco
            KWCOCO dataset to use (with sampler)
        dset : kwcoco
            KWCOCO dataset to use (before sampler)
        resume : str, optional
            path to dataset to continue annotating, by default ''
        save_path : str, optional
            where to save the newly annotated dataset, by default ''
        k : int, optional
            number of clusters for K-Means algorithm, by default 255
        """
        super().__init__()
        self.image_counter = 0
        self.dataset = dataset
        self.channels, self.timesteps, self.im_width, self.im_height = self.dataset[self.image_counter]['inputs']['im'].data.shape
        self.dset = dset
        self.save_path = save_path
        self.k = k
        self.seen_labels, self.class_labels_all = [], []
        self.class_labels_pairs = {}
        self.class_label_with = "Concrete"
        self.class_label_to_index = {"Nothing": 0, "Concrete": 1, "Vegetation": 2, "Soil": 3, "Water": 4}
        if len(resume) > 1:
            self.material_dset = kwcoco.CocoDataset(resume)
            self.image_counter = int(len(list(self.material_dset.index.imgs.keys())) / self.timesteps) + 1
        else:
            self.material_dset = kwcoco.CocoDataset()
            for key, value in self.class_label_to_index.items():
                if key != 0:
                    self.material_dset.add_category(name=key, id=value)

        self.width, self.height = self.dataset[self.image_counter]['tr'].data['space_dims']

        self.scale_factor = 1
        self.scaled_height, self.scaled_width = int(self.height * self.scale_factor), int(self.width * self.scale_factor)
        self.current_mask = np.zeros((self.width, self.height)).astype(np.uint8)
        self.separable_current_mask = np.zeros((len(list(self.class_label_to_index.keys())), self.width, self.height)).astype(np.uint8)
        self.widget = QWidget()
        self.widget.keyPressEvent = self.keyPressEvent
        self.load_images(index=self.image_counter)

        self.label_img = QLabel(self.widget)
        self.image = QPixmap(self.qImg)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20, 128, self.height, self.width)
        self.label_img.mousePressEvent = self.getImagePixel

        self.label_prediction = QLabel(self.widget)
        self.prediction = QPixmap(self.qPred)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.width + 20 + 120, 128, self.height, self.width)
        self.label_prediction.mousePressEvent = self.getMaskPixel

        self.label_mask = QLabel(self.widget)
        self.mask = QPixmap(self.qMask)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3 * self.width + 20, 128, self.height, self.width)

        self.output_textbox = QTextBrowser(self.widget)
        self.output_textbox.setGeometry(QtCore.QRect(20, self.height + 256 + 128 + 50, 100, 100))
        self.output_textbox.setObjectName("Current Clusters selected")

        self.output_textbox_name = QTextBrowser(self.widget)
        self.output_textbox_name.setGeometry(QtCore.QRect(140, self.height + 256 + 128 + 50, 100, 100))
        self.output_textbox_name.setObjectName("Labels of Selected Clusters")

        self.remove_cluster = QLineEdit(self.widget)
        self.remove_cluster.setObjectName("Cluster Removal Tool")
        self.remove_cluster.move(256, self.height + 256 + 128 + 50)
        self.remove_cluster.resize(100, 40)
        self.remove_cluster.textChanged.connect(self.textchanged)

        self.image_update = QPushButton(self.widget)
        self.image_update.setText("Previous Image")
        self.image_update.move(20, 20)
        self.image_update.clicked.connect(self.previous_image)

        self.image_update = QPushButton(self.widget)
        self.image_update.setText("Next Image")
        self.image_update.move(140, 20)
        self.image_update.clicked.connect(self.update_image)

        self.class_label = QComboBox(self.widget)
        self.class_label.addItems(["Nothing", "Concrete", "Vegetation", "Soil", "Water"])
        self.class_label.move(300, 20)
        self.class_label.currentIndexChanged.connect(self.class_label_selection)

        self.show_current_mask = QPushButton(self.widget)
        self.show_current_mask.setText("Show Current Mask")
        self.show_current_mask.move(500, 20)
        self.show_current_mask.clicked.connect(self.show_current_mask_clicked)

        self.increase_size = QPushButton(self.widget)
        self.increase_size.setText("Increase Image Size")
        self.increase_size.move(700, 20)
        self.increase_size.clicked.connect(self.increase_images_size)

        self.decrease_size = QPushButton(self.widget)
        self.decrease_size.setText("Decrease Image Size")
        self.decrease_size.move(900, 20)
        self.decrease_size.clicked.connect(self.decrease_images_size)

        self.timestamp_label = QComboBox(self.widget)
        self.timestamp_label.addItems([str(x) for x in range(self.timesteps)])
        self.timestamp_label.move(1100, 20)
        self.timestamp_label.currentIndexChanged.connect(self.change_inter_batch_image)

        self.gotoimage_label = QComboBox(self.widget)
        self.gotoimage_label.addItems([str(x) for x in range(self.dataset.__len__())])
        self.gotoimage_label.move(1200, 20)
        self.gotoimage_label.currentIndexChanged.connect(self.go_to_image)

        self.label_mask_title = QLabel(self.widget)
        self.label_mask_title.setText("Intermediate Mask")
        self.label_mask_title.move(2 * self.width + 256, 100)

        self.label_img_title = QLabel(self.widget)
        self.label_img_title.setText(f"Image {self.image_counter}")
        self.label_img_title.move(256, 100)

        self.label_prediction_title = QLabel(self.widget)
        self.label_prediction_title.setText("Clustering Prediction")
        self.label_prediction_title.move(self.width + 256, 100)

        self.remove_cluster_title = QLabel(self.widget)
        self.remove_cluster_title.setText("Write cluster to remove from the mask")
        self.remove_cluster_title.move(256, self.height + 256 + 128)

        self.output_textbox_title = QLabel(self.widget)
        self.output_textbox_title.setText("Current Clusters")
        self.output_textbox_title.move(20, self.height + 256 + 128)

        self.output_textbox_name_title = QLabel(self.widget)
        self.output_textbox_name_title.setText("Clusters Labels")
        self.output_textbox_name_title.move(140, self.height + 256 + 128)

        self.widget.setGeometry(50, 50, 1800, 800)
        self.widget.show()

    def change_inter_batch_image(self):
        """view a different image within the batch
        """
        image_data = self.dataset[self.image_counter]['inputs']['im'].data  # [b,c,t,h,w]
        image_data = image_data[:3, :, :self.width, :self.height]  # .copy()
        c, t, h, w = image_data.shape
        # random_t = random.randrange(t)
        self.t_selection = int(self.timestamp_label.currentText())
        # print(random_t)
        image_show = np.array(image_data[:, self.t_selection, :, :]).transpose(1, 2, 0).copy()  # visualize 0 indexe
        image_min = np.min(image_show)
        image_max = np.max(image_show)
        self.image_show = (image_show - image_min) / (image_max - image_min)
        self.qimage = qimage2ndarray.array2qimage(self.image_show * 255)  # .scaled(self.height, self.width)
        self.qImg = QPixmap(self.qimage)  # .scaled(256,256)

        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        # self.label_img.setGeometry(20,128,self.scaled_height, self.scaled_width)

    def increase_images_size(self):
        """enlarge size of displayed image and clusters
        """
        self.scale_factor += 0.25
        self.scaled_height, self.scaled_width = int(self.scale_factor * self.height), int(self.scale_factor * self.width)

        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20, 128, self.scaled_height, self.scaled_width)

        self.prediction = QPixmap(self.qPred).scaled(self.scaled_height, self.scaled_width)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.scaled_width + 20 + 70 * (1 + self.scale_factor), 128, self.scaled_height, self.scaled_width)

        # self.mask = QPixmap(self.qMask).scaled(self.scaled_height, self.scaled_width)
        # self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3 * self.scaled_width, 128, self.scaled_height, self.scaled_width)

    def decrease_images_size(self):
        """reduce size of displayed image and clusters
        """
        self.scale_factor -= 0.25
        self.scaled_height, self.scaled_width = int(self.scale_factor * self.height), int(self.scale_factor * self.width)

        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20, 128, self.scaled_height, self.scaled_width)

        self.prediction = QPixmap(self.qPred).scaled(self.scaled_height, self.scaled_width)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.scaled_width + 20 + 50 * (1 + self.scale_factor), 128, self.scaled_height, self.scaled_width)

        # self.mask = QPixmap(self.qMask).scaled(self.scaled_height, self.scaled_width)
        # self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3 * self.scaled_width, 128, self.scaled_height, self.scaled_width)

    def keyPressEvent(self, event):
        """allow change class to label with key press

        Parameters
        ----------
        event : int
            key press code
        """

        keys_to_class_dict = {48: "Nothing", 49: "Concrete", 50: "Vegetation", 51: "Soil", 52: "Water"}
        if event.key() in keys_to_class_dict.keys():
            self.class_label_with = keys_to_class_dict[event.key()]
            self.class_label.setCurrentText(keys_to_class_dict[event.key()])

    def show_current_mask_clicked(self):
        current_mask_no_bg = np.ma.masked_where(self.current_mask == 0, self.current_mask)
        plt.imshow(self.image_show)
        plt.imshow(current_mask_no_bg, alpha=0.6, cmap='tab20')
        plt.show()

    def finalize_mask(self):
        """Process current mask and add it to kwcoco dataset.
        """
        gids = self.dataset[self.image_counter]['tr'].data['gids']
        # widths_list = [x['width'] for x in self.dset.index.imgs[gids[0]]['auxiliary']]
        # max_dim_index = widths_list.index(max(widths_list))
        # max_dim_index = self.dset.index.imgs[gids[0]]['auxiliary'].index(max(self.dset.index.imgs[gids[0]]['auxiliary']))

        # im_space_width, im_space_height = self.dset.index.imgs[gids[0]]['auxiliary'][max_dim_index]['width'], self.dset.index.imgs[gids[0]]['auxiliary'][max_dim_index]['height']
        # transform = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
        # type="polygon"
        for i in range(1, len(list(self.class_label_to_index.keys()))):
            if len(np.unique(self.separable_current_mask[i, :, :])) > 1:

                binary_mask = self.separable_current_mask[i, :, :]
                binary_mask = kwimage.Mask(binary_mask, format='c_mask')
                # binary_coco = binary_polygon.to_coco(style='new')
                # binary_segmentation = kwimage.Segmentation.coerce(binary_polygon)#.to_coco(style="new")
                for gid in gids:
                    image_dict =  self.dset.index.imgs[gid]
                    im_space_height, im_space_width = image_dict['height'], image_dict['width']  # NOQA
                    img_to_vid_transform = image_dict['warp_img_to_vid']['matrix']
                    img_to_vid_transform_npy = np.array(img_to_vid_transform)
                    img_to_vid_transform_inv_npy = np.linalg.inv(img_to_vid_transform_npy)

                    binary_polygon = binary_mask.to_multi_polygon()
                    mask = binary_polygon.warp(img_to_vid_transform_inv_npy)  # , output_dims=(im_space_height, im_space_width))
                    # binary_mask_2 = binary_mask.warp(img_to_vid_transform_inv_npy, output_dims=(im_space_height, im_space_width))
                    # binary_polygon_2 = binary_mask.to_multi_polygon()
                    self.material_dset.add_annotation(image_id=gid,
                                                      category_id=i,
                                                      bbox=ub.peek(mask.bounding_box().to_coco()),
                                                      segmentation=mask.to_coco(style="new")
                                                      # bbox=ub.peek(binary_polygon.bounding_box().to_coco()),
                                                      # segmentation=binary_polygon.to_coco(style="new")
                                                      )

        self.material_dset.validate()
        self.material_dset._check_integrity()
        self.material_dset.dump(self.save_path, newlines=True)

    def class_label_selection(self):
        """Select class to label.
        """
        self.class_label_with = self.class_label.currentText()

    def textchanged(self, text):
        """Remove a cluster
        """
        num, ok = QInputDialog.getInt(self, "Select a cluster to remove", "enter a number")
        if ok and num:
            if num in self.seen_labels:
                xs, ys = np.where(self.prediction_show == num)
                self.current_mask[xs, ys] = 0
                self.update_mask()
                self.seen_labels.remove(num)

    def getImagePixel(self, event):
        x = int(event.pos().x() // self.scale_factor)
        y = int(event.pos().y() // self.scale_factor)
        self.value = self.prediction_show[y, x]
        xs, ys = np.where(self.prediction_show == self.value)

        self.current_mask[xs, ys] = self.class_label_to_index[self.class_label_with]
        self.separable_current_mask[self.class_label_to_index[self.class_label_with], xs, ys] = 1  # need to account for removed pixels!

        non_label_indices = np.array(list(set(list(self.class_label_to_index.values())) - set([self.class_label_to_index[self.class_label_with]])))
        for label_index in non_label_indices:
            self.separable_current_mask[label_index, xs, ys] = 0

        self.update_mask()
        self.class_labels_pairs[self.value] = self.class_label_with
        self.seen_labels.append(self.value)
        self.output_textbox.append(str(self.value))
        self.output_textbox_name.append(str(self.class_label_with))

    def getMaskPixel(self, event):
        x = int(event.pos().x() // self.scale_factor)
        y = int(event.pos().y() // self.scale_factor)
        self.value = self.prediction_show[y, x]
        xs, ys = np.where(self.prediction_show == self.value)

        self.current_mask[xs, ys] = self.class_label_to_index[self.class_label_with]
        self.separable_current_mask[self.class_label_to_index[self.class_label_with], xs, ys] = 1
        non_label_indices = np.array(list(set(list(self.class_label_to_index.values())) - set([self.class_label_to_index[self.class_label_with]])))
        for label_index in non_label_indices:
            self.separable_current_mask[label_index, xs, ys] = 0

        self.update_mask()
        self.class_labels_pairs[self.value] = self.class_label_with
        self.seen_labels.append(self.value)
        self.output_textbox.append(str(self.value))
        self.output_textbox_name.append(str(self.class_label_with))

    def update_mask(self):
        self.current_mask_cmap = (self.current_mask * 20).astype(np.uint8)
        self.current_mask_cmap = cv2.applyColorMap(self.current_mask_cmap, cmapy.cmap('viridis'))
        self.qmask = qimage2ndarray.array2qimage(self.current_mask_cmap).scaled(self.height, self.width)
        self.mask = QPixmap(self.qmask)  # .scaled(256,256)
        # self.image = QPixmap(self.qImg)
        self.label_mask.setPixmap(self.mask)

    def previous_image(self):
        pass

    def go_to_image(self):
        plt.close()
        self.finalize_mask()
        self.class_labels_all.append(self.class_labels_pairs)
        self.class_labels_pairs = {}
        self.seen_labels = []
        self.output_textbox.clear()
        self.output_textbox_name.clear()
        self.image_counter = int(self.gotoimage_label.currentText())
        self.label_img_title.setText(f"Image {self.image_counter}")
        self.width, self.height = self.dataset[self.image_counter]['tr'].data['space_dims']
        self.scale_factor = 1
        self.current_mask = np.zeros((self.width, self.height)).astype(np.uint8)
        self.separable_current_mask = np.zeros((len(list(self.class_label_to_index.keys())), self.width, self.height)).astype(np.uint8)
        self.load_images(index=self.image_counter)
        # print("updated image and prediction")

        self.image = QPixmap(self.qImg)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20, 128, self.height, self.width)

        self.prediction = QPixmap(self.qPred)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.width + 20 + 50, 128, self.height, self.width)

        self.mask = QPixmap(self.qMask)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(2 * self.width + 128, 128, self.height, self.width)

    def update_image(self):
        plt.close()
        self.finalize_mask()
        self.class_labels_all.append(self.class_labels_pairs)
        # print(self.class_labels_all)
        self.class_labels_pairs = {}
        self.seen_labels = []
        self.output_textbox.clear()
        self.output_textbox_name.clear()
        self.image_counter += 1
        print(f"current iteration: {self.image_counter} \n width, height: {self.dataset[self.image_counter]['tr'].data['space_dims']}")
        self.label_img_title.setText(f"Image {str(self.image_counter)}")
        self.width, self.height = self.dataset[self.image_counter]['tr'].data['space_dims']
        self.scale_factor = 1
        self.current_mask = np.zeros((self.width, self.height)).astype(np.uint8)
        self.separable_current_mask = np.zeros((len(list(self.class_label_to_index.keys())), self.width, self.height)).astype(np.uint8)
        # self.width_factor, self.height_factor = self.vis_width/self.width, self.vis_height/self.height
        self.load_images(index=self.image_counter)
        # print("updated image and prediction")

        self.image = QPixmap(self.qImg)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20, 128, self.height, self.width)

        self.prediction = QPixmap(self.qPred)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.width + 20 + 50, 128, self.height, self.width)

        self.mask = QPixmap(self.qMask)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(2 * self.width + 128, 128, self.height, self.width)

    def load_images(self, index):
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        image_data = self.dataset[index]['inputs']['im'].data  # [b,c,t,h,w]
        # print(f"image min:{image_data.min()} max: {image_data.max()}")
        gids = self.dataset[self.image_counter]['tr'].data['gids']
        for gid in gids:
            image_dict =  self.dset.index.imgs[gid]
            img_to_vid_transform = image_dict['warp_img_to_vid']['matrix']  # NOQA
            video_dict = self.dset.index.videos[image_dict['video_id']]
            if gid not in self.material_dset.index.imgs.keys():
                self.material_dset.add_image(**image_dict)

        if image_dict['video_id'] not in self.material_dset.index.videos.keys():
            self.material_dset.add_video(**video_dict)

        image_data = image_data[:, :, :self.width, :self.height]  # /20000#.copy()
        print(f"before image_data shape:{image_data.shape}")
        c, t, h, w = image_data.shape
        image_show = np.array(image_data[:3, 1, :, :]).transpose(1, 2, 0).copy()  # visualize 0 indexe
        image_min = np.min(image_show)
        image_max = np.max(image_show)
        self.image_show = (image_show - image_min) / (image_max - image_min)

        # print(f"image min:{self.image_show.min()} max: {self.image_show.max()}")
        # print(f"image counts: {np.unique(image_data, return_counts=True)}")
        # plt.imshow(self.image_show)
        # plt.show()

        image_data = image_data.contiguous().view(c, t, h * w)
        image_data = torch.transpose(image_data, 0, 2)
        image_data = torch.flatten(image_data, start_dim=1, end_dim=2)
        print(f"after image_data shape:{image_data.shape}")
        # print(f"image min:{image_data.min()} max: {image_data.max()}")
        kmeans.fit(image_data)
        cluster_labels = kmeans.labels_
        self.prediction_show = (cluster_labels.reshape(h, w)).astype(np.uint8)  # *6

        # self.fig = plt.figure()
        # ax = self.fig.add_subplot(1,1,1)
        # ax.imshow(self.prediction_show, cmap='viridis')
        # self.fig.show()

        self.prediction_show_cmap = cv2.applyColorMap(self.prediction_show, cmapy.cmap('viridis'))
        # print(f"prediciton shape: {self.prediction_show_cmap.shape}")
        b = self.prediction_show_cmap[:, :, 0]
        g = self.prediction_show_cmap[:, :, 1]
        r = self.prediction_show_cmap[:, :, 2]
        self.prediction_show_cmap = np.zeros(self.prediction_show_cmap.shape, dtype=np.uint8)
        self.prediction_show_cmap[:, :, 0] = r
        self.prediction_show_cmap[:, :, 1] = g
        self.prediction_show_cmap[:, :, 2] = b

        # self.fig = plt.figure()
        # ax = self.fig.add_subplot(1,1,1)
        # ax.imshow(self.prediction_show_cmap, cmap='viridis')
        # self.fig.show()

        # print(f"height: {self.height}, width: {self.width}")
        self.qimage = qimage2ndarray.array2qimage(self.image_show * 255)  # .scaled(self.height, self.width)
        self.qprediction = qimage2ndarray.array2qimage(self.prediction_show_cmap)  # .scaled(self.height, self.width)
        self.qmask = qimage2ndarray.array2qimage(self.current_mask)  # .scaled(self.height, self.width)

        self.qImg = QPixmap(self.qimage)  # .scaled(256,256)
        self.qPred = QPixmap(self.qprediction)  # .scaled(256,256)
        self.qMask = QPixmap(self.qmask)  # .scaled(256,256)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    # save_kwcoco_path = "/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2.1/material_labels.kwcoco.json"
    resume = ""
    # resume = "/media/native/data/data/smart_watch_dvc/drop0_aligned_msi/material_labels2.kwcoco.json"
    save_kwcoco_path = "/media/native/data/data/smart_watch_dvc/drop0_aligned_msi/material_labels2.kwcoco.json"

    coco_fpath = ub.expandpath('/media/native/data/data/smart_watch_dvc/drop0_aligned_msi/data_fielded.kwcoco.json')
    # coco_fpath = ub.expandpath('/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2.1/data_fielded_filtered.kwcoco.json')

    # "cirrus", "coastal", "costal", "green", "lwir11", "lwir12", "nir", "pan", "red", "swir16", "swir22"]
    # channels = 'red|green|blue|nir|lwir11|lwir12|swir16|swir22'#|coastal|costal'
    channels = 'red|green|blue|nir|swir16|swir22|cirrus'
    # channels = 'red|green|blue|nir|swir22|cirrus'

    expected_channels = channels.split('|')
    print(expected_channels)
    # expected_channels = [ "red"  ,"green", "blue", "nir", "lwir11", "lwir12", "swir16", "swir22", "cirrus"]
    dset = kwcoco.CocoDataset(coco_fpath)

    # Only select images with the correct channels
    gids_to_remove = []
    for gid, img in dset.index.imgs.items():
        try:
            # print(img['auxiliary'][0]['channels'])
            image_auxiliary_list = img['auxiliary']
            bands_available = [x['channels'] for x in image_auxiliary_list]
            # print(bands_available)
            if not all(elem in bands_available for elem in expected_channels):
                gids_to_remove.append(gid)
        except Exception:
            gids_to_remove.append(gid)
            continue

    dset.remove_images(gids_to_remove)

    sampler = ndsampler.CocoSampler(dset)

    number_of_timestamps, h, w = 3, 512, 512
    window_dims = (number_of_timestamps, h, w)  # [t,h,w]
    input_dims = (h, w)

    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    loader = dataset.make_loader(batch_size=1)

    window = Window(dataset, dset, resume, save_path=save_kwcoco_path)

    sys.exit(app.exec_())
