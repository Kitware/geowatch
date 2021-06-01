import sys
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QSizePolicy, 
                             QDoubleSpinBox, QLabel, QCheckBox, QMainWindow, 
                             QGridLayout, QComboBox, QTextBrowser, QLineEdit, QInputDialog)
from PyQt5.QtCore import Qt  
from PyQt5 import QtCore
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ubelt as ub
from material_seg.datasets.iarpa_dataset import *
import kwcoco
import ndsampler
import watch
import numpy as np
from watch.utils.util_norm import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pdb
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy
from PIL import Image
from PIL.ImageQt import ImageQt
import qimage2ndarray
import cmapy
import kwcoco
import random
import kwimage

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # print(sm)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)



class Window(QMainWindow):
    def __init__(self, dataset, dset, resume='', save_path=''):
        super().__init__()
        self.image_counter = 0
        self.dataset = dataset
        self.channels, self.timesteps, self.im_width, self.im_height = self.dataset[self.image_counter]['inputs']['im'].data.shape
        self.dset = dset
        self.save_path = save_path
        self.k = 80
        self.seen_labels = []
        self.class_labels_all = []
        self.class_labels_pairs = {}
        self.class_label_with = "Concrete"
        self.class_label_to_index = {"Nothing":0, "Concrete":1, "Vegetation":2, "Soil":3, "Water":4}
        if len(resume)>1:
            self.material_dset = kwcoco.CocoDataset(resume)
            self.image_counter = int(len(list(self.material_dset.index.imgs.keys()))/self.timesteps)
            print(self.image_counter)
        else:
            self.material_dset = kwcoco.CocoDataset()
            for key, value in self.class_label_to_index.items():
                if key !=0:
                    self.material_dset.add_category(name=key, id=value)
        

        print(self.dataset[self.image_counter]['inputs']['im'].data.shape)
        # self.data_dims = self.dataset[self.image_counter]['tr'].data['space_dims']
        self.width, self.height = self.dataset[self.image_counter]['tr'].data['space_dims']
        print(self.dataset[self.image_counter]['tr'].data['space_dims'])

        self.scale_factor = 1
        self.scaled_height, self.scaled_width = int(self.height*self.scale_factor), int(self.width*self.scale_factor)
        self.current_mask = np.zeros((self.width, self.height)).astype(np.uint8)
        self.separable_current_mask = np.zeros((len(list(self.class_label_to_index.keys())), self.width, self.height)).astype(np.uint8)
        self.widget = QWidget()
        self.widget.keyPressEvent = self.keyPressEvent
        self.load_images(index=self.image_counter)
        
        print(f"height: {self.height}, width: {self.width}")
        self.label_img = QLabel(self.widget)
        self.image = QPixmap(self.qImg)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20,128,self.height, self.width)
        self.label_img.mousePressEvent = self.getImagePixel
        
        self.label_prediction = QLabel(self.widget)
        self.prediction = QPixmap(self.qPred)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.width+20+120,128,self.height, self.width)
        self.label_prediction.mousePressEvent = self.getMaskPixel
        
        self.label_mask = QLabel(self.widget)
        self.mask = QPixmap(self.qMask)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3*self.width+20,128, self.height, self.width)
        
        self.output_textbox = QTextBrowser(self.widget)
        self.output_textbox.setGeometry(QtCore.QRect(20, self.height + 256 + 50, 100, 100))
        self.output_textbox.setObjectName("Current Clusters selected")
        
        self.output_textbox_name = QTextBrowser(self.widget)
        self.output_textbox_name.setGeometry(QtCore.QRect(140, self.height + 256 + 50, 100, 100))
        self.output_textbox_name.setObjectName("Labels of Selected Clusters")
        
        self.remove_cluster = QLineEdit(self.widget)
        self.remove_cluster.setObjectName("Cluster Removal Tool")
        self.remove_cluster.move(256, self.height + 256 + 50)
        self.remove_cluster.resize(100,40)
        self.remove_cluster.textChanged.connect(self.textchanged)

        self.image_update = QPushButton(self.widget)
        self.image_update.setText("Next Image")
        self.image_update.move(140,20)
        self.image_update.clicked.connect(self.update_image)
        
        self.class_label = QComboBox(self.widget)
        self.class_label.addItems(["Nothing","Concrete", "Vegetation", "Soil", "Water"])
        self.class_label.move(300,20)
        self.class_label.currentIndexChanged.connect(self.class_label_selection)
        
        self.show_current_mask = QPushButton(self.widget)
        self.show_current_mask.setText("Show Current Mask")
        self.show_current_mask.move(500,20)
        self.show_current_mask.clicked.connect(self.show_current_mask_clicked)
        
        self.increase_size = QPushButton(self.widget)
        self.increase_size.setText("Increase Image Size")
        self.increase_size.move(700,20)
        self.increase_size.clicked.connect(self.increase_images_size)
        
        self.decrease_size = QPushButton(self.widget)
        self.decrease_size.setText("Decrease Image Size")
        self.decrease_size.move(900,20)
        self.decrease_size.clicked.connect(self.decrease_images_size)
        
        self.timestamp_label = QComboBox(self.widget)
        self.timestamp_label.addItems([str(x) for x in range(self.timesteps)])
        self.timestamp_label.move(1100,20)
        self.timestamp_label.currentIndexChanged.connect(self.change_inter_batch_image)
        
        self.save_images = QPushButton(self.widget)
        self.save_images.setText("Update Material Dataset")
        self.save_images.move(1200,20)
        self.save_images.clicked.connect(self.save_images_clicked)

        self.label_mask_title = QLabel(self.widget)
        self.label_mask_title.setText("Intermediate Mask")
        self.label_mask_title.move(2*self.width+256,100)
        
        self.label_img_title = QLabel(self.widget)
        self.label_img_title.setText(f"Image {self.image_counter}")
        self.label_img_title.move(256,100)
        
        self.label_prediction_title = QLabel(self.widget)
        self.label_prediction_title.setText("Clustering Prediction")
        self.label_prediction_title.move(self.width+256,100)
        
        self.remove_cluster_title = QLabel(self.widget)
        self.remove_cluster_title.setText("Write cluster to remove from the mask")
        self.remove_cluster_title.move(256, self.height + 256)
        
        self.output_textbox_title = QLabel(self.widget)
        self.output_textbox_title.setText("Current Clusters")
        self.output_textbox_title.move(20, self.height + 256)
        
        self.output_textbox_name_title = QLabel(self.widget)
        self.output_textbox_name_title.setText("Clusters Labels")
        self.output_textbox_name_title.move(140, self.height + 256)
        
        self.widget.setGeometry(50,50,1800,800)
        self.widget.show()
    
    # def convert_mask_to_binaries(self):
        
    
    def save_images_clicked(self):
        gids = self.dataset[self.image_counter]['tr'].data['gids']
        for i in range(1,len(list(self.class_label_to_index.keys()))):
            if len(np.unique(self.separable_current_mask[i,:,:])) > 1:
                binary_mask = kwimage.Mask(self.separable_current_mask[i,:,:], format='c_mask')
                binary_polygon = binary_mask.to_multi_polygon()
                # binary_coco = binary_polygon.to_coco(style='new')
                binary_segmentation = kwimage.Segmentation.coerce(binary_polygon).to_coco(style="new")
                for gid in gids:
                    self.material_dset.add_annotation(image_id=gid, category_id=i, segmentation=binary_segmentation)
        self.material_dset.validate()
        self.material_dset._check_integrity()
        self.material_dset.dump(self.save_path, newlines=True)
        
    def change_inter_batch_image(self):
        image_data = self.dataset[self.image_counter]['inputs']['im'].data # [b,c,t,h,w]
        image_data = image_data[:,:, :self.width, :self.height]#.copy()
        c, t, h, w = image_data.shape
        # random_t = random.randrange(t)
        self.t_selection = int(self.timestamp_label.currentText())
        # print(random_t)
        image_show = np.array(image_data[:,self.t_selection,:,:]).transpose(1, 2, 0).copy() # visualize 0 indexe
        image_min = np.min(image_show)
        image_max = np.max(image_show)
        self.image_show = (image_show - image_min)/(image_max - image_min)
        self.qimage = qimage2ndarray.array2qimage(self.image_show*255)#.scaled(self.height, self.width)
        self.qImg = QPixmap(self.qimage)#.scaled(256,256)
        
        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        # self.label_img.setGeometry(20,128,self.scaled_height, self.scaled_width)
    
    def increase_images_size(self):
        self.scale_factor += 0.25
        self.scaled_height, self.scaled_width = int(self.scale_factor*self.height), int(self.scale_factor*self.width)
        
        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20,128,self.scaled_height, self.scaled_width)
        
        self.prediction = QPixmap(self.qPred).scaled(self.scaled_height, self.scaled_width)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.scaled_width+20+70*(1+self.scale_factor),128,self.scaled_height, self.scaled_width)
        
        self.mask = QPixmap(self.qMask).scaled(self.scaled_height, self.scaled_width)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3*self.scaled_width,128, self.scaled_height, self.scaled_width)
    
    def decrease_images_size(self):
        self.scale_factor -= 0.25
        self.scaled_height, self.scaled_width = int(self.scale_factor*self.height), int(self.scale_factor*self.width)
        
        self.image = QPixmap(self.qImg).scaled(self.scaled_height, self.scaled_width)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20,128,self.scaled_height, self.scaled_width)
        
        self.prediction = QPixmap(self.qPred).scaled(self.scaled_height, self.scaled_width)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.scaled_width+20+50*(1+self.scale_factor),128,self.scaled_height, self.scaled_width)
        
        self.mask = QPixmap(self.qMask).scaled(self.scaled_height, self.scaled_width)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(3*self.scaled_width,128, self.scaled_height, self.scaled_width)
    
    def keyPressEvent(self, event):
        # if event.key() == Qt.Key_Space:
        
        keys_to_class_dict = {48:"Nothing",49:"Concrete", 50:"Vegetation", 51:"Soil", 52:"Water"}
        # print(event.key())
        if event.key() in keys_to_class_dict.keys():
            self.class_label_with = keys_to_class_dict[event.key()]
            self.class_label.setCurrentText(keys_to_class_dict[event.key()])
        
    def show_current_mask_clicked(self):
        current_mask_no_bg = np.ma.masked_where(self.current_mask==0,self.current_mask)
        plt.imshow(self.image_show)
        plt.imshow(current_mask_no_bg, alpha=0.6, cmap='tab20')
        plt.show()
        
    def finalize_mask(self):
        gids = self.dataset[self.image_counter]['tr'].data['gids']
        for i in range(1,len(list(self.class_label_to_index.keys()))):
            if len(np.unique(self.separable_current_mask[i,:,:])) > 1:
                binary_mask = kwimage.Mask(self.separable_current_mask[i,:,:], format='c_mask')
                binary_polygon = binary_mask.to_multi_polygon()
                # binary_coco = binary_polygon.to_coco(style='new')
                binary_segmentation = kwimage.Segmentation.coerce(binary_polygon)#.to_coco(style="new")
                for gid in gids:
                    self.material_dset.add_annotation(image_id=gid, 
                                                      category_id=i, 
                                                      bbox=list(binary_polygon.bounding_box().to_coco(style="new"))[0], 
                                                      segmentation=binary_segmentation.to_coco(style="new"))
        self.material_dset.validate()
        self.material_dset._check_integrity()
        self.material_dset.dump(self.save_path, newlines=True)
        
    def class_label_selection(self):
        self.class_label_with = self.class_label.currentText()
    
    def textchanged(self, text):
        # if text in self.output_textbox:
        # self.output_textbox.remove(text)
        num, ok = QInputDialog.getInt(self,"Select a cluster to remove", "enter a number")
        # print(ok)
        if ok and num:
            # self.remove_cluster.setText(str(num))
            if num in self.seen_labels:
                xs, ys = np.where(self.prediction_show==num)
                self.current_mask[xs,ys] = 0
                self.update_mask()
                self.seen_labels.remove(num)
    
    
    def getImagePixel(self, event):
        x = int(event.pos().x()//self.scale_factor)
        y = int(event.pos().y()//self.scale_factor)
        self.value = self.prediction_show[y,x]
        xs, ys = np.where(self.prediction_show==self.value)
        
        # print(self.value)
        self.current_mask[xs,ys] = self.class_label_to_index[self.class_label_with]
        self.separable_current_mask[self.class_label_to_index[self.class_label_with],xs,ys] = 1
        
        self.update_mask()
        self.class_labels_pairs[self.value] = self.class_label_with
        self.seen_labels.append(self.value)
        self.output_textbox.append(str(self.value))   
        self.output_textbox_name.append(str(self.class_label_with))
        # print(self.class_labels_pairs)
            
    def getMaskPixel(self, event):
        x = int(event.pos().x()//self.scale_factor)
        y = int(event.pos().y()//self.scale_factor)
        self.value = self.prediction_show[y,x]
        xs, ys = np.where(self.prediction_show==self.value)
        
        # print(self.value)
        self.current_mask[xs,ys] = self.class_label_to_index[self.class_label_with]
        self.separable_current_mask[self.class_label_to_index[self.class_label_with],xs,ys] = 1
        
        self.update_mask()
        self.class_labels_pairs[self.value] = self.class_label_with
        self.seen_labels.append(self.value)
        self.output_textbox.append(str(self.value))   
        self.output_textbox_name.append(str(self.class_label_with))
        # print(self.class_labels_pairs)
    
    def update_mask(self):
        # plt.imshow(self.current_mask)
        # plt.show()
        self.current_mask_cmap = (self.current_mask*20).astype(np.uint8)
        # print(self.current_mask_cmap)
        self.current_mask_cmap = cv2.applyColorMap(self.current_mask_cmap, cmapy.cmap('viridis'))     
        self.qmask = qimage2ndarray.array2qimage(self.current_mask_cmap).scaled(self.height, self.width)
        self.mask = QPixmap(self.qmask)#.scaled(256,256)
        # self.image = QPixmap(self.qImg)
        self.label_mask.setPixmap(self.mask)
        # print(np.unique(self.current_mask))
        
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
        self.label_img_title.setText(f"Image {self.image_counter}")
        self.width, self.height = self.dataset[self.image_counter]['tr'].data['space_dims']
        self.current_mask = np.zeros((self.width, self.height)).astype(np.uint8)
        # self.width_factor, self.height_factor = self.vis_width/self.width, self.vis_height/self.height
        self.load_images(index=self.image_counter)
        print("updated image and prediction")
        
        self.image = QPixmap(self.qImg)
        self.label_img.setPixmap(self.image)
        self.label_img.setGeometry(20,128,self.height, self.width)
        
        self.prediction = QPixmap(self.qPred)
        self.label_prediction.setPixmap(self.prediction)
        self.label_prediction.setGeometry(self.width+20+50,128,self.height, self.width)
        
        self.mask = QPixmap(self.qMask)
        self.label_mask.setPixmap(self.mask)
        self.label_mask.setGeometry(2*self.width+128,128, self.height, self.width)
        
    def load_images(self, index):
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        image_data = self.dataset[index]['inputs']['im'].data # [b,c,t,h,w]
        gids = self.dataset[self.image_counter]['tr'].data['gids']
        for gid in gids:
            image_dict =  self.dset.index.imgs[gid]
            video_dict = self.dset.index.videos[image_dict['video_id']]
            if gid not in self.material_dset.index.imgs.keys():
                self.material_dset.add_image(**image_dict)
            
        if image_dict['video_id'] not in self.material_dset.index.videos.keys():
            self.material_dset.add_video(**video_dict)
        
        image_data = image_data[:,:, :self.width, :self.height]#.copy()
        c, t, h, w = image_data.shape
        image_show = np.array(image_data[:,1,:,:]).transpose(1, 2, 0).copy() # visualize 0 indexe
        image_min = np.min(image_show)
        image_max = np.max(image_show)
        self.image_show = (image_show - image_min)/(image_max - image_min)

        image_data = image_data.contiguous().view(c,t, h*w)
        image_data = torch.transpose(image_data,0,2)
        image_data = torch.flatten(image_data,start_dim=1, end_dim=2)
        
        kmeans.fit(image_data)
        cluster_labels = kmeans.labels_
        self.prediction_show = (cluster_labels.reshape(h,w)*6).astype(np.uint8)
        
        # self.fig = plt.figure()
        # ax = self.fig.add_subplot(1,1,1)
        # ax.imshow(self.prediction_show, cmap='viridis')
        # self.fig.show()
        
        self.prediction_show_cmap = cv2.applyColorMap(self.prediction_show, cmapy.cmap('viridis'))
        print(f"prediciton shape: {self.prediction_show_cmap.shape}")
        b = self.prediction_show_cmap[:,:,0]
        g = self.prediction_show_cmap[:,:,1]
        r = self.prediction_show_cmap[:,:,2]
        self.prediction_show_cmap = np.zeros(self.prediction_show_cmap.shape, dtype=np.uint8)
        self.prediction_show_cmap[:,:,0] = r
        self.prediction_show_cmap[:,:,1] = g
        self.prediction_show_cmap[:,:,2] = b

        # self.fig = plt.figure()
        # ax = self.fig.add_subplot(1,1,1)
        # ax.imshow(self.prediction_show_cmap, cmap='viridis')
        # self.fig.show()

        print(f"height: {self.height}, width: {self.width}")
        self.qimage = qimage2ndarray.array2qimage(self.image_show*255)#.scaled(self.height, self.width)
        self.qprediction = qimage2ndarray.array2qimage(self.prediction_show_cmap)#.scaled(self.height, self.width)
        self.qmask = qimage2ndarray.array2qimage(self.current_mask)#.scaled(self.height, self.width)

        self.qImg = QPixmap(self.qimage)#.scaled(256,256)
        self.qPred = QPixmap(self.qprediction)#.scaled(256,256)
        self.qMask = QPixmap(self.qmask)#.scaled(256,256)

app = QApplication(sys.argv)
coco_fpath = ub.expandpath('/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/data_fielded_filtered_rgb.kwcoco.json')
dset = kwcoco.CocoDataset(coco_fpath)

# material_coco_fpath = ub.expandpath('/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/material_labels.kwcoco.json')


# print(material_dset)

sampler = ndsampler.CocoSampler(dset)

# # print(sampler)
number_of_timestamps, h, w = 4, 512, 512
window_dims = (number_of_timestamps, h, w) #[t,h,w]
input_dims = (h, w)

# # channels = 'r|g|b|gray|wv1'
channels = 'r|g|b'
# channels = 'gray'
dataset = IARPAVideoDataset(sampler, window_dims, input_dims, channels)
loader = dataset.make_loader(batch_size=1)
# print(dset.dataset[2])


# for item in dataset:
# print(dataset[1]['inputs']['im'])
# resume = "/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/material_labels.kwcoco.json"
resume = ""
save_kwcoco_path = "/home/native/core534_data/datasets/smart_watch/processed/drop0_aligned_v2/material_labels2.kwcoco.json"
window = Window(dataset, dset, resume, save_path=save_kwcoco_path)
# image=image_show[0,0,:,:,:], prediction=prediction
# for batch in loader:    
#     # pdb.set_trace()
#     image_data = batch['inputs']['im'].data[0] # [b,c,t,h,w]
#     b, c, t, h, w = image_data.shape
#     mask_data = batch['label']['class_masks'].data[0] #len(mask_data) = b
#     mask_data = torch.stack(mask_data)
    
#     image_show = np.array(image_data).transpose(0, 2, 3, 4, 1)/500 # visualize 0 indexed in batch
#     # image_show = image_show[0,]
#     # mask_show = np.array(mask_data) # [b,t,h,w]
    
#     image_data = image_data.view(b, c*t, h*w)
#     image_data = torch.transpose(image_data,1,2)
#     image_data = torch.flatten(image_data,start_dim=0, end_dim=1)
#     # image_data = torch.transpose(image_data,0,1)
#     # print(image_data.shape)
#     out_feat_embed = TSNE(n_components=2).fit_transform(image_data)
#     # data = image_data
#     data = out_feat_embed
#     kmeans.fit(data)
#     cluster_centers = kmeans.cluster_centers_
#     cluster_labels = kmeans.labels_
#     y_kmeans = kmeans.predict(data)
#     print(cluster_centers)
#     print(cluster_labels)
#     prediction = cluster_labels.reshape(h,w)
    
#     window.update_image(image_show[0,0,:,:,:])
# window.show()
# WaitKey(0)
    
sys.exit(app.exec_())