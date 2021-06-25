from PIL import Image
import material_seg.utils.utils as utils
import torchvision.transforms.functional as FT
import torch
import numpy as np
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
import itertools
torch.set_printoptions(precision=6, sci_mode=False)

IMG_EXTENSIONS = ['*.png', '*.jpeg', '*.jpg', '*.npy']

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class DeepGlobeDataset(object):
    def __init__(self, root, transforms, split=False):
        self.root = root
        self.transforms = transforms
        self.split = split

        self.images_root = f"{self.root}/{split}/images/"
        self.masks_root = f"{self.root}/{split}/masks/"
        self.masks_paths = utils.dictionary_contents(
            path=self.masks_root, types=['*.png'])

        self.mask_mapping = {0: 0,    # 0, unknown
                             179: 1,  # 179, urban land
                             226: 2,  # 226, agriculture land
                             105: 3,  # 105, rangeland, non-forest, non farm, green land
                             150: 4,  # 150, forest land
                             29: 5,   # 29, water
                             255: 6}  # 255, barren land, mountain, rock, dessert

        self.possible_combinations = [list(i) for i in itertools.product([0, 1], repeat=len(self.mask_mapping.keys()))]
        # possible_combinations = map(lambda x: x/len(self.mask_mapping.keys()), possible_combinations)
        self.possible_combinations = np.divide(self.possible_combinations, len(self.mask_mapping.keys()))
        # print(self.possible_combinations)
        
    def __getitem__(self, idx):

        mask_path = self.masks_paths[idx]
        image_name = mask_path.split('/')[-1].split('.')[0]
        img_path = f"{self.images_root}/{image_name}.png"

        labels = torch.zeros(size=(1,len(self.mask_mapping.keys())))
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # .convert("L"))
        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # plt.show()
        
        new_mask = FT.to_tensor(mask) * 255
        total_pixels = new_mask.shape[2]*new_mask.shape[1]
        label_inds, label_counts = torch.unique(new_mask, return_counts=True)
        label_inds = label_inds.long()
        # print(total_pixels)
        # print(label_inds)
        # print(label_counts)
        distribution = label_counts/total_pixels
        
        for label_ind, label_count in zip(label_inds, label_counts):
            labels[0, label_ind] = label_count/total_pixels
            
        from scipy.spatial import distance
        # print(self.possible_combinations.shape)
        # print(labels.shape)
        distances = distance.cdist(self.possible_combinations, labels, 'cityblock')
        label = np.argmin(distances).item()
        # print(label)
        # print(type(label))
        # label = torch.Tensor(label)
        # print(label)
        
        new_image = self.transforms(img)
        outputs = {}
        outputs['visuals'] = {'image': new_image, 'mask': new_mask, 'image_name': image_name}
        outputs['inputs'] = {'image': new_image, 'mask': new_mask, 'labels':label}

        return outputs

    def __len__(self):
        return len(self.masks_paths)
