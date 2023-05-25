import os
import pickle
import hashlib

import kwcoco
import numpy as np
import ubelt as ub
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset

from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_image import add_buffer_to_image
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import get_crop_slices, get_repo_paths, generate_image_slice_object

def create_hash_str(method_name='sha256', **kwargs):

    hashed_str = ''
    for name, value in kwargs.items():
        if isinstance(value, str):
            pass
        if isinstance(value, list):
            kwargs[name] = '_'.join(value)
        else:
            kwargs[name] = str(value)

        hashed_str += kwargs[name]

    if method_name == 'sha256':
        sha = hashlib.sha256()
        sha.update(hashed_str.encode())
        output_str = sha.hexdigest()
    else:
        raise NotImplementedError(f'Not method "{method_name}" for hashing.')

    return output_str


def _get_video_examples(coco_images, channel_code, crop_params):
    coco_image = coco_images[0]
    img = coco_image.imdelay(channel_code, space='video', resolution='10GSD').finalize()
    region_height, region_width, _ = img.shape

    crop_slices = get_crop_slices(region_height, region_width, crop_params.height,
                                  crop_params.width, crop_params.stride)

    region_examples = []
    for coco_image in coco_images:
        for crop_slice in crop_slices:
            example = {
                'crop_slice': crop_slice,
                'image_id': coco_image.img['id'],
                'vid': coco_image.video['id'],
                'region_name': coco_image.video['name'],
                'region_res': [region_height, region_width],
            }
            region_examples.append(example)

    return region_examples


class InferenceDataset(Dataset):

    def __init__(self,
                 video_id,
                 channels,
                 kwcoco_path,
                 crop_params,
                 sensors=['S2', 'L8'],
                 n_cache_workers=4,
                 force_regenerate_cache=False):
        self.sensors = sensors
        self.channels = channels
        self.video_id = video_id
        self.crop_params = crop_params
        self.n_cache_workers = n_cache_workers

        # Get channel info.
        self.channel_code, self.n_channels = self.get_channel_code(channels)

        print(f'Loading kwcoco file ... \n{os.path.split(kwcoco_path)[1]}')
        self.coco_dset = kwcoco.CocoDataset(kwcoco_path)

        # Check if example file already exists.
        try:
            cache_dir = get_repo_paths('inference_dataset_cache')
        except FileNotFoundError:
            cache_dir = os.path.join(os.getcwd(), 'material_inference_dataset_cache')
            os.makedirs(cache_dir, exist_ok=True)

        cache_name = create_hash_str(**{
            'channels': channels,
            'crop_params': crop_params,
            'kwcoco_name': os.path.split(kwcoco_path)[1],
            'sensors': sensors,
            'video_id': str(video_id)
        },
                                     method_name='sha256')
        cache_save_path = os.path.join(cache_dir, cache_name + '.p')

        # Create examples.
        if (os.path.exists(cache_save_path) is False) or (force_regenerate_cache):
            print(f'Creating: {cache_save_path}')
            self.examples = self._cache_examples(cache_save_path)
        else:
            print(f'Loading: {cache_save_path}')
            self.examples = pickle.load(open(cache_save_path, 'rb'))

    def get_channel_code(self, channel_str):
        if channel_str == 'RGB':
            channel_code = 'red|green|blue'
        elif channel_str == 'RGB_NIR':
            channel_code = 'red|green|blue|nir'
        else:
            raise NotImplementedError

        n_channels = len(channel_code.split('|'))

        return channel_code, n_channels

    def _cache_examples(self, save_path):
        video_ids = list(self.coco_dset.videos())

        if self.video_id not in video_ids:
            raise ValueError(f'Video id {self.video_id} not found in kwcoco file.')
        else:
            video_ids = [self.video_id]

        jobs = ub.JobPool(mode='process', max_workers=1)

        for vid in video_ids:
            video_image_ids = self.coco_dset.index.vidid_to_gids[vid]

            # Filter image ids by sensor.
            filtered_image_ids = []
            for image_id in video_image_ids:
                if self.coco_dset.imgs[image_id]['sensor_coarse'] in self.sensors:
                    filtered_image_ids.append(image_id)

            images = self.coco_dset.images(image_ids=filtered_image_ids)
            coco_images = images.coco_images

            jobs.submit(_get_video_examples, coco_images, self.channel_code, self.crop_params)

        examples = []
        for job in tqdm(jobs.as_completed(),
                        colour='green',
                        desc='Generating examples',
                        total=len(video_ids)):
            region_examples = job.result()
            examples.extend(region_examples)

        pickle.dump(examples, open(save_path, 'wb'))
        return examples

    def __len__(self):
        return len(self.examples)

    def _load_crop_norm(self, image_id, channels, crop_slice=None):
        # Load & crop.
        if crop_slice:
            h0, w0, dh, dw = crop_slice

        coco_image = self.coco_dset.coco_image(image_id).imdelay(channels=channels,
                                                                 space='video',
                                                                 resolution='10GSD')

        if crop_slice:
            frame = coco_image.crop((slice(h0, h0 + dh), slice(w0, w0 + dw))).finalize()
        else:
            frame = coco_image.finalize()

        # Change dimension ordering.
        frame = rearrange(frame, 'h w c -> c h w')

        # Normalize.
        frame = np.clip(frame / 2**16, 0, 1)

        return frame

    def to_RGB(self, norm_image, gamma=0.6):
        norm_image *= 2**4
        if self.channels == 'RGB':
            rgb_image = np.clip(norm_image**gamma, 0, 1)
        elif self.channels == 'RGB_NIR':
            rgb_image = np.clip(norm_image[:3]**gamma, 0, 1)
        else:
            raise NotImplementedError
        return rgb_image

    def __getitem__(self, index):
        example = self.examples[index]

        frame = self._load_crop_norm(example['image_id'],
                                     channels=self.channel_code,
                                     crop_slice=example['crop_slice'])

        frame, _ = add_buffer_to_image(frame, self.crop_params.height, self.crop_params.width)

        output = {}
        output['frame'] = np.nan_to_num(frame).astype('float32')

        output.update(example)

        return output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ex_kwcoco_path = '/data4/datasets/dvc-repos/smart_data_dvc/Drop6/baseline-data_vali_split2_1year_mean_sep_True_anno.kwcoco.zip'
    ex_crop_params = generate_image_slice_object(400)
    ex_channels = 'RGB'
    dataset = InferenceDataset(ex_channels,
                               ex_kwcoco_path,
                               ex_crop_params,
                               sensors=['S2', 'L8'],
                               n_cache_workers=4,
                               force_regenerate_cache=False)
    index = 85
    ex = dataset.__getitem__(index)
    print(ex['region_name'])
    print(ex['image_id'])
    rgb_frame = dataset.to_RGB(ex['frame']).transpose(1, 2, 0)
    plt.imshow(rgb_frame)
    plt.savefig(f'Inference_Dataset_{index}.png')