import os

from algorithm_toolkit import Algorithm, AlgorithmChain
from utils.globalmaptiles import GlobalMercator
from utils.image_utils import tile_fname_to_x_y_zoom, num2deg, make_uint8_mask

import numpy as np
from PIL import Image


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        self.logger.info("stitching map tiles")

        tile_fnames = params['image_filenames'].split(',')

        if len(tile_fnames) != 0:
            image_chip_size = 256
            global_mercator = GlobalMercator()

            top_left_lats = []
            top_left_lons = []
            bottom_left_lats = []
            bottom_left_lons = []
            x_array = []
            y_array = []

            web_mercator_top_left_lon = float('inf')
            web_mercator_top_left_lat = float('-inf')
            web_mercator_bot_right_lon = float('-inf')
            web_mercator_bot_right_lat = float('inf')
            ntiles = len(tile_fnames)
            for tile_fname in tile_fnames:
                x, y, zoom = tile_fname_to_x_y_zoom(
                    os.path.basename(tile_fname))
                x_array.append(x)
                y_array.append(y)
                # note that we need to negate the lat due to get this to work
                tile_bounds = global_mercator.TileBounds(x, y, zoom)

                if tile_bounds[0] < web_mercator_top_left_lon:
                    web_mercator_top_left_lon = tile_bounds[0]
                if -tile_bounds[1] > web_mercator_top_left_lat:
                    web_mercator_top_left_lat = -tile_bounds[1]

                if tile_bounds[2] > web_mercator_bot_right_lon:
                    web_mercator_bot_right_lon = tile_bounds[2]
                if -tile_bounds[3] < web_mercator_bot_right_lat:
                    web_mercator_bot_right_lat = -tile_bounds[3]

                top_left_pixel_center = num2deg(x + 0.5, y + 0.5, zoom)
                bottom_right_pixel_center = num2deg(x + 255.5, y + 255.5, zoom)
                top_left_lats.append(top_left_pixel_center[0])
                top_left_lons.append(top_left_pixel_center[1])
                bottom_left_lats.append(bottom_right_pixel_center[0])
                bottom_left_lons.append(bottom_right_pixel_center[1])

            min_x = min(x_array)
            max_x = max(x_array)
            min_y = min(y_array)
            max_y = max(y_array)
            nx = max_x - min_x + 1
            ny = max_y - min_y + 1

            tmp_data = np.array(Image.open(tile_fnames[0]))
            image = np.zeros(
                (ny * image_chip_size, nx * image_chip_size, 3),
                dtype=tmp_data.dtype)

            for cnt, fname in enumerate(tile_fnames):
                per = float(cnt) / float(ntiles) * 100.0
                cl.set_status('Stitching tile: ' + fname, round(per))
                image_chip = np.array(Image.open(fname))
                chip_x, chip_y, chip_zoom = tile_fname_to_x_y_zoom(
                    os.path.basename(fname))
                x_loc = (chip_x - min_x) * image_chip_size
                y_loc = (chip_y - min_y) * image_chip_size
                image[
                    y_loc:y_loc + image_chip_size,
                    x_loc:x_loc + image_chip_size,
                    :
                ] = image_chip[:, :, 0:3]

            cl.set_status('writing out stitched image to disk', 90)
            output_fn = os.path.join(
                cl.get_temp_folder(), "stitched_tiles.png")
            mask = make_uint8_mask(image)
            image = np.dstack((image, mask))
            pil_img = Image.fromarray(image)
            pil_img.save(output_fn)
            cl.add_to_metadata('image_path', output_fn)

            tl_lat, tl_lon = global_mercator.MetersToLatLon(
                web_mercator_top_left_lon, web_mercator_top_left_lat)
            lr_lat, lr_lon = global_mercator.MetersToLatLon(
                web_mercator_bot_right_lon, web_mercator_bot_right_lat)

            coords = []
            coords.append([tl_lat, tl_lon])
            coords.append([lr_lat, lr_lon])

            cl.add_to_metadata('image_bounds', str(coords))
        else:
            self.raise_client_error("no map tiles found to stitch!")

        return cl
