import base64

from algorithm_toolkit import Algorithm, AlgorithmChain


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        self.logger.info("sending image to client")

        fn = params['image_path']
        bounds = params['image_bounds']

        cl.add_to_metadata('image_url', fn)
        cl.add_to_metadata('image_extent', bounds)

        encoding = 'utf-8'
        with open(fn, 'rb') as img_file:
            img_bytes = base64.b64encode(img_file.read())
            img_string = img_bytes.decode(encoding)
        chain_output = {
            'output_type': 'geo_raster',
            'output_value': {
                'extent': bounds,
                'raster': img_string
            }
        }
        cl.add_to_metadata('chain_output_value', chain_output)

        return cl
