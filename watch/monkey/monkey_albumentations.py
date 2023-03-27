
def patch_albumentations_for_311():
    """
    Backport https://github.com/albumentations-team/albumentations/pull/1426
    """
    import sys
    from packaging.version import parse as Version
    if sys.version_info[0:2] >= (3, 11):
        import albumentations
        import random
        from typing import Dict, Any
        import numpy as np
        import cv2
        albumentations_version = Version(albumentations.__version__)

        if albumentations_version <= Version('1.0.0'):
            from albumentations.augmentations.transforms import Blur, MotionBlur

            def _patched_blur_get_params(self) -> Dict[str, Any]:
                # return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}
                return {"ksize": int(random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))))}

            def _patched_motion_blur_get_params(self) -> Dict[str, Any]:
                ksize = random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
                if ksize <= 2:
                    raise ValueError("ksize must be > 2. Got: {}".format(ksize))
                kernel = np.zeros((ksize, ksize), dtype=np.uint8)
                xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
                if xs == xe:
                    ys, ye = random.sample(range(ksize), 2)
                else:
                    ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
                cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

                # Normalize kernel
                kernel = kernel.astype(np.float32) / np.sum(kernel)
                return {"kernel": kernel}

            Blur.get_params = _patched_blur_get_params
            MotionBlur.get_params = _patched_motion_blur_get_params

        elif albumentations_version <= Version('1.3.0'):
            from albumentations.augmentations.blur.transforms import Blur, MotionBlur

            def _patched_blur_get_params(self) -> Dict[str, Any]:
                # return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}
                return {"ksize": int(random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))))}

            def _patched_motion_blur_get_params(self) -> Dict[str, Any]:
                import numpy as np
                import cv2
                ksize = random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
                if ksize <= 2:
                    raise ValueError("ksize must be > 2. Got: {}".format(ksize))
                kernel = np.zeros((ksize, ksize), dtype=np.uint8)
                x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
                if x1 == x2:
                    y1, y2 = random.sample(range(ksize), 2)
                else:
                    y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

                def make_odd_val(v1, v2):
                    len_v = abs(v1 - v2) + 1
                    if len_v % 2 != 1:
                        if v2 > v1:
                            v2 -= 1
                        else:
                            v1 -= 1
                    return v1, v2

                if not self.allow_shifted:
                    x1, x2 = make_odd_val(x1, x2)
                    y1, y2 = make_odd_val(y1, y2)

                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2

                    center = ksize / 2 - 0.5
                    dx = xc - center
                    dy = yc - center
                    x1, x2 = [int(i - dx) for i in [x1, x2]]
                    y1, y2 = [int(i - dy) for i in [y1, y2]]

                cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

                # Normalize kernel
                return {"kernel": kernel.astype(np.float32) / np.sum(kernel)}

            Blur.get_params = _patched_blur_get_params
            MotionBlur.get_params = _patched_motion_blur_get_params
