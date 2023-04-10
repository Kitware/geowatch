
import cv2
import numpy as np

#-----------------------------------------------------------------------
# Align with perspective model
#-----------------------------------------------------------------------

# (ORB) feature based alignment


def featureAlign(im1, im2):

    max_features = 5000
    feature_retention = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * feature_retention)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

#-----------------------------------------------------------------------
# Warp image with optical flow
#-----------------------------------------------------------------------


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_copy = flow.copy()
    # flow = -flow
    flow_copy[:, :, 0] += np.arange(w)
    flow_copy[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow_copy, None, cv2.INTER_LINEAR)
    return res

#-----------------------------------------------------------------------
# Compute optical flow
#-----------------------------------------------------------------------


def optical_flow(img, img_B):
    prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img)

    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    # Image.fromarray(rgb).save('/media/FastData/yguo/imgD.png')

    # return rgb

    return flow, rgb

#-----------------------------------------------------------------------
# Compute correlation
#-----------------------------------------------------------------------


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()

    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def sim_correlation(im1, im2):
    # correlation = np.zeros_like(im1)
    correlation = np.zeros(im1.shape)

    sh_row, sh_col = im1.shape
    # d = 9
    d = 5

    for i in range(d, sh_row - (d + 1)):
        for j in range(d, sh_col - (d + 1)):
            val = correlation_coefficient(im1[i - d: i + d + 1, j - d: j + d + 1],
                                          im2[i - d: i + d + 1, j - d: j + d + 1])
            correlation[i, j] = val
    return correlation


#-----------------------------------------------------------------------
'''
# compute normalized correlation between two images
def compute_ncc(img1, img2):

    x = image_to_tensor(img1)
    y = image_to_tensor(img2)

    ncc, ncc_map = functional.normalized_cross_correlation(x, y, return_map=True)

    print('normalized correlation score is : ', ncc)

    ncc_map = ncc_map.cpu().data.numpy()
    norm_map = (ncc_map - ncc_map.min())/(ncc_map.max() - ncc_map.min())

    return norm_map
#-----------------------------------------------------------------------
'''
