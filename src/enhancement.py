import cv2
import numpy as np

##### To-do #####

# 1. cv2 add


def enhancement_1(img):
    """
    This function should return an image that goes through enhancement.
    """
    val = 50

    val_set = np.full(img.shape, (val, val, val), dtype=np.uint8)

    result_img = cv2.add(img, val_set)
    result_img = np.clip(result_img, 0, 255)

    return result_img

# 2. contrast


def enhancement_2(img):
    """
    This function should return an image that goes through enhancement.
    """
    bright_img = enhancement_1(img)

    alpha = 0.5
    result_img = (1+alpha) * bright_img - 128 * alpha
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    return result_img

# 3. histogram equallization


def enhancement_3(img):
    """
    This function should return an image that goes through enhancement.
    """
    # ravel : make flat
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    hist_sum = hist.cumsum()

    # except 0 by mask
    hist_m0 = np.ma.masked_equal(hist_sum, 0)

    # equallization
    hist_m0 = (hist_m0 - hist_m0.min()) * 255 / (hist_m0.max() - hist_m0.min())

    # restore 0 from mask excepted
    hist_sum = np.ma.filled(hist_m0, 0).astype('uint8')

    result_img = hist_sum[img]

    return result_img


# test_img = cv2.imread('task3/engi_hall_low_light.jpg')

# enhanced_img_1 = enhancement_1(test_img)
# enhanced_img_2 = enhancement_2(test_img)
# enhanced_img_3 = enhancement_3(test_img)

# print("0. original img")
# plt.imshow(test_img)
# plt.axis('off')
# plt.show()

# print("1. cv2 add")
# plt.imshow(enhanced_img_1)
# plt.axis('off')
# plt.show()

# print("2. contrast")
# plt.imshow(enhanced_img_2)
# plt.axis('off')
# plt.show()

# print("3. histogram equallization")
# plt.imshow(enhanced_img_3)
# plt.axis('off')
# plt.show()
