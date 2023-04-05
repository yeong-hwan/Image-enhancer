import cv2
import numpy as np
import math


def apply_average_filter(img, kernel_size):
    """
    It takes 2 arguments, 
    'img' is input image.
    'kernel_size' is size of kernel for average filter.
    """
    edge = int((kernel_size - 1) / 2)

    row_len = len(img)
    col_len = len(img[0])

    adj_value = 1 / (kernel_size ** 2)

    img_result = np.full((row_len, col_len, 3), 0)

    # if kernel_size == 1:
    #     return img

    for row in range(edge, row_len-edge):
        for col in range(edge, col_len-edge):
            temp_0, temp_1, temp_2 = 0, 0, 0
            
            for result_row in range(row-edge, row+edge+1):
                for result_col in range(col-edge, col+edge+1):
                    temp_0 += img[result_row, result_col, 0]
                    temp_1 += img[result_row, result_col, 1]
                    temp_2 += img[result_row, result_col, 2]

            temp_0, temp_1, temp_2 = temp_0 * adj_value, temp_1 * adj_value, temp_2 * adj_value
            img_result[row, col] = [temp_0, temp_1, temp_2]

    return img_result

def apply_sobel_filter(img, kernel_size, is_vertical):
    """
    It takes 3 arguments,
    'img' is input image.
    'kernel_size' is size of kernel for sobel filter.
    'is_vertical' is boolean value. If it is True, you should apply vertical sobel filter.
    Otherwise, you should apply horizontal sobel filter.
    """

    # sobel kernel setting
    sobel_kernel = np.full((kernel_size, kernel_size), 0)
    blur, derivative = [], []


    if kernel_size == 3:
        blur = [1, 2, 1]
        derivative = [-1, 0, 1]
    elif kernel_size == 5:
        blur = [1, 4, 6 ,4 ,1]
        derivative = [-1, -2, 0, 2, 1]
    elif kernel_size == 7:
        blur = [1, 6, 15, 20, 15, 6, 1]
        derivative = [-1, -4, -5, 0, 5, 4, 1]


    for row in range(kernel_size):
        for col in range(kernel_size):
            ver_val = blur[row] * derivative[col]
            hor_val = derivative[row] * blur[col]

            sobel_kernel[row][col] = ver_val if is_vertical == True else hor_val

    # result setting
    edge = int((kernel_size - 1) / 2)

    row_len = len(img)
    col_len = len(img[0])

    img_result = np.full((row_len, col_len, 1), 0)

    for row in range(edge, row_len-edge):
        for col in range(edge, col_len-edge):
            temp = 0

            for result_row in range(row-edge, row+edge+1):
                for result_col in range(col-edge, col+edge+1):
                    # 0 ~ kernel_size
                    kernel_row = result_row - (row-edge)
                    kernel_col = result_col - (col-edge)

                    temp += img[result_row, result_col, 0] * sobel_kernel[kernel_row][kernel_col]

            img_result[row, col, 0] = np.clip(temp, 0, 255)

    return img_result

# Gaussian

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def gaussian_kernel(k_size, sigma):
    size = k_size//2

    y, x = np.ogrid[-size:size+1, -size:size+1]
    gaussian_filter = 1/(2*np.pi * (sigma**2)) * \
        np.exp(-1 * (x**2 + y**2) / (2*(sigma**2)))

    sum_of_filter = gaussian_filter.sum()
    gaussian_filter /= sum_of_filter

    return gaussian_filter


def padding(img, k_size):
    pad_size = k_size//2
    rows, cols, ch = img.shape

    res = np.zeros((rows + (2*pad_size), cols +
                   (2*pad_size), ch), dtype=np.float64)

    if pad_size == 0:
        res = img.copy()
    else:
        res[pad_size:-pad_size, pad_size:-pad_size] = img.copy()

    return res


def apply_gaussian_filter(img, k_size=3, sigma=1):
    rows, cols, channels = img.shape
    gaussian_filter = gaussian_kernel(k_size, sigma)
    pad_img = padding(img, k_size)
    filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)

    for ch in range(0, channels):
        for i in range(rows):
            for j in range(cols):
                filtered_img[i, j, ch] = np.sum(
                    gaussian_filter * pad_img[i:i+k_size, j:j+k_size, ch])

    return filtered_img.astype(np.uint8)


# Mefian

def apply_median_filter(img, kernel_size):
    edge = int((kernel_size - 1) / 2)

    row_len = len(img)
    col_len = len(img[0])

    img_result = np.full((row_len, col_len, 3), 0)

    for row in range(row_len):
        for col in range(col_len):
            temp_0, temp_1, temp_2 = [], [], []

            for result_row in range(row-edge, row+edge+1):
                for result_col in range(col-edge, col+edge+1):
                    try:
                        temp_0.append(img[result_row, result_col, 0])
                    except:
                        pass
                    try:
                        temp_1.append(img[result_row, result_col, 1])
                    except:
                        pass
                    try:
                        temp_2.append(img[result_row, result_col, 2])
                    except:
                        pass

            color_0, color_1, color_2 = np.median(
                temp_0), np.median(temp_1), np.median(temp_2)
            img_result[row, col] = [color_0, color_1, color_2]

    return img_result

# bilateral

def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter//2

    for color in range(3):
        i_filtered = 0
        Wp = 0
        i = 0

        while i < diameter:
            j = 0

            while j < diameter:
                neighbour_x = x - (hl - i)
                neighbour_y = y - (hl - j)

                if neighbour_x >= len(source):
                    neighbour_x -= len(source)

                if neighbour_y >= len(source[0]):
                    neighbour_y -= len(source[0])

                # print(source[neighbour_x][neighbour_y])
                # print(source[x][y])
                # print(source[neighbour_x][neighbour_y] - source[x][y])

                gi = gaussian(source[neighbour_x]
                              [neighbour_y][color] - source[x][y][color], sigma_i)
                gs = gaussian(
                    distance(neighbour_x, neighbour_y, x, y), sigma_s)

                w = gi * gs
                i_filtered += source[neighbour_x][neighbour_y][color] * w

                Wp += w
                j += 1
            i += 1

        i_filtered = i_filtered / Wp
        filtered_image[x][y][color] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):

    row_len = len(source)
    col_len = len(source[0])
    filtered_image = img_result = np.full((row_len, col_len, 3), 0)

    row = 0
    while row < row_len:
        col = 0
        while col < col_len:
            apply_bilateral_filter(
                source, filtered_image, row, col, filter_diameter, sigma_i, sigma_s)
            col += 1
        row += 1

    print(filtered_image)
    return filtered_image