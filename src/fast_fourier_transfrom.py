import cv2
import matplotlib.pyplot as plt
import numpy as np


def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    row_len, col_len = img.shape
    fft_shifted = np.roll(img, (row_len//2, col_len//2), axis=(0, 1))

    return fft_shifted


def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    row_len, col_len = np.shape(img)
    fft_unshifted = np.roll(img, (-row_len//2, -col_len//2), axis=(0, 1))

    return fft_unshifted


def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    fft_img = np.fft.fft2(img)
    result_spectrum = np.log(np.abs(fftshift(fft_img)) + 1)
    return result_spectrum


def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    fft_img = np.fft.fft2(img)
    fftshift_img = fftshift(fft_img)

    row_len, col_len = fftshift_img.shape
    row_center, col_center = int(row_len/2), int(col_len/2)

    result_fft_img = fftshift_img.copy()

    for row in range(row_len):
        for col in range(col_len):
            edge_radius = np.sqrt((row_center-row)**2 + (col_center-col)**2)

            if edge_radius > r:
                result_fft_img[row, col] = 0

    ifft_img = ifftshift(result_fft_img)
    result_img = np.fft.ifft2(ifft_img).real

    return result_img


def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    fft_img = np.fft.fft2(img)
    fftshift_img = fftshift(fft_img)

    row_len, col_len = fftshift_img.shape
    row_center, col_center = int(row_len/2), int(col_len/2)

    result_fft_img = fftshift_img.copy()

    for row in range(row_len):
        for col in range(col_len):
            edge_radius = np.sqrt((row_center-row)**2 + (col_center-col)**2)

            if edge_radius < r:
                result_fft_img[row, col] = 0

    ifft_img = ifftshift(result_fft_img)
    result_img = np.fft.ifft2(ifft_img).real

    return result_img

#################


if __name__ == '__main__':
    img = cv2.imread('task2/task2_filtering.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)

    # save the filtered/denoised images
    # cv2.imwrite('low_passed.png', low_passed)
    # cv2.imwrite('high_passed.png', high_passed)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2, 7, 1), img, 'Original')
    drawFigure((2, 7, 2), low_passed, 'Low-pass')
    drawFigure((2, 7, 3), high_passed, 'High-pass')
    # drawFigure((2, 7, 4), fft_shifted, 'fft-shifted')

    drawFigure((2, 7, 8), fm_spectrum(img), 'Spectrum')
    drawFigure((2, 7, 9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2, 7, 10), fm_spectrum(high_passed), 'Spectrum')

    plt.show()

    # print("lpf & spectrum")
    # plt.imshow(low_passed, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # plt.imshow(fm_spectrum(low_passed), cmap='gray')
    # plt.axis('off')
    # plt.show()

    # print("hpf & spectrum")
    # plt.imshow(high_passed, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # plt.imshow(fm_spectrum(high_passed), cmap='gray')
    # plt.axis('off')
    # plt.show()
