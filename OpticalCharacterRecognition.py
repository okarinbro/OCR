import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


class OpticalCharacterRecognition:
    def __init__(self, file_path):
        # invert image!
        self.image = scipy.ndimage.imread(file_path, mode='L')
        # inversion
        self.image = 255 - self.image
        # TO DO -> load patterns
        self.letters_positions = {}

    def get_correlation(self, image, pattern):
        return np.real(
            np.abs(np.fft.ifft2(np.multiply(np.fft.fft2(image), np.fft.fft2(np.rot90(pattern, 2), image.shape)))))

    def print_image(self, image):
        plt.imshow(image)
        plt.show()

    def get_letter_position(self):
        pass

    def remove_letter(self):
        pass

    def get_letters_match(self):
        pass
