import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import string


class OpticalCharacterRecognition:
    def __init__(self, file_path):
        # invert image!
        self.image = scipy.ndimage.imread(file_path, mode='L')
        # inversion
        self.image = 255 - self.image
        self.letters_positions = []
        self.patterns = self.letters_patterns()

    def letters_patterns(self):
        path = 'data/'
        letter_patterns = {}
        for letter in string.ascii_lowercase:
            letter_patterns[letter] = (255 - scipy.ndimage.imread(path + letter + '.bmp'))
        return letter_patterns

    def get_correlation(self, image, pattern):
        return np.real(
            np.abs(np.fft.ifft2(np.multiply(np.fft.fft2(image), np.fft.fft2(np.rot90(pattern, 2), image.shape)))))

    def print_image(self, image):
        plt.imshow(image)
        plt.show()

    def get_letters_match(self):
        pass

    def get_letter_positions(self, correlation, letter):
        pos = []
        lw, lh = letter.shape
        x_coord, y_coord = (-1 * lw, -1 * lh)
        for (x, y), cor_val in np.ndenumerate(correlation):
            if cor_val > 0.0 and not (x_coord + lw > x and y_coord + lh > y):
                pos.append((x, y))
                x_coord, y_coord = x, y
        return pos

    def remove_letter_from_image(self, pos, letter):
        for x, y in pos:
            i, j = self.patterns[letter].shape
            self.image[x - i:x, y - j:y] = 255


