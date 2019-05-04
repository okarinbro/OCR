import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import string


class OpticalCharacterRecognition:
    def __init__(self, file_path):
        self.image = scipy.ndimage.imread(file_path, mode='I')
        self.print_image()
        self.letters_positions = {}
        self.patterns = self.letters_patterns()

    def letters_patterns(self):
        path = 'data/'
        letter_patterns = {}
        for letter in string.ascii_lowercase:
            letter_patterns[letter] = (255 - scipy.ndimage.imread(path + letter + '.bmp', mode='I'))
        return letter_patterns

    def get_correlation(self, pattern, coefficient=0.95):
        tmp_fft = np.fft.fft2(self.invertImage(self.image))
        m = np.multiply(tmp_fft, np.fft.fft2(np.rot90(pattern, 2), tmp_fft.shape))
        corr = np.abs(np.fft.ifft2(m)).astype(float)
        corr[corr < coefficient * np.amax(corr)] = 0
        corr[corr != 0] = 254
        # print(corr)
        return corr

    def print_image(self):
        plt.imshow(self.image)
        plt.show()

    def get_letters_match(self):
        for l in string.ascii_lowercase:
            letter_pattern = self.patterns[l]
            corr = self.get_correlation(letter_pattern)
            self.letters_positions[l] = self.get_letter_positions(corr, letter_pattern)
            print("letter: ", l, "positions: ", self.letters_positions[l])
            self.remove_letter_from_image(self.letters_positions[l], l)
            # self.print_image()

    def get_letter_positions(self, correlation, letter):
        pos = []
        i, j = letter.shape
        x_coord, y_coord = (-1 * i, -1 * j)
        for (x, y), cor_val in np.ndenumerate(correlation):
            if cor_val > 0.0 and not (x_coord + i > x and y_coord + j > y):
                pos.append((x, y))
                x_coord, y_coord = x, y
        return pos

    def remove_letter_from_image(self, pos, letter):
        # color of background
        for x, y in pos:
            i, j = self.patterns[letter].shape
            self.image[x - i:x, y - j:y] = 255

    def invertImage(self, image):
        image = 255 - image
        return image

    def to_text(self):
        positions = []
        for k, v in self.letters_positions.items():
            print(k, 'key')
            for coords in v:
                print(coords)
                positions.append((k, coords[0], coords[1]))
        positions = sorted(positions, key=lambda tri_tuple: (tri_tuple[1], tri_tuple[2]))
        print('sorted positions: ', positions)
        result_text = ""
        for i in range(len(positions) - 1):
            result_text += positions[i][0]
            result_text += chr(32)
        return result_text


if __name__ == '__main__':
    ocr = OpticalCharacterRecognition('data\sample_text.png')
    ocr.get_letters_match()
    ocr.print_image()
    print(ocr.to_text())
