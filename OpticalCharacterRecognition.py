import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import string


class OpticalCharacterRecognition:
    def __init__(self, file_path, order):
        self.image = scipy.ndimage.imread(file_path, mode='I')
        self.print_image()
        self.letters_positions = {}
        self.patterns = self.letters_patterns()
        self.order = [('x', 1), ('z', 1), ('w', 1), ('y', 1), ('f', 1), ('k', 1), ('g', 8),
                      ('b', 2), ('p', 8), ('m', 1), ('a', 1), ('t', 5), ('u', 10), ('s', 1),
                      ('j', 2), ('v', 2), ('h', 2), ('q', 12), ('d', 8), ('l', 8), ('e', 8),
                      ('n', 3), ('r', 3), ('i', 4), ('o', 8), ('c', 10)]

    def letters_patterns(self):
        path = 'data/'
        letter_patterns = {}
        for letter in string.ascii_lowercase:
            letter_patterns[letter] = self.invertImage(scipy.ndimage.imread(path + letter + '.bmp', mode='I'))
        return letter_patterns

    def get_correlation(self, image, pattern, extra=0.0, coefficient=0.87):
        coefficient += extra
        fi = np.fft.fft2(self.invertImage(image))
        fp = np.fft.fft2(np.rot90(pattern, 2), fi.shape)
        m = np.multiply(fi, fp)
        corr = np.fft.ifft2(m)
        corr = np.abs(corr)
        corr = corr.astype(float)
        corr[corr < coefficient * np.amax(corr)] = 0
        corr[corr != 0] = 254
        return corr

    def print_image(self):
        plt.imshow(self.image)
        plt.show()

    def get_letters_match(self):
        for l, freq in self.order:
            letter_pattern = self.patterns[l]
            extra = 8 * freq / 1000
            corr = self.get_correlation(self.image, letter_pattern, extra)
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
        text = ""
        pos = []
        for k, v in self.letters_positions.items():
            for tri_pos in v:
                pos.append((k, tri_pos[0] - tri_pos[0] % 100, tri_pos[1]))
        pos = sorted(pos, key=lambda e: (e[1], e[2]))
        endline = 100  # look at the picture
        for i in range(len(pos) - 1):
            text += pos[i][0]
            if abs(pos[i][1] - pos[i + 1][1]) >= endline:
                text += '\n'
        return text


if __name__ == '__main__':

    def get_letter_order(text):
        order_map = {}
        for l in text:
            if l in string.ascii_lowercase:
                order_map[l] = order_map.get(l, 0) + 1
        pos = sorted(order_map.items(), key=lambda tuple: (tuple[1]))
        return pos


    text = "sample text created in order to test ocr program, some pangram: jived fox nymph grabs quick waltz."
    ocr = OpticalCharacterRecognition('data\sample_text.png', get_letter_order(text))
    ocr.get_letters_match()
    ocr.print_image()
    print(ocr.to_text())
