import gzip
import numpy as np
import scipy.stats as stats
import utils
from skimage.transform import resize

def extract_data(filename, num_images):
    IMAGE_SIZE = 28
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

class mnist_documents(object):

    # Create an object to generate document-like images with mnist digits.
    def __init__(self, num_row, num_images_row, scaling_factor, empty_percent, **kwargs):
        # Arguments:
        #   scaling_factor: 0<=float<=1, the digits will be scaled according to a normal distribution
        #   empty_percent: 0<=float<=1, the percentage of empty spots in each generated image
        assert 0 <= scaling_factor and scaling_factor <= 1, 'Enter a float between 0 and 1 inclusive'

        self.num_row = num_row
        self.num_images = num_images_row

        self.scaling_factor = scaling_factor
        self.scaling_range = kwargs.get('scaling_range', 0.3)

        self.empty_percent = empty_percent
        self._clip_a = (0.4 - self.scaling_factor)/self.scaling_range
        self._clip_b = (1 - self.scaling_factor)/self.scaling_range
        self._image_size = 28
        self.images, self.labels = self._add_empty_images()

        self.num_sample = self.images.shape[0]

    @property
    def image_size(self):
        return self._image_size

    @staticmethod
    def resize_pad_image(image, scale):
        # resize and scale the image
        output_shape = [int(image.shape[0]*scale)]*2
        scaled_image = resize(image, output_shape)

        # calculate the dimensional differences between scaled and original image
        # and pad the image to retain 28*28 size
        dim_difference = image.shape[0] - scaled_image.shape[0]
        pad_left, pad_right = np.floor(dim_difference / 2), np.ceil(dim_difference / 2)
        padded_image = np.pad(scaled_image, (int(pad_left), int(pad_right)), mode='constant')

        return padded_image

    @staticmethod
    def bounding_box_location(digit_image, mode='width'):
        # Finds the simplest bonding box for each digit in the mnist dataset
        # Arguments:
        # digit_image: array, simple 28*28 mnist like image
        # mode: string, either "width" or "coord"
        # return:
        # the bottom left and top right coordinates of the bounding box

        x, y = digit_image.nonzero()
        x1 = x.min()
        x2 = x.max()
        y1 = y.max()
        y2 = y.min()

        height = y1 - y2
        width = x2 - x1

        x_center = int(width / 2)
        y_center = int(height / 2)

        return np.array([x_center, y_center,
                         height/digit_image.shape[0], width/digit_image.shape[0]])

    def _add_empty_images(self):
        # extract and append an empty image

        temp_iamge = extract_data(r'data\train-images-idx3-ubyte.gz', 10000)
        temp_labels = extract_labels(r'data\train-labels-idx1-ubyte.gz', 10000)

        image = np.concatenate([np.zeros((1, self._image_size, self._image_size)), temp_iamge])
        labels = np.append(np.nan, temp_labels)

        return image, labels

    def generate(self):
        # Generate an image of a fixed size composed of randomly selected digits
        # return the image as well as the coordinates of each digits
        # Arguments:
        #    num_images: int, the number of digits to be included in the output image.
        num_row = self.num_row
        num_col = self.num_images
        image_size = self._image_size
        scaling_factor = self.scaling_factor
        scaling_range = self.scaling_range

        glued_images = np.zeros((num_row * image_size, num_col * image_size))

        glued_data = np.zeros((num_row, num_col, 15))



        if scaling_factor != 1:
            for j in range(num_row):

                # generate random scaling factors from N(scaling_factor, 0.3)
                random_index = np.random.choice(range(0, self.num_sample), num_col,
                                                p=[self.empty_percent] + [(1 - self.empty_percent) / (self.num_sample - 1)] * (
                                                self.num_sample - 1))


                scalings = scaling_range * stats.truncnorm(self._clip_a,self._clip_b).rvs(num_col) + scaling_factor

                for i, index in enumerate(random_index):

                    image = self.images[index]
                    label = self.labels[index] if index != 0 else None

                    if label is not None:
                        #resize and pad the image
                        padded_image = mnist_documents.resize_pad_image(image, scalings[i])

                        # now find the bounding box for the digit
                        bounding_box = mnist_documents.bounding_box_location(padded_image, mode='width')

                        #fit the image in and the bounding boxes
                        glued_images[image_size * j: image_size * (j + 1),
                                     image_size * i: image_size * (i + 1)] = padded_image
                        glued_data[j, i, 0] = 1
                        glued_data[j ,i, 1:5] = bounding_box
                        glued_data[j ,i, int(label) + 5] = 1

                    else:
                        glued_images[image_size * j: image_size * (j + 1),
                                     image_size * i: image_size * (i + 1)] = image

        else:
            for j in range(num_row):
                for i, index in enumerate(random_index):
                    image = self.images[index]
                    label = self.labels[index] if index != 0 else None

                    glued_images[image_size * j: image_size * (j + 1),
                                 image_size * i: image_size * (i + 1)] = image

                    if label is not None:
                        bounding_box = generate_images.bounding_box_location(image, mode='width')

                        glued_data[j ,i, 0] = 1
                        glued_data[j, i, 1:5] = bounding_box
                        glued_data[j, i, int(label) + 5] = 1

                        # bounding_boxes[i,:] = bounding_box

        return glued_images, glued_data


def generate_documents(nb_documents, nb_rows, scale=0.9, empty_percent=0.5):
    images = np.zeros((nb_documents, nb_rows * 28, nb_rows * 28, 1))
    labels = np.zeros((nb_documents, nb_rows**2 * 15))

    mnist_class = mnist_documents(nb_rows, nb_rows, scale, empty_percent)

    for i in range(nb_documents):
        if i%100==0:
            print(i)
        image, label = mnist_class.generate()
        images[i, :, :, 0] = image
        labels[i, :] = utils.unwrap_data(label)

    return images, labels
