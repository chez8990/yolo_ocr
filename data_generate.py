import gzip
import numpy as np
import scipy as sp

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

class generate_images(object):

    # Create an object to generate document-like images with mnist digits.
    def __init__(self, scaling_factor, empty_percent, **kwargs):
        # Arguments:
        #   scaling_factor: 0<=float<=1, the digits will be scaled according to a normal distribution
        #   empty_percent: 0<=float<=1, the percentage of empty spots in each generated image
        assert 0 <= scaling_factor and scaling_factor <= 1, 'Enter a float between 0 and 1 inclusive'

        self.scaling_factor = scaling_factor
        self.scaling_range = kwargs.get('scaling_range', 0.3)
        self.empty_percent = empty_percent
        self._clip_a = -self.scaling_factor/self.scaling_range
        self._clip_b = (1-self.scaling_factor)/self.scaling_range
        self._image_size = 28
        self.images, self.labels = self._add_empty_images()

        self.num_sample = self.images.shape[0]

    @property
    def image_size(self):
        return self._image_size

    @staticmethod
    def resize_pad_image(image, scale):
        # resize and scale the image
        scaled_image = sp.misc.imresize(image, scale)

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

        if mode == 'width':
            return np.array([x1, y1, x2 - x1, y1 - y2])
        else:
            return np.array([[x1, y1], [x2, y2]])

    def _add_empty_images(self):
        # extract and append an empty image

        temp_iamge = extract_data(r'data/train-images-idx3-ubyte.gz', 10000)
        temp_labels = extract_labels(r'data/train-labels-idx1-ubyte.gz', 10000)

        image = np.concatenate([np.zeros((1, self._image_size, self._image_size)), temp_iamge])
        labels = np.append(np.nan, temp_labels)

        return image, labels

    def image_gluing(self, num_images):
        # Generate an image of a fixed size composed of randomly selected digits
        # return the image as well as the coordinates of each digits
        # Arguments:
        #    num_images: int, the number of digits to be included in the output image.

        image_size = self._image_size
        scaling_factor = self.scaling_factor
        scaling_range = self.scaling_range

        random_index = np.random.choice(range(0,self.num_sample), num_images,
                                        p=[self.empty_percent]+[(1-self.empty_percent)/(self.num_sample-1)]*(self.num_sample-1))

        glued_images = np.zeros((image_size, image_size * num_images))
        # glued_labels = np.zeros((num_images, 10))
        glued_data = np.zeros((num_images, 15))
        # bounding_boxes = np.zeros((num_images, 4))

        if scaling_factor != 1:
            # generate random scaling factors from N(scaling_factor, 0.3)
            scalings = scaling_range * sp.stats.truncnorm(self._clip_a,self._clip_b).rvs(num_images) + scaling_factor
            for i, index in enumerate(random_index):
                image = self.images[index]
                label = self.labels[index] if index != 0 else None

                if label is not None:
                    #resize and pad the image
                    padded_image = generate_images.resize_pad_image(image, scalings[i])

                    # now find the bounding box for the digit
                    bounding_box = generate_images.bounding_box_location(padded_image, mode='width')

                    #fit the image in and the bounding boxes
                    glued_images[:, image_size * i: image_size * (i+1)] = padded_image
                    glued_data[i, 0] = 1
                    glued_data[i, 1:5] = bounding_box
                    glued_data[i, int(label) + 5] = 1

                else:
                    glued_images[:, image_size * i: image_size * (i+1)] = image

        else:
            for i, index in enumerate(random_index):
                image = self.images[index]
                label = self.labels[index] if index != 0 else None

                glued_images[:, image_size * i] = image
                if label is not None:
                    bounding_box = generate_images.bounding_box_location(image, mode='width')

                    glued_data[i, 0] = 1
                    glued_data[i, 1:5] = bounding_box
                    glued_data[i, int(label) + 5] = 1

                    bounding_boxes[i,:] = bounding_box

        return glued_images, np.expand_dims(glued_data, axis=0)

def document_like_digits(num_rows, num_image_per_row, scaling_factor, empty_percent):
    # Generate word like documents consisting mnist digits

    gi = generate_images(scaling_factor, empty_percent)

    images = np.zeros((gi.image_size * num_rows, gi.image_size * num_image_per_row))
    data_array = np.zeros((num_rows, num_image_per_row, 15))

    for i in range(num_rows):
        image, data = gi.image_gluing(num_image_per_row)

        images[i*gi.image_size:(i+1)*gi.image_size] = image
        data_array[i, :, :] = data

    return images, data_array