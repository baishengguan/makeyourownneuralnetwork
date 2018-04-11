# helper to load data from PNG image files
import scipy.misc
# glob helps select multiple files using patterns
import glob
import numpy

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
    print("loading ...", image_file_name)
    # using the filename to set the correct label
    label = int(image_file_name[-5:-4])
    # load image data from png files into an array
    img_array = scipy.misc.imread(image_file_name,  flatten=True)
    # reshape from 28*28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    # append label and image data to test data set
    record = numpy.append(label, img_data)
    # print(record)
    our_own_dataset.append(record)
    pass
