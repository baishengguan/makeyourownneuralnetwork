import numpy
import matplotlib.pyplot
# scipy.ndimage for rotating image arrays
import scipy.ndimage

# open the CSV file and read its contents into a list
data_file = open("data/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# which record will be use
record = 6

# scale input to range 0.01 to 1.00
all_values = data_list[record].split(',')
scaled_input = ((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01).reshape(28, 28)

print(numpy.min(scaled_input))
print(numpy.max(scaled_input))

# plot the original image
matplotlib.pyplot.imshow(scaled_input, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# create rotated variations
# rotated anticlockwise by 10 degrees
inputs_plus10_img = scipy.ndimage.rotate(scaled_input, 10.0, cval=0.01, order=1, reshape=False)
# rotated clockwise by 10 degrees
inputs_minus10_img = scipy.ndimage.rotate(scaled_input, -10.0, cval=0.01, order=1, reshape=False)

print(numpy.min(inputs_plus10_img))
print(numpy.max(inputs_plus10_img))

# plot the +10 degree rotated variation
matplotlib.pyplot.imshow(inputs_plus10_img, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# plot the -10 degree rotated variation
matplotlib.pyplot.imshow(inputs_minus10_img, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
