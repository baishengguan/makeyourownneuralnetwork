import numpy
import matplotlib.pyplot as plt

# open the CSV file and read its contents into a list
data_file = open("data/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# print(len(data_list))
# print(data_list[0])

# take the data from a record rearrange it into a 28*28 array and plot it as an image
all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.09) + 0.01
print(scaled_input)

# output nodes is 10(example)
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)
