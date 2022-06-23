import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

file_name = 'yummy_macarons.jpg'

# single dimension array
my_array = np.array([1.1, 9.2, 8.1, 4.7])
my_array.shape
my_array[0]
my_array.ndim

# 2-dimensional array(ie. a matrix)
array_2d = np.array([[1, 2, 3, 9], [5, 6, 7, 8]])
array_2d.shape
array_2d[1, 2]  # accessing a particular value, in this example the 3rd value in the 2nd row
array_2d[0, :]  # access all values in the first row

# N-dimensions
array_n = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[7, 86, 6, 98], [5, 1, 0, 4]], [[5, 36, 32, 48], [97, 0, 27, 18]]])
array_n.shape  # 3 dimensions, 2 columns, 4 rows
array_n[2, 1, 3]  # dimension, row within dimension, value within row
array_n[2, 1:]  # 1 dimension vector with values [97, 0, 27, 18]
array_n[:, :, 0]  # all the first elements within the 3 dimensional array(3 axis array)

a_range = np.arange(10, 30)  # create a vector with values ranging from 10 to 29
a_range[-3:]  # create an array containing the last 3 values
a_range[3:6]  # create a subset with only the 4th, 5th and 6th values
a_range[12:]  # create a subset containing all values except for the first 12 numbers
a_range[::2]  # create a subset that only contains even numbers
a_range[::-1]  # create a subset that is reverse in order or np.flip()
b = ([6, 0, 9, 0, 0, 5, 0])
nz_indices = np.nonzero(b)  # print all non-zero elements in the array
np.random.rand(3, 3, 3)  # create a 3x3x3 array with random numbers
np.linspace(0, 100, num=9).astype(int)  # create a vector of size 9 with values evenly spaced out between 0 and 100
# create another vector of size 9 with values between - 3 and 3
rng_2 = np.linspace(-3, 3, num=9).astype(int)
plt.plot(rng_2, rng_2)

noise = np.random.rand(128, 128, 3)
plt.imshow(noise)

# matrix multiplication
a1 = np.array([[1, 3],
               [0, 1],
               [6, 2],
               [9, 7]])

b1 = np.array([[4, 1, 3],
               [5, 8, 5]])

np.matmul(a1, b1)

img = misc.face()
plt.imshow(img)
img.dtype  # find datatype of img
type(img)
img.shape  # find shape of img
img.ndim

# convert image to gray
img_alter = img / 255
grey_vals = np.array([0.2126, 0.7152, 0.0722])
img_gray = np.matmul(img_alter, grey_vals)
plt.imshow(img_gray, cmap='gray')

# flip image
img_gray_flipped = np.flip(img_gray)
plt.imshow(img_gray_flipped, cmap='gray')

# rotate color image
img_rotated = np.rot90(img, k=1)
plt.imshow(img_rotated)

# invert color image
img_invert = np.invert(img)
plt.imshow(img_invert)

# opening image
my_img = Image.open(file_name)
img_array = np.array(my_img)
img_array.ndim
img_array.shape
plt.imshow(254 - img_array)