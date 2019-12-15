# coding:utf-8
import cv2
import numpy as np
import os
import struct

# 参考mnist的结构读取
# http://yann.lecun.com/exdb/mnist/


path = "data/train-labels-idx1-ubyte"

with open(path, "rb") as f:
    # 读取二进制数据
    bin_data = f.read()
    offset = 0
    # 解析文件头信息，依次为魔数和标签数
    fmt_header = ">ii"
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, num_labels: %d' % (magic_number, num_labels))
    # 解析数据集,offset 是 magic_number, num_labels 过后的位置
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    # 试着读取第一个标签
    label = struct.unpack_from(fmt_image, bin_data, offset)
    print(label)
    # label 只占一个比特，因此加一代表着第二个标签
    label = struct.unpack_from(fmt_image, bin_data, offset + 1)
    print(label)

path = "data/train-images-idx3-ubyte"

with open(path, "rb") as f:
    # 读取二进制数据
    bin_data = f.read()
    offset = 0
    # 解析文件头信息，依次为魔数和标签数
    fmt_header = ">iiii"
    magic_number, number_of_images, number_of_rows, number_of_columns = \
        struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, number_of_images: %d, number_of_rows: %d, number_of_columns:%d'
          % (magic_number, number_of_images, number_of_rows, number_of_columns))
    # 解析数据集,offset 是 magic_number, num_labels 过后的位置
    offset += struct.calcsize(fmt_header)
    image_size = number_of_rows * number_of_columns
    fmt_image = '>' + str(image_size) + 'B'  # '>784B'的意思就是用大端法读取784个unsigned byte

    # 试着读取第一张图片
    img = struct.unpack_from(fmt_image, bin_data, offset)
    print(img)
    # 图片占28*28比特，因此加28*28代表着第二张图片
    img = struct.unpack_from(fmt_image, bin_data, offset + 784)
    print(img)
    print(len(img))

    img = np.array(img).reshape((28, 28))
    cv2.imwrite("1.png", img)
