#Hello
#This code is used to identify corroded area in an image.

import numpy as np
import cv2
import matplotlib.pyplot as plt


# The funtion of downsampling gray intensity value of the image
def downsample(input):
    new_image = input.copy()
    max_gray_level = maxGrayLevel(input)
    height = input.shape[0]
    width = input.shape[1]

    # if the max gray scale of the image is larger than gray_level that user set,
    # then the max gray intensity will be downsampled to reduce the size of the gray level co-occurrence matrix
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                new_image[j][i] = new_image[j][i] * gray_level / max_gray_level
    return new_image

# Pad the image to square for easy extraction of patches with size of 15(pixels)*15(pixels)
def pad_image( image ):
    height = image.shape[0]
    width = image.shape[1]
    hei = height % 15
    wid = width % 15
    new_height = height+15-hei
    new_width = width+15-wid
    if new_height >= new_width:
        img = cv2.copyMakeBorder(image, 0, 15-hei, 0, new_height-width, cv2.BORDER_CONSTANT, value=[255,255,255])
    else:
        img = cv2.copyMakeBorder(image, 0,new_width-height , 0, 15-wid, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return img

# Cut the input image into patches with size of 15(pixels)*15(pixels)
def cut_image(image):
    height = image.shape[0]
    width = image.shape[1]
    hei_num = int(height/15)
    wid_num = int(width/15)
    crop_list=[]
    count=0
    for j in range(wid_num):
        for i in range(hei_num):
            count+=1
            cropped=image[(i*15):((i*15)+14),(j*15):((j*15)+14)]
            crop_list.append(cropped)

    return crop_list


# Calculate the maximum gray scale value of the image
def maxGrayLevel(image):
    max_gray_level = 0
    height = image.shape[0]
    width = image.shape[1]
    for y in range(height):
        for x in range(width):
            if image[y][x] > max_gray_level:
                max_gray_level = image[y][x]
    return max_gray_level+1

# Calculate the gray-level co-occurrence matrix
def getGLCM(image, x, y):
    srcdata = image.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    height = image.shape[0]
    width = image.shape[1]

    for j in range(height - y):
        for i in range(width - x):
            rows = srcdata[j][i]
            cols = srcdata[j + y][i + x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


# Calculate the energy of the gray-level co-occurrence matrix
def Feature(glcm):
    Energy = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            if glcm[i][j] > 0.0:
                Energy += glcm[i][j] * glcm[i][j]
    return  Energy


# Read the input image and convert to gray image and HSV image
img = cv2.imread('/Users/pro/Desktop/Summer Project/Identification of corroded area/Images/test9.png')
img1 =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_height = img.shape[0]
original_width = img.shape[1]
print("The size of the original image:")
print(img.shape)

img2 = pad_image(img1)
print("The size of the padded image:")
print(img2.shape)

gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
HSV = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

# Read the training set
training_data1 = cv2.imread('/Users/pro/Desktop/Summer Project/Identification of corroded area/Training data/Training1.png')
training_data2 = cv2.imread('/Users/pro/Desktop/Summer Project/Identification of corroded area/Training data/Training2.png')
training_data1 = cv2.cvtColor(training_data1,cv2.COLOR_BGR2RGB)
training_data2 = cv2.cvtColor(training_data2,cv2.COLOR_BGR2RGB)
training_data = np.concatenate([training_data1, training_data2], axis=1)

# Calculate the maximum gray scale value of the image
max_gray_level = maxGrayLevel(gray)
print("The max gray level of the image is:")
print(max_gray_level)

# Set the maximum gray level
gray_level = 31

# Generate a new image through the downsampling process
new_image = downsample(gray)

# Cut image into a number of patches with size of 15(pixels)*15(pixels)
gray_list = cut_image(new_image)
HSV_list = cut_image(HSV)
img_list = cut_image(img2)

count_patch=0
# Calculate the energy of gray-level co-occurrence matrix of each patch and select those rough ones
select_patch =[]
for patch1 in gray_list:
    glcm = getGLCM(patch1,5,0)
    energy = Feature(glcm)
    if energy < 0.2:  # Users can set threshold of energy of GLCM to obtain the best identification result
        select_patch.append('True')
        count_patch+=1
    else:
        select_patch.append('False')
print("The total number of patches:")
print(len(select_patch))
print("The number of patches that have passed the roughness stage:")
print(count_patch)


# Generate the HS bi-dimensional histogram of training set
hsv = cv2.cvtColor(training_data, cv2.COLOR_RGB2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Calculate the peak value of HS bi-dimensional histogram
max_value = 0
for row in hist:
    for value in row:
        if value >= max_value:
            max_value = value
print("The peak of HS bi-dimensional histogram of training set:")
print(max_value)

# Label the pixels of each patch depending on the percentage of the height of corresponding histogram bin and the peak value
for (value1,patch1,patch2) in zip(select_patch,HSV_list,img_list):
    if(value1=='True'):
        for(raw1,raw2) in zip(patch1,patch2):
            for(pixel1,pixel2) in zip(raw1,raw2):
                if((pixel1[1]<50 and (pixel1[2]<200 and pixel1[2]>50)) or (pixel1[1]>50 and pixel1[2]>50)):
                    h = pixel1[0]
                    s = pixel1[1]
                    v = pixel1[2]
                    value = hist[h][s]
                    rate = value/max_value
                    if rate <=1 and rate > 0.75:
                        pixel1[0]=160
                        pixel1[1]=200
                        pixel1[2]=200
                        pixel2[0]=255
                        pixel2[1]=0
                        pixel2[2]=0

                    elif rate <= 0.75 and rate > 0.5:
                        pixel1[0] = 20
                        pixel1[1] = 200
                        pixel1[2] = 200
                        pixel2[0] = 230
                        pixel2[1] = 117
                        pixel2[2] = 80

                    elif rate <= 0.5 and rate > 0.25:
                        pixel1[0] = 60
                        pixel1[1] = 255
                        pixel1[2] = 255
                        pixel2[0] = 0
                        pixel2[1] = 255
                        pixel2[2] = 0

                    elif rate <=0.25 and rate >0.1:
                        pixel1[0] = 110
                        pixel1[1] = 180
                        pixel1[2] = 180
                        pixel2[0] = 0
                        pixel2[1] = 0
                        pixel2[2] = 255

count=0
# Merge all the processed patches to get the result image
height = img2.shape[0]
width = img2.shape[1]
hei_num = int(height / 15)
wid_num = int(width / 15)
for j in range(wid_num):
    for i in range(hei_num):
        HSV[(i * 15):((i * 15) + 14), (j * 15):((j * 15) + 14)] = HSV_list[count]
        img2[(i * 15):((i * 15) + 14), (j * 15):((j * 15) + 14)] = img_list[count]
        count+=1


img2=img2[0:original_height,0:original_width]

plt.subplot(1,2,1),plt.imshow(img1)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(img2)
plt.title('Precision Result'), plt.xticks([]), plt.yticks([])

plt.show()
