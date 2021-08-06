#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:04:39 2020

"""
import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt
#import matplotlib.image as mpltimg 
import numpy as np



#_______ 3 user defined functions which you have to code their functionality_________


def convolution2D(img, kernel, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
        # 0 - zero padding
        # 1 - duplicate boundary pixels for padding
        # 2 - padding is done by mirroring the pixels
        
    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image
    new_img = np.zeros_like(img)

    # Add one padding if want zero paddind then write np.zeros()
    img_pad = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    img_pad[1:-1, 1:-1] = img

    #for conv
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            new_img[j, i]=(kernel * img_pad[j: j+3, i: i+3]).sum()
    
    return new_img
    


def medianFiltering(img, kernel_size, padding_type):
    # write your code here
    # padding_type can take values 0, 1 or 2
        # 0 - zero padding
        # 1 - duplicate boundary pixels for padding
        # 2 - padding is done by mirroring the pixels
        
    # should handle kernel of any size but odd values only eg. 5x5, 7x7
    # image is a grayscale image
    
    # Add one padding if want zero paddind then write np.zeros()
    '''f(x-1, y-1) f(x-1, y) f(x-1, y+1)
    f(x, y-1) f(x, y) f(x, y+1)
    f(x+1, y-1) f(x+1, y) f(x+1, y+1)'''

    img_pad = np.ones((img.shape[0] + 2, img.shape[1] + 2))
    img_pad[1:-1, 1:-1] = img

    h ,w =img.shape
    med_image=np.zeros([h,w])
    for i in range(1,h-1):
        for j in range(1,w-1):
            flag = [img[i-1, j-1], 
                    img[i-1, j], 
                    img[i-1, j + 1], 
                    img[i, j-1], 
                    img[i, j], 
                    img[i, j + 1], 
                    img[i + 1, j-1], 
                    img[i + 1, j], 
                    img[i + 1, j + 1]] 
            flag=sorted(flag)
            med_image[i,j]=flag[kernel_size]
    
    return med_image



# You can club the above two functions/write any other functions additionally if you wish

def computePSNR(image1, image2):

    mse = np.mean((image1 - image2) ** 2) 
    if(mse == 0):  
        return 100 #same img
    psnr = 20 * log10(255.0 / sqrt(mse)) 
       
    return psnr


# _____________________main program begins here___________________

def main():
    # reading a noisy image
    noisy_image = cv2.imread('images/noisy_image.jpg',0) # 0 for only 0 to 255

    original_image = cv2.imread('images/original.jpg',0) # 0 for only 0 to 255


# _____________________________________________________________________
# Average filter kernel
    kernel = 1/9 * np.array([[ 1, 1, 1],
                          [ 1, 1, 1],
                          [ 1, 1, 1]]) 




    low_pass_filtered_image = convolution2D(noisy_image, kernel, 1)

    
    avg_psnr = computePSNR(original_image,low_pass_filtered_image)


    med_filtered_image = medianFiltering(noisy_image, 4, 1)
    
    med_psnr = computePSNR(original_image,med_filtered_image)
    
    
    
# _____________________________________________________________________
# reading a blurry image
    blurry_image = cv2.imread('images/blurry_image.jpg',0) # 0 for only 0 to 255
    
    
    # Laplacian filter kernel
    kernel =        np.array([[ 1, 1, 1],
                              [ 1, -8, 1],
                              [ 1, 1, 1]]) 


    
    laplacian_filtered_image = convolution2D(blurry_image, kernel, 1)
    
# perform the addition as in Eqn. 3.6.7 to obtain the sharpened image

    # Sharpened filter kernel
    kernel = np.array([[-1,-1,-1],
                        [-1,9,-1],
                        [-1,-1,-1]])
    
    sharpened_image= convolution2D(blurry_image, kernel, 1)
    
    
    
# _____________________________________________________________________
# Code to display the images
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    
    ax = axes.ravel()
    
    ax[0].imshow(noisy_image, cmap='gray')
    ax[0].set_title("Noisy image")
    ax[0].set_axis_off()
    
    ax[1].imshow(low_pass_filtered_image, cmap='gray')
    ax[1].set_title("Low Pass Filter Output")
    ax[1].set_axis_off()
    ax[1].text(x=40, y=230, s="PSNR = %1.2f db" %avg_psnr)
    
    ax[2].imshow(med_filtered_image, cmap='gray')
    ax[2].set_title("Median Filter Output")
    ax[2].set_axis_off()
    ax[2].text(x=40, y=230, s="PSNR = %1.2f db" %med_psnr)
    
    
    ax[3].imshow(blurry_image, cmap='gray')
    ax[3].set_title("Blurry Input Image")
    ax[3].set_axis_off()
    
    
    ax[4].imshow(laplacian_filtered_image, cmap='gray')
    ax[4].set_title("Laplacian Filter Output")
    ax[4].set_axis_off()
    
    
    ax[5].imshow(sharpened_image, cmap='gray')
    ax[5].set_title("Sharpened Image")
    ax[5].set_axis_off()
    
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

