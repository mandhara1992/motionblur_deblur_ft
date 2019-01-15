#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplot', 'inline')


# # Elimination of Motion Blur using Fourier Transform
#  we will mainly try to understand the importance of fourier transform (magnitude + phase) and it's applications in the field of image processing.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from PIL import Image
from pyblur import *

from scipy.signal import convolve2d as conv2


# In[4]:


img = color.rgb2gray(data.load('test.jpg'))


f = np.fft.fft2(img) #Fourier Transform
fshift = np.fft.fftshift(f) 
fft_mag = 20*np.log(np.abs(fshift)) #Fourier Magnitude

fft_phase = np.angle(f) #Fourier Phase


# In[5]:


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15), sharex=True, sharey=True)

plt.gray()

ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Image')

ax[1].imshow(fft_mag)
ax[1].axis('off')
ax[1].set_title('Fourier Magnitude')

ax[2].imshow(fft_phase)
ax[2].axis('off')
ax[2].set_title('Fourier Phase')

fig.tight_layout()

plt.show()


# > As a result of the fourier transform, we will have two components: amplitude and phase spectrum. 
# > By the way, many people forget about the phase. 
# > Please note that the amplitude spectrum is shown in a logarithmic scale, because its values vary tremendously - by several orders of magnitude, in the center the values are maximum (millions) and they quickly decay almost to zero ones as they are getting farther from the center. Due to this very fact inverse filtering will work only in case of zero or almost zero noise values.

# In[6]:


#m_blurred = LinearMotionBlur(img, 9, 0, 'full') 
kernel = LineKernel(9, 60, 'full') #Kernel for sythetic linear motion
#kernel


# ##### For testing purposes, we have added a synthetic blur to a clear image. 
# > The 'astro' variable is a 2d array formed by the convolution of the original image and a motion blur kernel(LineKernel) formed above. Also some noise & distortion has been added to observe the relative performance in presence of noise. (Comment line2 to remove noise.)

# In[7]:


astro = conv2(img, kernel, 'same') #Convoluting Image with a motion blur
astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape) #addition of random noise+distortion


# > Wiener filter uses Fourier Transformations for deconvolution of an image. It makes it much easier and computationally possible to do so. Below is the equation of the following:
# 
# <img src="equation.png">

# In[8]:


deblur, _ = restoration.unsupervised_wiener(astro, kernel) #Fourier Wiener


# In[9]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

ax[0].imshow(astro, vmin=deblur.min(), vmax=deblur.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deblur)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()

