#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


img = cv2.imread(r"C:\Users\yjosh\Desktop\Face Images\Mark Zuckerberg.jfif")


# In[3]:


grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[4]:


invert_img=cv2.bitwise_not(grey_img)


# In[5]:


blur_img=cv2.GaussianBlur(invert_img, (111,111),0)


# In[6]:


invblur_img=cv2.bitwise_not(blur_img)


# In[7]:


sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)


# In[8]:


plt.imshow(img[:,:,::-1])
plt.show()


# In[9]:


RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)
plt.show()


# In[11]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Original image', size=18)
plt.imshow(RGB_img)
plt.subplot(1,2,2)
plt.title('Sketch', size=18)
rgb_sketch=cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_sketch)
plt.show()


# In[12]:


plt.imshow(img)
plt.axis(False)
plt.show()


# In[13]:


if(img is not None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 1)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 11)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    plt.title('Cartoonify image', size=18)
    plt.imshow(cartoon)
    plt.show()


# In[ ]:




