import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from skimage import io, util, filters
from skimage.morphology import disk

train_path = os.path.join('other_model', 'RFI_TOTAL_train.csv')
test_path = os.path.join('other_model', 'RFI_TOTAL_test.csv')

data = pd.read_csv(train_path)
N = data.shape[0]

for i in range(N):
    PID = data.loc[i,'PID']
    STAGE = data.loc[i,'STAGE']
    SLICE = data.loc[i,'SLICE']
    FILE = PID + '_' + str(SLICE) + '_image.jpg'
    PATH = os.path.join('..', 'Image', str(STAGE), FILE)

    img = io.imread(PATH)
    image = Image.fromarray(img).convert('L')

    img_noise = util.random_noise(img, mode='pepper',amount=0.01)

    io.imsave(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'image'), img, quality=95)
    io.imsave(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_noise_image.jpg'), img_noise, quality=95)

    image.filter(ImageFilter.GaussianBlur(radius=2)).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_gau_image.jpg'), 'JPEG', quality=95)
    image.filter(ImageFilter.MedianFilter(size=3)).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_med_image.jpg'), 'JPEG', quality=95)

    ImageEnhance.Contrast(image).enhance(2).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_con2_image.jpg'), 'JPEG', quality=95)
    ImageEnhance.Contrast(image).enhance(0.5).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_con1_image.jpg'), 'JPEG', quality=95)

    ImageEnhance.Brightness(image).enhance(2).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_bri2_image.jpg'), 'JPEG', quality=95)
    ImageEnhance.Brightness(image).enhance(0.5).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_bri1_image.jpg'), 'JPEG', quality=95)

    ImageEnhance.Sharpness(image).enhance(10).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_shp2_image.jpg'), 'JPEG', quality=95)
    ImageEnhance.Sharpness(image).enhance(5).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_shp1_image.jpg'), 'JPEG', quality=95)

    image.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_flipLR_image.jpg'), 'JPEG', quality=95)
    image.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_flipTB_image.jpg'), 'JPEG', quality=95)
    
    image.rotate(45).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_rot45_image.jpg'), 'JPEG', quality=95)
    image.transpose(Image.ROTATE_90).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'rot90_image.jpg'), 'JPEG', quality=95)
    image.transpose(Image.TRANSPOSE).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_tp_image.jpg'), 'JPEG', quality=95)
    image.rotate(135).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_rot135_image.jpg'), 'JPEG', quality=95)
    image.transpose(Image.ROTATE_180).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_rot180_image.jpg'), 'JPEG', quality=95)
    image.transpose(Image.ROTATE_270).save(os.path.join('..', 'Image_Aug', str(STAGE), PID+'_'+SLICE+'_rot270_image.jpg'), 'JPEG', quality=95)