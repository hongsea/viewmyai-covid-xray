import torch
from torchvision.transforms import transforms
import os
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate  

class Xray:
    def __init__(self, gpu=False):
        self.CLASS_NAMES = ['COVID-19','NORMAL','Pneumonia']
        # EFFNET
        corona_images_path = 'covid-dataset/'
        tfms = get_transforms()
        data = ImageDataBunch.from_folder(corona_images_path, train='train', valid='val', ds_tfms=tfms, size=128, bs=10)

        arch = models.resnet50
        self.learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
        self.learn.load('Corona_model_final')
       

    def predict_dense(self, image):
        image = Image(pil2tensor(image, dtype=np.float32).div_(255))
        out = self.learn.predict(image)
        return out[2]

    def predict(self, image):
        image = image.convert("RGB")
        disease_proba = self.predict_dense(image)
        disease_proba = list(zip(self.CLASS_NAMES, disease_proba))
        disease_proba.sort(key=lambda x: x[1], reverse=True)
        
        r = {"condition rate": disease_proba}
        return r

