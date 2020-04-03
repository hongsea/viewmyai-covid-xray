import torch
from torchvision.transforms import transforms
import os
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate  


class Xray:
    def __init__(self, gpu=False):
        torch.set_printoptions(precision=4)
        self.CLASS_NAMES = ['COVID-19','NORMAL','Pneumonia']
        tfms = get_transforms()
        arch = models.resnet50
        #xray image detect
        xray_image_path = 'xray-dataset/'
        xray_data = ImageDataBunch.from_folder(xray_image_path, train='train', valid='val', ds_tfms=tfms, size=128, bs=10)
        self.learn_xray = cnn_learner(xray_data, models.resnet50, metrics=[error_rate, accuracy])
        self.learn_xray.load('xray_image')
        # EFFNET
        corona_images_path = 'covid-dataset/'
        covid_data = ImageDataBunch.from_folder(corona_images_path, train='train', valid='val', ds_tfms=tfms, size=128, bs=10)


        self.learn_covid = cnn_learner(covid_data, models.resnet50, metrics=[error_rate, accuracy])
        self.learn_covid.load('Corona_model_final')
       
    def predict_dense(self, image):
        image = Image(pil2tensor(image, dtype=np.float32).div_(255))
        xray_predect =  self.learn_xray.predict(image)
        if str(xray_predect[0]) != "xray":
            return []
        out = self.learn_covid.predict(image)
        return out[2].numpy()

    def predict(self, image):
        image = image.convert("RGB")
        disease_proba = self.predict_dense(image)
        disease_proba = list(zip(self.CLASS_NAMES, disease_proba))
        r = {"condition rate": disease_proba}
        return r

