import torch
from torchvision.transforms import transforms
from torch.nn.functional import softmax
import os

from .DenseNet import DenseNet121
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate  

class Xray:
    def __init__(self, gpu=False):
        models_directory = os.path.dirname(os.path.abspath(__file__))
        # DENSENET
        self.N_CLASSES = 14
        self.CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                            'Hernia']
        if gpu:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # initialize and load the model
        model_dense = DenseNet121(self.N_CLASSES).to(device).eval()
        if gpu:
            model_dense = torch.nn.DataParallel(model_dense).to(device).eval()
            checkpoint = torch.load(os.path.join(models_directory, "gpu_weight.pth"))
        else:
            checkpoint = torch.load(os.path.join(models_directory, "cpu_weight.pth"), map_location=device)

        model_dense.load_state_dict(checkpoint)

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        self.transform_dense = transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([self.normalize(crop) for crop in crops]))
        ])

        self.model_dense = model_dense.to(device).eval()
        self.device = device

        # EFFNET
        corona_images_path = 'covid-dataset/'
        tfms = get_transforms()
        data = ImageDataBunch.from_folder(corona_images_path, train='train', valid='val', ds_tfms=tfms, size=128, bs=10)

        arch = models.resnet50
        self.learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])
        self.learn.load('Corona_model_final')
       
    def predict_dense(self, image):
        x = self.transform_dense(image).to(self.device)
        out = self.model_dense(x).cpu().detach()
        probas = softmax(out*5).mean(0)
        return list(probas.numpy()*100)

    def predcit_eff(self, image):
        image = Image(pil2tensor(image, dtype=np.float32).div_(255))
        out = self.learn.predict(image)
        proba = out[2][0].numpy()
        if str(out[0]) == "Pneumonia":
           proba = out[2][2].numpy()
           return proba*100,out[0] 
        if str(out[0]) == "NORMAL":
           proba = out[2][1].numpy()
           return proba*100,out[0] 
        if 0.90 < proba < 0.95:
            return proba*100,"Pneumonia"
        if proba < 0.90:
            return proba*100,"NORMAL" 
        return proba*100,out[0] 

    def predict(self, image):
        image = image.convert("RGB")
        healthy_proba,_type = self.predcit_eff(image)
        disease_proba = self.predict_dense(image)
        disease_proba = list(zip(self.CLASS_NAMES, disease_proba))
        disease_proba.sort(key=lambda x: x[1], reverse=True)
        if str(_type) == "Pneumonia" or str(_type) == "COVID-19":
            result = "Not Healthy"
        else:
             result = "Healthy"
        r = {"result":result,
             "type":str(_type),
             "probability": healthy_proba,
             "condition similarity rate": disease_proba}
        return r

