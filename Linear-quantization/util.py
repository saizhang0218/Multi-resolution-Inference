import warnings
import pickle
import io
import time
import os
import shutil
import math
import torchvision
import msgpack
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


def loss_teacher_student(student_outputs, labels, teacher_outputs):
    alpha = 0.5
    T = 4.0
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(student_outputs, labels) * ((1. - alpha)/2) + F.cross_entropy(teacher_outputs, labels) * ((1. - alpha)/2)

    return KD_loss

