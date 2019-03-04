# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src.lbp import lbp_face_recognition
from sklearn.decomposition import PCA
from src.pca import eigen




def lbp_pca_recognition(train_data, train_label, test_data, test_label):
    """
   LBP算法识别
    """
    pca = PCA(n_components=1)
    pca.fit(train_label)
    pca.fit(test_label)
    lbp_face_recognition(train_data, train_label, test_data, test_label)

