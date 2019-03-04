# -*- coding: utf-8 -*-
from src import lbp_pca,lbp,pca
from sklearn import model_selection, svm
#对识别出来的结果进行分类（可以是三种识别方法的任意一种结果）
def svm_classify():
	emotion={0:'happy',1:'anger',2:'sad',4:'surprise',5:'neutal',6:'serious'}

	return 