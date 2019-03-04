from src.data_util import load_ORLdataset
from src.pca import pca_face_recognitioin
from src.lbp import lbp_face_recognition
from src.lbp_pca import lbp_pca_recognition


if __name__ == '__main__':
    orl_path = "face_att"
    # orl_path = "/Users/apple/Desktop/FYP/Datasets/att_faces/"
    grayImg = True
    train_X, train_Y, test_X, test_Y = load_ORLdataset(orl_path, grayImg)

    print('训练数据纬度: ', train_X.shape)
    print('训练数据标签纬度: ', train_Y.shape)
    print('测试数据纬度: ', test_X.shape)
    print('测试数据标签纬度: ', test_Y.shape)

    num_components = 50
    lbp_pca_recognition(train_X, train_Y, test_X, test_Y)
    #lbp_face_recognition(train_X, train_Y, test_X, test_Y)
    pca_face_recognitioin(train_X, train_Y, test_X, test_Y, num_components)
















