import numpy as np
import matplotlib.pyplot as plt


def eigen(data):
    """
    :param train_data: M * N^2 matrix, each row represents each image N*N
    :return:
    eigenValues: M, each eigenvalue corresponding one eigenvector, in descending order
    eigenVectors: M * N^2 matrix, each row represents each eigenface, in descending order
    """
    train_data_transpose = np.transpose(data)
    cov_matrix = np.dot(data, train_data_transpose)
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    eigenVectors = np.dot(eigenVectors, data)
    # Sort the eigenVectors in descending order corresponding to the eigenValues
    sort = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[sort]
    eigenVectors = eigenVectors[sort, :]

    return eigenValues, eigenVectors

def train_PCA(data, num_components):
    """
    :param train_data:  M * N^2, each row corresponding to each image, which is reshaped into 1-D vector
    :param num_components: The number of the largest eigenVector to be kept
    :return:
    """
    mean_image = np.mean(data, axis=0)
    data = data - mean_image
    eigenValues, eigenVectors = eigen(data)
    eigenVectors = eigenVectors[:num_components]

    weiVec_train = np.dot(data, eigenVectors.T)

    return mean_image, eigenVectors, weiVec_train


def pca_face_recognitioin(train_data, train_label, test_data, test_label, num_components):
    """
    PCA算法识别
    """

    print('*' * 40)
    print('PCA算法识别 ...')

    if len(train_data.shape[1:]) == 3:
        isRGB = True
        HEIGHT, WIDTH, CHANNEL = train_data.shape[1], train_data.shape[2], train_data.shape[3]
    else:
        isRGB = False
        HEIGHT, WIDTH = train_data.shape[1], train_data.shape[2]

    # Reshape the image data into rows
    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    print('训练数据降维后: ', train_data.shape)
    print('测试数据降维后: ', test_data.shape)

    mean_image, eigenVectors, weiVec_train = train_PCA(train_data, num_components)

    # Plot the average face
    if isRGB:
        plt.imshow(mean_image.reshape(HEIGHT, WIDTH, CHANNEL).astype('uint8'))
    else:
        plt.imshow(mean_image.reshape(HEIGHT, WIDTH).astype('uint8'))
    plt.title('The average face')
    plt.show()
    # Plot 10 most significant eigenFaces
    eigenFaces = []
    if isRGB:
        for i in range(eigenVectors.shape[0]):
            eigenFaces.append(eigenVectors[i].reshape(HEIGHT, WIDTH, CHANNEL))
    else:
        for i in range(eigenVectors.shape[0]):
            eigenFaces.append(eigenVectors[i].reshape(HEIGHT, WIDTH))
    eigenFaces = np.asarray(eigenFaces)
    if num_components > 10:
        for plt_idx in range(1, 11):
            plt.subplot(5, 2, plt_idx)
            plt.imshow(eigenFaces[plt_idx - 1].astype('uint8'))
            plt.axis('off')
        plt.show()

    # Verify the test faces
    test_data = test_data - mean_image
    weiVec_test = np.dot(test_data, eigenVectors.T)
    correct_count = 0
    # Caculate the L2 distance for the testing weight and training weight
    for i in range(weiVec_test.shape[0]):
        dist = np.linalg.norm((weiVec_train - weiVec_test[i]), axis=1)
        index_min = np.argmin(dist)
        if train_label[index_min] == test_label[i]:
            correct_count += 1

    pca_accuracy = correct_count / len(test_label)
    print('PCA算法识别率：', pca_accuracy)
    print('*' * 40)