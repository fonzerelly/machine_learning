import mnist as data
import numpy as np
import os.path as path
import sys
from PIL import Image, ImageEnhance, ImageFilter

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward (X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

def classify(X, w):

    y_hat = forward(X, w)
    labels = np.argmax(y_hat, axis=1)
    result = labels.reshape(-1,1)
    return result

def loss(X, Y, w):
    y_hat = forward(X,w)
    first_term = Y * np.log(y_hat)
    second_term = (1-Y) * np.log(1-y_hat)
    return -np.sum(first_term + second_term) / X.shape[0]
    # return -np.average(
    #   first_term + second_term
    # )

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]
        
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1],10))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w)))
        report(i, X, Y, data.X_test, data.Y_test, w)
        w -= gradient(X, Y, w) * lr
    report(iterations, X, Y, data.X_test, data.Y_test, w)
    return w

def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    n_test_examples = Y_test.shape[0]
    matches_rate = matches * 100.0 / n_test_examples
    training_loss = loss(X_train, Y_train, w)
    print('%d - loss: %.20f, %.2f%%' % (iteration, training_loss, matches_rate))


if __name__ == '__main__':
    if (path.isfile('.training_data.npy')):
        w = np.load('.training_data.npy')
        report(-1, data.X_train, data.Y_train, data.X_test, data.Y_test, w)    
    else:
        w = train(data.X_train, data.Y_train, iterations=200, lr=1e-5)
        np.save('.training_data', w)
    

    img = Image.open(sys.argv[1])
    img.save('./.original.png')
    info = img.info
    print(info)


    kontrast = ImageEnhance.Contrast(img)
    img_enhanced = ImageEnhance.Brightness(kontrast.enhance(5.0)).enhance(3.0).filter(ImageFilter.EDGE_ENHANCE_MORE)
    img_enhanced.save('./.enhanced.png')
    img_resized = img_enhanced.resize((28, 28))
    img_gray = img_resized.convert('L')
    img_gray.save('./.final.png')
    img_array = np.array(img_gray)
    print(img_array.shape)
    images = img_array.reshape(1,-1)
    print(images.shape)
    images_with_bias = data.prepend_bias(images)
    print(images_with_bias.shape)
    inverted_img_array = 255 - images_with_bias
    print(inverted_img_array.shape)
    result = classify(inverted_img_array, w)

    print ("Classified:", result[0][0])