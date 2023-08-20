import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict (X, w, b):
    return X * w + b

def loss(X, Y, w, b):
    return np.average(
      (predict(X, w, b) - Y)
      ** 2
    )

def gradient(X, Y, w, b):
    wGradient = 2 * np.average (X * (predict(X, w, 0) - Y))
    bGradient = 2 * np.average (predict(X,w,b) - Y)
    return (wGradient,bGradient)

# 
# n= 2

# 1/2*(Y1 - Y)² = 1/2*(Y1² - 2Y1Y + Y²)
# = 1/2* ((X*w + b)² - (X*w + b) * Y + Y)
# = 1/2 ((Xwb)² + 2Xwb + b² - XYw + Yb + Y
#   1/2 (X²w²b² + 2Xwb + b² - XYw + Yb + Y) 
#        X²w²b² + 2Xwb - XYw + b² + Yb + Y

#         A= X²b²
#         B= 2Xb-XY
#         C= b² + Yb + Y

#         Aw² + Bw + C

#         X²b²w²+(2Xb-XY)w+b²+Yb+Y
        
def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, 0)))
        wGradient, bGradient = gradient(X,Y,w,b)
        w -= wGradient * lr
        b -= bGradient * lr
    return w,b



if __name__ == '__main__':
    sns.set()
    plt.axis([0,60,0,100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Reservations", fontsize=30)
    plt.ylabel("Pizzas", fontsize=30)


    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
    w, b = train(X, Y, iterations=20000, lr=0.001)

    print("\nw=%.3f b=%.3f" % (w, b))

    print("Prediction: x=%d => y=%.2f" % (20, predict(20,w,b)))


    XDach = np.arange(0,50,0.01)
    YDach = predict(XDach, w,b)

    plt.plot(XDach, YDach)

    plt.plot(X,Y, "bo")
    # print(loss(np.array([3]), np.array([9]), 0.5, 0.5))
    plt.savefig("plot.png")