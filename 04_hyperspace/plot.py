import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict (X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average(
      (predict(X, w) - Y)
      ** 2
    )

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]
        
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1],1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w



if __name__ == '__main__':
    sns.set()
    plt.axis([0,60,0,100])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Reservations", fontsize=30)
    plt.ylabel("Pizzas", fontsize=30)


    x1, x2, x3, y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
    X = np.column_stack((np.ones(x1.size), x1, x2, x3))
    Y = y.reshape(-1,1)
    w = train(X, Y, iterations=20000, lr=0.001)

    print("\nw=%s"%(w.T))

    print("\nA few preditions:")
    for i in range(5):
        print("X[%d] -> %.4f (label: %d)" % (
            i, 
            predict(X[i], w),
            Y[i]
        ))


    XReservations = np.arange(0,50,0.01).reshape(-1, 1)
    YReservations = predict(XReservations, w[0])

    plt.plot(XReservations, YReservations)

    plt.plot(x1,y, "bo")
    plt.savefig("reservations.png")
    plt.close()


    XTemperature = np.arange(0,50,0.01).reshape(-1, 1)
    YTemperature = predict(XTemperature, w[1])
    plt.plot(XTemperature, YTemperature)
    plt.plot(x2,y, "bo")
    plt.savefig("temperature.png")
    plt.close()


    XTourists = np.arange(0,50,0.01).reshape(-1, 1)
    YTourists = predict(XTourists, w[2])
    plt.plot(XTourists, YTourists)
    plt.plot(x3,y, "bo")
    plt.savefig("tourists.png")
    plt.close()
