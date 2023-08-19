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


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss=loss(X,Y,w,b)
        print ("Iteration %d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w -lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b -lr) < current_loss:
            b -= lr
        elif loss(X, Y, w, b +lr) < current_loss:
            b += lr
        else:
            return w,b

    raise Exception("Could't converge withing %d iterations" % iterations)




sns.set()
plt.axis([0,60,0,100])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=100000, lr=0.001)

print("\nw=%.3f b=%.3f" % (w, b))

print("Prediction: x=%d => y=%.2f" % (20, predict(20,w,b)))


XDach = np.arange(0,50,0.01)
YDach = predict(XDach, w,b)

plt.plot(XDach, YDach)

plt.plot(X,Y, "bo")
# print(loss(np.array([3]), np.array([9]), 0.5, 0.5))
plt.savefig("plot.png")