import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def predict (X, w):
    return X * w

def loss(X, Y, w):
    return np.average(
      (predict(X, w) - Y)
      ** 2
    )


def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss=loss(X,Y,w)
        print ("Iteration %d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w -lr) < current_loss:
            w -= lr
        else:
            return w

    raise Exception("Could't converge withing %d iterations" % iterations)




sns.set()
plt.axis([0,60,0,100])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=30)
plt.ylabel("Pizzas", fontsize=30)


X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w = train(X, Y, iterations=10000, lr=0.01)

print("\nw=%.3f" % w)

print("Prediction: x=%d => y=%.2f" % (20, predict(20,w)))


XDach = np.arange(0,50,0.01)
YDach = predict(XDach, w)

plt.plot(XDach, YDach)

plt.plot(X,Y, "bo")
print(loss(np.array([3]), np.array([9]), 0.5))
plt.savefig("plot.png")