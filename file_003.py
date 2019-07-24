import matplotlib.pyplot as plt
import numpy as np
X = [6,8,10,14,18]
y = [7,9,13,17.5,18]
plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression

X=np.array(X)
y=np.array(y)

X=X.reshape(-1,1)
y=y.reshape(-1,1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)
test_vect=[100]
test_vect=np.array(test_vect)
test_vect=test_vect.reshape(-1,1)
print('A pizza should cost: $%.2f' % model.predict(test_vect))

print('Residual sum of squares: %.2f' % np.mean((model.predict(X)
- y) ** 2))
