
import numpy as np
position = np.array([[-5.12, 5.12]])
if position.ndim == 1:
    position = position.reshape(1, -1)
x = position[:, 0]
y = position[:, 1]
numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
denominator = 0.5 * (x**2 + y**2) + 2