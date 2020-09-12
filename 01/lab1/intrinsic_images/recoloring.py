import cv2
import matplotlib.pyplot as plt
import numpy as np

ball = plt.imread('ball.png')
albedo = plt.imread('ball_albedo.png')
shading = cv2.cvtColor(plt.imread('ball_shading.png'), cv2.COLOR_GRAY2RGB)

# conversion of shading to RGB mapped the values to [0,1], therefore (0,255,0) = (0,1,0)
albedo[np.where(albedo[:,:,] != (0,0,0))[:-1]] = (0,1.,0)

plt.subplot(1,2,1)
plt.imshow(ball)
plt.subplot(1,2,2)
plt.imshow(albedo * shading)

plt.show()