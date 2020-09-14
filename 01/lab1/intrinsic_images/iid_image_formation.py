import cv2
import matplotlib.pyplot as plt


ball = plt.imread('ball.png')
albedo = plt.imread('ball_albedo.png')
shading = cv2.cvtColor(plt.imread('ball_shading.png'), cv2.COLOR_GRAY2RGB)
result = albedo * shading

plt.subplot(2,2,1)
plt.imshow(albedo)
plt.subplot(2,2,2)
plt.imshow(shading)

plt.subplot(2,2,3)
plt.imshow(ball)
plt.subplot(2,2,4)
plt.imshow(result)

plt.show()