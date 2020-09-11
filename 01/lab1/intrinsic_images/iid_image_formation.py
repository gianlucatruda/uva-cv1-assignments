import cv2
import matplotlib.pyplot as plt


ball = cv2.cvtColor(cv2.imread('ball.png'), cv2.COLOR_BGR2RGB)
albedo = cv2.cvtColor(cv2.imread('ball_albedo.png'), cv2.COLOR_BGR2RGB)
shading = cv2.cvtColor(cv2.imread('ball_shading.png'), cv2.COLOR_BGR2RGB)
result = albedo * shading

#result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#cv2.imshow("Product", result_rgb)
#cv2.waitKey(0)
#cv2.destroyWindow("Product")

plt.subplot(2,2,1)
plt.imshow(albedo)
plt.subplot(2,2,2)
plt.imshow(shading)

plt.subplot(2,2,3)
plt.imshow(ball)
plt.subplot(2,2,4)
plt.imshow(result)

plt.show()