img_d = f"/home/{USER}/Downloads/Segment-and-Track-Anything/point_extract/input_image.jpg"

import cv2
import numpy as np
# 读取图像并转换为灰度图像
image = cv2.imread(img_d)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 对灰度图像应用高斯滤波器
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 应用Canny边缘检测
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
# 获取边缘坐标
edge_coordinates = np.column_stack(np.where(edges > 0))
# 寻找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算轮廓的凸包点
convex_hulls = [cv2.convexHull(contour) for contour in contours]

# 提取凸包点的坐标
representative_points = [list(point[0]) for hull in convex_hulls for point in hull]

# 创建一个新的空白灰度图像
blank_image = 255 * np.ones_like(gray_image)

# 在空白灰度图像上绘制凸包点
for point in representative_points:
    cv2.circle(blank_image, point, 1, (0, 0, 0), -1)

# 显示原始图像、边缘检测后的图像和凸包点灰度图像
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Convex Hull Points (Grayscale)", blank_image)

print("Representative coordinates:", representative_points)
print("Edge coordinates (NumPy array):", edge_coordinates)
cv2.waitKey(0)
cv2.destroyAllWindows()