img_d = f"/home/{USER}/Downloads/Segment-and-Track-Anything/point_extract/dns-pong-clip.mp4"
import cv2
import numpy as np

# 读取输入视频
cap = cv2.VideoCapture(img_d)

# 检查视频是否正常打开
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

# 读取前两帧
_, prev_frame = cap.read()
_, next_frame = cap.read()

# 转换为灰度图像
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

# 提取特征点 (goodFeaturesToTrack() 用于提取帧中的特征点)
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# 光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while cap.isOpened():
    # 计算光流
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # 沿运动方向提取边缘点
    edge_points = prev_points + (next_points - prev_points)

    # 在图像上画出边缘点
    for i, (prev, edge) in enumerate(zip(prev_points, edge_points)):
        a_x, a_y = prev.ravel()
        b_x, b_y = edge.ravel()
    
        # 转换坐标值为整数
        a_x, a_y = int(a_x), int(a_y)
        b_x, b_y = int(b_x), int(b_y)
    
        # 绘制线段表示运动方向
        next_frame = cv2.line(next_frame, (a_x, a_y), (b_x, b_y), (0, 0, 255), 2)

    # 显示处理后的帧
    cv2.imshow("Edges", next_frame)

    # 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 更新帧和特征点
    prev_gray = next_gray.copy()
    _, next_frame = cap.read()
    if next_frame is None:
        break
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(next_points, dtype=np.float32)

# 关闭并释放资源
cap.release()
cv2.destroyAllWindows()
