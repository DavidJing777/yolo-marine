from ultralytics import YOLO

# 加载你训练好的模型（比如 best.pt）
model = YOLO(r"F:\yolo\ultralytics-main\ultralytics-main\runs\detect\train_material_ours_rectify\weights\best.pt")  # 替换成你的模型路径

# 预测单张图像
results = model.predict(source="F:/yolo/ultralytics-main/ultralytics-main/material_version/val/images/vid_000052_frame0000023.jpg", save=True, conf=0.25)