from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'F:/yolo/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/my_yolov8_LWN.yaml').load(r'F:\yolo\ultralytics-main\ultralytics-main\weights\detection\yolov8n.pt')  # 直接加载预训练模型
    # results = model.train(data='F:/yolo/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/my_yolov8_LWN.yaml',
    #                       epochs=100, imgsz=320, batch=64)

    model.train(data="F:/yolo/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/my_yolov8_LWN.yaml",
                epochs=220,
                cos_lr=True,  # 余弦退火
                imgsz=416, batch=128)  # 前 10 轮冻结骨干网络