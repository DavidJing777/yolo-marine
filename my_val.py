from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./runs/detect/train7/weights/best.pt')
    # 2. 在验证集上进行评估
    results = model.val()


