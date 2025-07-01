import cv2
import numpy as np
from openvino.runtime import Core
import datetime

core = Core()
model = core.read_model(
    model="intel/face-detection-0206/FP32/face-detection-0206.xml",
    weights="intel/face-detection-0206/FP32/face-detection-0206.bin"
)
compiled_model = core.compile_model(model, "CPU")

# 輸入輸出
input_layer = compiled_model.input(0)
boxes_output = compiled_model.output('boxes')
labels_output = compiled_model.output('labels')
b, c, h, w = input_layer.shape

# 開啟攝影機
cap = cv2.VideoCapture(0)
thisTime = datetime.time(12,0,0,1)
while True:
    print('frame read start',datetime.datetime.utcnow())
    ret, frame = cap.read()
    print('frame read end',datetime.datetime.utcnow())
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # 前處理
    image_resized = cv2.resize(frame, (w, h))
    image_input = image_resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

    # 推論
    print('result read start',datetime.datetime.utcnow())
    results = compiled_model([image_input])
    print('result read end',datetime.datetime.utcnow())
    boxes = results[boxes_output]
    labels = results[labels_output]

    scale_x = orig_w / w
    scale_y = orig_h / h

    # 畫框
    for i in range(boxes.shape[0]):
        x_min, y_min, x_max, y_max, conf = boxes[i]
        label = int(labels[i])
        if conf > 0.5 and label == 0:
            x_min = int(x_min * scale_x)
            y_min = int(y_min * scale_y)
            x_max = int(x_max * scale_x)
            y_max = int(y_max * scale_y)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{conf:.2f}", (x_min, y_min-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
    #print('show frame start')
    cv2.imshow('Face Detection (OpenVINO)', frame)
    print('show frame end',datetime.datetime.utcnow())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 