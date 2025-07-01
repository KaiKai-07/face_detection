import cv2
import numpy as np
from openvino.runtime import Core
import datetime
# 初始化 OpenVINO Core
core = Core()

# 指定模型的絕對路徑
model = core.read_model(
    model="intel/face-detection-0204/FP32/face-detection-0204.xml",
    weights="intel/face-detection-0204/FP32/face-detection-0204.bin"
)
compiled_model = core.compile_model(model, "CPU")

# 取得模型輸入輸出資訊
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
b, c, h, w = input_layer.shape  # 模型輸入大小

landmark_model = core.read_model(
    model="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
    weights="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin"
)
landmark_compiled_model = core.compile_model(landmark_model, "CPU")

land_input = landmark_compiled_model.input(0)
land_output = landmark_compiled_model.output(0)
b_l, c_l, h_l, w_l = land_input.shape

# 開啟攝影機
cap = cv2.VideoCapture(0)

#print(f"模型預期輸入大小: {w}x{h}")

while True:
    #print('frame read start',datetime.datetime.utcnow())
    ret, frame = cap.read()
    #print('frame read end',datetime.datetime.utcnow())
    if not ret:
        break

    # 前處理：resize、轉換成 BCHW
    image_resized = cv2.resize(frame, (w, h))
    image_input = image_resized.transpose((2, 0, 1))  # HWC -> CHW
    image_input = image_input[np.newaxis, :].astype(np.float32)  # 增 batch 維度

    # 執行推論
    #print('result read start',datetime.datetime.utcnow())
    results = compiled_model([image_input])[output_layer]
    #print('result read end',datetime.datetime.utcnow())
    # 繪製預測框
    for detection in results[0][0]:
        conf = float(detection[2])
        if conf > 0.5:
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(
                frame, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            face_crop = frame[y_min:y_max, x_min:x_max]
            face_crop = cv2.resize(face_crop, (w_l, h_l))
            face_input = face_crop.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

            # 推論 landmarks 模型
            landmarks = landmark_compiled_model([face_input])[land_output]
            landmarks = landmarks.reshape(-1, 2)  # (5,2)

            # 把 landmarks 座標換算回原影像座標
            for (lx, ly) in landmarks:
                px = int(x_min + lx * (x_max - x_min))
                py = int(y_min + ly * (y_max - y_min))
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

    # 顯示畫面
    cv2.imshow('Face Detection (OpenVINO)', frame)
    #print('show frame end',datetime.datetime.utcnow())
    # 按 q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
