import cv2
import numpy as np
from openvino.runtime import Core
import os

# 初始化 OpenVINO Core
core = Core()

# 指定模型的絕對路徑
face_model = core.read_model(
    model="intel/face-detection-0204/FP32/face-detection-0204.xml",
    weights="intel/face-detection-0204/FP32/face-detection-0204.bin"
)
face_compiled_model = core.compile_model(face_model, "CPU")

# 取得模型輸入輸出資訊
input_layer = face_compiled_model.input(0)
output_layer = face_compiled_model.output(0)
b, c, h, w = input_layer.shape  # 模型輸入大小

landmark_model = core.read_model(
    model="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml",
    weights="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin"
)
landmark_compiled_model = core.compile_model(landmark_model, "CPU")

land_input = landmark_compiled_model.input(0)
land_output = landmark_compiled_model.output(0)
b_l, c_l, h_l, w_l = land_input.shape

rec_model = core.read_model(
    model="intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml",
    weights="intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin"
)
rec_compiled_model = core.compile_model(rec_model, "CPU")

rec_input = rec_compiled_model.input(0)
rec_output = rec_compiled_model.output(0)
b_r, c_r, h_r, w_r = rec_input.shape

# 載入參考圖像資料夾中的所有圖像
reference_folder = "gallery"  
reference_embeddings = []
reference_names = []

if not os.path.exists(reference_folder):
    raise FileNotFoundError(f"資料夾 {reference_folder} 不存在")

for filename in os.listdir(reference_folder):
    if filename.lower().endswith('.jpg'):
        name = os.path.splitext(filename)[0]  # 從檔名提取名字
        ref_image = cv2.imread(os.path.join(reference_folder, filename))
        if ref_image is None:
            print(f"無法載入 {filename}，跳過")
            continue

        # 前處理參考圖像
        ref_image_resized = cv2.resize(ref_image, (w, h))
        ref_image_input = ref_image_resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

        # 檢測參考圖像中的人臉
        ref_results = face_compiled_model([ref_image_input])[output_layer]
        ref_face = None
        for detection in ref_results[0][0]:
            conf = float(detection[2])
            if conf > 0.5:
                x_min = int(detection[3] * ref_image.shape[1])
                y_min = int(detection[4] * ref_image.shape[0])
                x_max = int(detection[5] * ref_image.shape[1])
                y_max = int(detection[6] * ref_image.shape[0])
                ref_face = ref_image[y_min:y_max, x_min:x_max]
                break

        if ref_face is None:
            print(f"{filename} 中未檢測到人臉，跳過")
            continue

        # 提取參考人臉的 landmarks 並校正
        ref_face_crop = cv2.resize(ref_face, (w_l, h_l))
        ref_face_input = ref_face_crop.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
        ref_landmarks = landmark_compiled_model([ref_face_input])[land_output].reshape(-1, 2)
        ref_landmark = ref_landmarks * np.float32([w_l, h_l])

        landmark_reference = np.float32([
            [0.31556875000000000, 0.4615741071428571],
            [0.68262291666666670, 0.4615741071428571],
            [0.50026249999999990, 0.6405053571428571],
            [0.34947187500000004, 0.8246919642857142],
            [0.65343645833333330, 0.8246919642857142]
        ])
        landmark_ref = landmark_reference * np.float32([w_l, h_l])
        M = cv2.getAffineTransform(ref_landmark[0:3], landmark_ref[0:3])
        ref_align = cv2.warpAffine(ref_face_crop, M, (w_l, h_l))

        # 提取參考人臉特徵
        ref_face_rec = cv2.resize(ref_align, (w_r, h_r))
        ref_face_rec_input = ref_face_rec.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
        ref_embedding = rec_compiled_model([ref_face_rec_input])[rec_output].flatten()

        reference_embeddings.append(ref_embedding)
        reference_names.append(name)

if not reference_embeddings:
    raise ValueError("參考資料夾中未找到有效的人臉圖像")

# 計算餘弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 前處理：resize、轉換成 BCHW
    image_resized = cv2.resize(frame, (w, h))
    image_input = image_resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

    # 執行推論
    results = face_compiled_model([image_input])[output_layer]

    # 繪製預測框
    for detection in results[0][0]:
        conf = float(detection[2])
        if conf > 0.3:
            x_min = int(detection[3] * frame.shape[1])
            y_min = int(detection[4] * frame.shape[0])
            x_max = int(detection[5] * frame.shape[1])
            y_max = int(detection[6] * frame.shape[0])
            
            # 繪製人臉框
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 裁剪人臉並進行 landmarks 檢測
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
            face_crop = cv2.resize(face_crop, (w_l, h_l))
            face_input = face_crop.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

            # 推論 landmarks 模型
            landmarks = landmark_compiled_model([face_input])[land_output]
            landmarks = landmarks.reshape(-1, 2)

            # 把 landmarks 座標換算回原影像座標
            for (lx, ly) in landmarks:
                px = int(x_min + lx * (x_max - x_min))
                py = int(y_min + ly * (y_max - y_min))
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

            # 校正角度
            landmark = landmarks * np.float32([w_l, h_l])
            M = cv2.getAffineTransform(landmark[0:3], landmark_ref[0:3])
            align = cv2.warpAffine(face_crop, M, (w_l, h_l))

            # 人臉辨識
            face_rec = cv2.resize(align, (w_r, h_r))
            face_rec_input_data = face_rec.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
            embedding = rec_compiled_model([face_rec_input_data])[rec_output].flatten()

            # 與所有參考特徵比較
            max_similarity = -1
            best_match_name = "Unknown"
            for ref_embedding, name in zip(reference_embeddings, reference_names):
                similarity = cosine_similarity(embedding, ref_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_name = name if similarity > 0.5 else "Unknown"

            # 顯示名字和置信度
            cv2.putText(
                frame, f"{best_match_name} ({max_similarity:.2f})", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    # 顯示畫面
    cv2.imshow('Face Detection (OpenVINO)', frame)

    # 按 q 結束
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()