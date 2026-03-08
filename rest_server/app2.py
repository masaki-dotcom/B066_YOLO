from flask_restful import abort, Api, Resource #pip install flask-restful をインストール
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import base64

# =====================
# Flask
# =====================
app = Flask(__name__)
CORS(app)
api = Api(app)

# =====================
# モデル設定
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
INPUT_SIZE = 1280
NAMES = ["pipe", "muku"]

# =====================
# ONNX Runtime
# =====================
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = sess.get_inputs()[0].name
print("ONNX input:", input_name)

# =====================
# WeChat QR（1回だけロード）
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

wechat_detector = cv2.wechat_qrcode.WeChatQRCode(
    os.path.join(MODEL_DIR, "detect.prototxt"),
    os.path.join(MODEL_DIR, "detect.caffemodel"),
    os.path.join(MODEL_DIR, "sr.prototxt"),
    os.path.join(MODEL_DIR, "sr.caffemodel"),
)
print("WeChat QR Loaded")

# =====================
# Ultralytics互換 letterbox
# =====================
def letterbox(
    img,
    new_shape=(1280, 1280),
    color=(114, 114, 114),
    scaleup=True,
    stride=32
):
    shape = img.shape[:2]  # (h, w)

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img, r, (dw, dh)

# =====================
# 前処理（Ultralytics互換）
# =====================
def preprocess(img):
    img_lb, ratio, (dw, dh) = letterbox(img, INPUT_SIZE)

    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    img_chw = np.transpose(img_rgb, (2, 0, 1))
    blob = np.expand_dims(img_chw, axis=0)

    return blob, ratio, dw, dh

# =====================
# 推論API
# =====================
class smart_mat_url(Resource,):
    def post(self,name):
        if name=='predict':
            if "image" not in request.files:
                return jsonify({"error": "no image"}), 400

            # Nuxtからの表示指定
            display_classes = request.form.getlist("classes[]")

            # --------------------
            # 画像読み込み
            # --------------------
            file = request.files["image"]
            img_np = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "invalid image"}), 400

            orig_h, orig_w = img.shape[:2]

            # --------------------
            # ROI取得
            # --------------------
            try:
                x1 = int(request.form.get("x1"))
                y1 = int(request.form.get("y1"))
                x2 = int(request.form.get("x2"))
                y2 = int(request.form.get("y2"))
                kind = int(request.form.get("kind"))
            except:
                return jsonify({"error": "invalid roi"}), 400

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            x1 = max(0, min(x1, orig_w - 1))
            x2 = max(0, min(x2, orig_w))
            y1 = max(0, min(y1, orig_h - 1))
            y2 = max(0, min(y2, orig_h))

            if x2 <= x1 or y2 <= y1:
                return jsonify({"error": "empty roi"}), 400

            roi_img = img[y1:y2, x1:x2].copy()
            roi_h, roi_w = roi_img.shape[:2]

            # --------------------
            # 前処理
            # --------------------
            blob, ratio, dw, dh = preprocess(roi_img)

            # --------------------
            # 推論
            # --------------------
            outputs = sess.run(None, {input_name: blob})
            preds = outputs[0][0]  # (C, N)

            boxes = []
            scores = []
            class_ids = []

            for i in range(preds.shape[1]):
                xc, yc, bw, bh = preds[0:4, i]
                class_scores = preds[4:, i]

                cls = int(np.argmax(class_scores))
                score = float(class_scores[cls])

                if score < 0.6:
                    continue

                # ★ Ultralytics互換の逆変換
                x = (xc - bw / 2 - dw) / ratio
                y = (yc - bh / 2 - dh) / ratio
                w_box = bw / ratio
                h_box = bh / ratio

                x = int(max(0, min(x, roi_w - 1)))
                y = int(max(0, min(y, roi_h - 1)))
                w_box = int(min(roi_w - x, w_box))
                h_box = int(min(roi_h - y, h_box))

                boxes.append([x, y, w_box, h_box])
                scores.append(score)
                class_ids.append(cls)

            # --------------------
            # NMS
            # --------------------
            indices = cv2.dnn.NMSBoxes(
                boxes, scores,
                score_threshold=0.3,
                nms_threshold=0.3
            )

            # --------------------
            # 描画
            # --------------------
            img_draw = roi_img.copy()
            counts = defaultdict(int)

            #文字、丸のサイズ設定--------------
            h, w = img_draw.shape[:2]
            scale = max(w, h) / 640.0

            font_scale = 0.6 * scale
            font_thickness = max(1, int(2 * scale*0.8))
            box_thickness = max(1, int(2 * scale))
            circle_radius = max(3, int(5 * scale))
            #--------------------------------

            
            # --------------------
            # 検出結果をリスト化
            # --------------------
            detections = []

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w_box, h_box = boxes[i]
                    cls = class_ids[i]
                    score = scores[i]
                    detections.append((x, y, w_box, h_box, cls, score))

            # --------------------
            # 行ごとにグループ化
            # --------------------
            row_threshold = 30 * scale  # 行とみなす高さ許容値（調整可能）

            # y座標でソート
            detections.sort(key=lambda d: d[1])

            rows = []
            current_row = []

            for det in detections:
                x, y, w_box, h_box, cls, score = det

                if not current_row:
                    current_row.append(det)
                    continue

                ref_y = current_row[0][1]

                if abs(y - ref_y) < row_threshold:
                    current_row.append(det)
                else:
                    rows.append(current_row)
                    current_row = [det]

            if current_row:
                rows.append(current_row)

            # --------------------
            # 各行を左から並べる
            # --------------------
            sorted_detections = []
            # for row in reversed(rows):   # ← ここがポイント
            for row in rows:
                row.sort(key=lambda d: d[0])  # x座標でソート
                sorted_detections.extend(row)

            # --------------------
            # 描画（行順 → 左順）
            # --------------------
            number = 0

            # for det in reversed(sorted_detections):
            for det in sorted_detections:
                x, y, w_box, h_box, cls, score = det
                number += 1

                class_name = NAMES[cls]
                counts[class_name] += 1

                color = (0, 0, 255) if cls == 0 else (255, 0, 0)

                # --------------------
                # Box
                # --------------------
                if "Box" in display_classes:
                    cv2.rectangle(
                        img_draw,
                        (x, y),
                        (x + w_box, y + h_box),
                        (0, 255, 0),
                        box_thickness
                    )

                # --------------------
                # Label
                # --------------------
                if "Label" in display_classes:
                    label = f"{class_name} {score*100:.1f}"
                    cv2.putText(
                        img_draw,
                        label,
                        (x, max(int(20 * scale), y - int(5 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )
                

                # --------------------
                # Number
                # --------------------
                if "Numbers" in display_classes:
                    cx = x + w_box // 2
                    cy = y + h_box // 2

                    cv2.putText(
                        img_draw,
                        str(number),
                        (cx - int(10 * scale), cy + int(10 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 1.2,
                        color,
                        font_thickness + 1,
                        cv2.LINE_AA
                    )

                # --------------------
                # Center circle
                # --------------------
                if "Circle" in display_classes:
                    cx = x + w_box // 2
                    cy = y + h_box // 2
                    cv2.circle(
                        img_draw,
                        (cx, cy),
                        circle_radius,
                        color,
                        -1
                    )


            # =====================
            # WeChat QR検出
            # =====================
            qr_texts = []

            decoded_info, points = wechat_detector.detectAndDecode(img_draw)

            if points is not None and len(decoded_info) > 0:
                for text, pts in zip(decoded_info, points):
                    if text != "":
                        qr_texts.append(text)
                        pts = pts.astype(int)

                        for j in range(4):
                            pt1 = tuple(pts[j])
                            pt2 = tuple(pts[(j+1) % 4])
                            cv2.line(img_draw, pt1, pt2, (0,255,0), font_thickness)

                        cv2.putText(
                            img_draw,
                            text,
                            (pts[0][0], pts[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0,255,0),
                            font_thickness
                        )

            # --------------------
            # 返却
            # --------------------
            _, buf = cv2.imencode(".jpg", img_draw)
            img_base64 = base64.b64encode(buf).decode("utf-8")

            return jsonify({
                "counts": counts,
                "image": img_base64,
                "qr": qr_texts
            })

# =====================
# main
# =====================
api.add_resource(smart_mat_url, '/api/smart_mat_url/<name>')


if __name__ == "__main__":
   app.run(host='localhost', debug=True,port=5001)
    # app.run(debug=True, host='0.0.0.0', port=5001)
