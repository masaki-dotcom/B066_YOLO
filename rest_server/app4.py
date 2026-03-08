import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import abort, Api, Resource
from ultralytics import YOLO
from flask_cors import CORS

# =========================
# Flask
# =========================
app = Flask(__name__)
CORS(app)
api = Api(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# モデル
# =========================

model_pipe = YOLO(os.path.join(BASE_DIR, "pipe/best.pt"))
model_yari = YOLO(os.path.join(BASE_DIR, "yari/best.pt"))

NAMES = {
    0: "muku",
    1: "pipe"
}

# =========================
# QR
# =========================

qr_detector = cv2.QRCodeDetector()

# =========================
# 行ソート
# =========================

def row_sort(centers, row_threshold=40):

    centers = sorted(centers, key=lambda x: x[1])

    rows = []
    current = []

    last_y = None

    for c in centers:

        x,y,color = c

        if last_y is None:
            current.append(c)
            last_y = y
            continue

        if abs(y-last_y) < row_threshold:
            current.append(c)
        else:
            rows.append(current)
            current = [c]

        last_y = y

    if current:
        rows.append(current)

    ordered = []

    for r in rows:
        r = sorted(r, key=lambda x: x[0])
        ordered.extend(r)

    return ordered


# =========================
# 推論API
# =========================

class smart_mat_url(Resource):

    def post(self, name):
        if name=='predict':
            print(name)
            if "image" not in request.files:
                return jsonify({"error": "no image"}), 400

            file = request.files["image"]
            kind = request.form.get("kind")

            display_classes = request.form.getlist("classes[]")

            roi_x = int(request.form.get("roi_x",0))
            roi_y = int(request.form.get("roi_y",0))
            roi_w = int(request.form.get("roi_w",0))
            roi_h = int(request.form.get("roi_h",0))

            circle_radius = int(request.form.get("circle_radius",6))
            box_thickness = int(request.form.get("box_thickness",2))
            font_scale = float(request.form.get("font_scale",0.6))
            font_thickness = int(request.form.get("font_thickness",2))

            # =========================
            # 画像読み込み
            # =========================

            img = cv2.imdecode(
                np.frombuffer(file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )

            img_draw = img.copy()

            # =========================
            # ROI
            # =========================

            if roi_w > 0 and roi_h > 0:

                roi_img = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            else:

                roi_img = img
                roi_x = 0
                roi_y = 0
                roi_h, roi_w = img.shape[:2]

            # =========================
            # モデル選択
            # =========================

            model = model_yari if kind == "yari" else model_pipe

            results = model(roi_img, conf=0.6)[0]

            counts = {"muku":0,"pipe":0}

            centers = []

            overlay = img_draw.copy()

            # =========================
            # Segmentation
            # =========================

            if results.masks is not None:
                print(9999)

                masks = results.masks.data.cpu().numpy()

                for mask, cls in zip(masks, results.boxes.cls):

                    cls_id = int(cls)
                    color = (0,0,255) if cls_id==0 else (255,0,0)

                    mask = cv2.resize(mask,(roi_w,roi_h))
                    mask_bool = mask > 0.5

                    ys,xs = np.where(mask_bool)

                    if len(xs)>0:

                        cx = int(xs.mean()) + roi_x
                        cy = int(ys.mean()) + roi_y

                        centers.append((cx,cy,color))

                    if "MaskFill" in display_classes:

                        overlay[
                            roi_y:roi_y+roi_h,
                            roi_x:roi_x+roi_w
                        ][mask_bool] = color

            if "MaskFill" in display_classes:

                img_draw = cv2.addWeighted(
                    overlay,
                    0.25,
                    img_draw,
                    0.75,
                    0
                )

            # =========================
            # Box / Label
            # =========================

            number = 1

            for box, cls, conf in zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf
            ):

                bx1,by1,bx2,by2 = map(int,box)

                bx1 += roi_x
                bx2 += roi_x
                by1 += roi_y
                by2 += roi_y

                cls_id = int(cls)
                score = float(conf)

                class_name = NAMES[cls_id]
                counts[class_name]+=1

                color = (0,0,255) if cls_id==0 else (255,0,0)

                cx = (bx1+bx2)//2
                cy = (by1+by2)//2

                if "Box" in display_classes:

                    cv2.rectangle(
                        img_draw,
                        (bx1,by1),
                        (bx2,by2),
                        (0,255,0),
                        box_thickness
                    )

                if "Label" in display_classes:

                    label = f"{class_name} {score*100:.1f}"

                    cv2.putText(
                        img_draw,
                        label,
                        (bx1,by1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0,255,0),
                        font_thickness
                    )

            # =========================
            # 行ソート
            # =========================

            centers = row_sort(centers)

            if "Numbers" in display_classes:

                for i,(cx,cy,color) in enumerate(centers):

                    cv2.putText(
                        img_draw,
                        str(i+1),
                        (cx-10,cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale*1.2,
                        color,
                        font_thickness
                    )

            # =========================
            # Circle
            # =========================

            if "Circle" in display_classes:

                for cx,cy,color in centers:

                    cv2.circle(
                        img_draw,
                        (cx,cy),
                        circle_radius,
                        color,
                        -1
                    )

            # =========================
            # QR
            # =========================

            qr_text = ""

            data, bbox, _ = qr_detector.detectAndDecode(img)

            if data:
                qr_text = data

                if bbox is not None:

                    bbox = bbox.astype(int)

                    for i in range(len(bbox[0])):

                        cv2.line(
                            img_draw,
                            tuple(bbox[0][i]),
                            tuple(bbox[0][(i+1)%4]),
                            (255,0,0),
                            2
                        )

            # =========================
            # 画像返却
            # =========================

            _, buffer = cv2.imencode(".jpg", img_draw)

            img_base64 = base64.b64encode(buffer).decode()

            return jsonify({
                "image": img_base64,
                "counts": counts,
                "qr": qr_text
            })


# =====================
# main
# =====================
api.add_resource(smart_mat_url, '/api/smart_mat_url/<name>')

if __name__ == "__main__":
   app.run(host='localhost', debug=True,port=5001)
    # app.run(debug=True, host='0.0.0.0', port=5001)