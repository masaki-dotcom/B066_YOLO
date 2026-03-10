import os
import cv2
import base64
import numpy as np

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from ultralytics import YOLO

# ==========================
# Flask
# ==========================

app = Flask(__name__)
CORS(app)
api = Api(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# YOLO Load (1回だけ)
# ==========================

print("Loading YOLO models...")

model_pipe = YOLO(os.path.join(BASE_DIR, "pipe/best.pt"))
model_yari = YOLO(os.path.join(BASE_DIR, "yari/best.pt"))
# model_pipe.to("cuda")
# model_yari.to("cuda")

print("YOLO models loaded")

# warmup
dummy = np.zeros((640,640,3),dtype=np.uint8)
model_pipe(dummy)
model_yari(dummy)

print("YOLO warmup finished")

NAMES = {0:"pipe",1:"muku"}

# ==========================
# QR detector
# ==========================

qr_detector = cv2.QRCodeDetector()

MODEL_DIR = os.path.join(BASE_DIR,"model")

wechat_detector = cv2.wechat_qrcode.WeChatQRCode(
    os.path.join(MODEL_DIR,"detect.prototxt"),
    os.path.join(MODEL_DIR,"detect.caffemodel"),
    os.path.join(MODEL_DIR,"sr.prototxt"),
    os.path.join(MODEL_DIR,"sr.caffemodel")
)

print("WeChat QR loaded")

# ==========================
# row sort(行毎に数字を表示する)
# ==========================

def row_sort(centers, th_ratio=0.03):

    if not centers:
        return centers

    ys = [c[1] for c in centers]
    h = max(ys) - min(ys)

    th = max(10, int(h * th_ratio))

    centers = sorted(centers, key=lambda x: x[1])

    rows=[]
    cur=[centers[0]]

    for c in centers[1:]:

        if abs(c[1]-cur[-1][1]) < th:
            cur.append(c)
        else:
            rows.append(cur)
            cur=[c]

    rows.append(cur)

    out=[]

    for r in rows:
        r=sorted(r,key=lambda x:x[0])
        out.extend(r)

    return out


# ==========================
# API
# ==========================

class SmartMatAPI(Resource):

    def post(self,name):

        if name!="predict":
            return {"error":"invalid api"},400

        # --------------------------
        # image load
        # --------------------------

        if "image" not in request.files:
            return {"error":"no image"},400

        file = request.files["image"]

        img = cv2.imdecode(
            np.frombuffer(file.read(),np.uint8),
            cv2.IMREAD_COLOR
        )

        orig_h,orig_w = img.shape[:2]

        # --------------------------
        # ROI
        # --------------------------

        x1 = int(request.form.get("x1",0))
        y1 = int(request.form.get("y1",0))
        x2 = int(request.form.get("x2",0))
        y2 = int(request.form.get("y2",0))

        kind = request.form.get("kind","pipe/muku")

        x1,x2 = sorted([x1,x2])
        y1,y2 = sorted([y1,y2])

        x1=max(0,min(x1,orig_w-1))
        x2=max(0,min(x2,orig_w))
        y1=max(0,min(y1,orig_h-1))
        y2=max(0,min(y2,orig_h))

        if x2<=x1 or y2<=y1:
            roi = img
        else:
            roi = img[y1:y2,x1:x2]

        display = request.form.getlist("classes[]")

        # --------------------------
        # draw parameter
        # --------------------------

        roi_h,roi_w = roi.shape[:2]
        scale = max(roi_h,roi_w)/640.0

        circle_small = max(3,int(5*scale))
        circle_big = max(24,int(orig_h/200))

        font_scale = 0.6*scale
        font_th = max(1,int(2*scale))

        # --------------------------
        # YOLO
        # --------------------------

        if kind=="yari":
            results = model_yari(roi,conf=0.6,imgsz=1280)[0]
        else:
            results = model_pipe(roi,conf=0.6,imgsz=1280)[0]

        img_draw = roi.copy()

        overlay = img_draw.copy()

        centers=[]
        counts={}

        # ==========================
        # Segmentation
        # ==========================

        if results.masks is not None:

            masks = results.masks.data.cpu().numpy()

            for mask,cls in zip(masks,results.boxes.cls):

                cls=int(cls)

                color=(0,0,255) if cls==0 else (255,0,0)

                mask=cv2.resize(mask,(roi.shape[1],roi.shape[0]))

                mask_bool = mask>0.5

                ys,xs = np.where(mask_bool)

                if len(xs)>0:

                    cx=int(xs.mean())
                    cy=int(ys.mean())

                    centers.append((cx,cy,color))

                if "MaskFill" in display:
                    overlay[mask_bool]=color

        if "MaskFill" in display:

            img_draw=cv2.addWeighted(
                overlay,0.3,
                img_draw,0.7,
                0
            )

        # ==========================
        # BOX
        # ==========================

        for box,cls,conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):

            x1,y1,x2,y2 = map(int,box)

            cls=int(cls)

            name=NAMES[cls]

            counts[name]=counts.get(name,0)+1

            color=(0,0,255) if cls==0 else (255,0,0)

            cx=(x1+x2)//2
            cy=(y1+y2)//2

            if kind=="pipe/muku":
                centers.append((cx,cy,color))

            if "Box" in display:

                cv2.rectangle(
                    img_draw,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    font_th
                )

            if "Label" in display:

                label=f"{name} {conf*100:.1f}"

                cv2.putText(
                    img_draw,
                    label,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0,255,0),
                    font_th
                )

        # ==========================
        # row sort
        # ==========================

        centers = row_sort(centers)

        # ==========================
        # numbers
        # ==========================

        # if "Numbers" in display:

        #     for i,(cx,cy,color) in enumerate(centers):

        #         cv2.putText(
        #             img_draw,
        #             str(i+1),
        #             (cx-10,cy+10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             font_scale,
        #             color,
        #             font_th
        #         )

        if "Numbers" in display:

            for i,(cx,cy,color) in enumerate(centers):

                if kind == "yari":
                    offset_x = -14
                    offset_y = 14
                else:
                    offset_x = -12
                    offset_y = 12

                cv2.putText(
                    img_draw,
                    str(i+1),
                    (cx+offset_x, cy+offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    font_th
                )

        # ==========================
        # circle
        # ==========================

        if "Circle" in display:

            for cx,cy,color in centers:

                r = circle_big if kind=="yari" else circle_small

                cv2.circle(
                    img_draw,
                    (cx,cy),
                    r,
                    color,
                    -1
                )

        # ==========================
        # QR detect
        # ==========================

        qr_texts=[]

        decoded_info,points = wechat_detector.detectAndDecode(roi)

        if points is not None:

            for text,pts in zip(decoded_info,points):

                if text=="": continue

                qr_texts.append(text)

                pts=pts.astype(int)

                for j in range(4):

                    cv2.line(
                        img_draw,
                        tuple(pts[j]),
                        tuple(pts[(j+1)%4]),
                        (0,255,0),
                        font_th
                    )

                cv2.putText(
                    img_draw,
                    text,
                    (pts[0][0],pts[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0,255,0),
                    font_th
                )

        # ==========================
        # return
        # ==========================

        _,buffer=cv2.imencode(".jpg",img_draw)

        img_base64=base64.b64encode(buffer).decode()

        return jsonify({
            "image":img_base64,
            "counts":counts,
            "qr":qr_texts
        })


# ==========================
# route
# ==========================

api.add_resource(
    SmartMatAPI,
    "/api/smart_mat_url/<name>"
)

# ==========================
# main
# ==========================

if __name__=="__main__":

    print("YOLO Factory Inspection Server Start")

    app.run(
        host="localhost",
        port=5001,
        debug=True
    )