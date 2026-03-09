import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from ultralytics import YOLO
from flask_cors import CORS

# ========================
# Flask
# ========================

app = Flask(__name__)
CORS(app)
api = Api(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# YOLO モデル
# ========================

model_pipe = YOLO(os.path.join(BASE_DIR,"pipe/best.pt"))
model_yari = YOLO(os.path.join(BASE_DIR,"yari/best.pt"))

NAMES = {0:"muku",1:"pipe"}

qr_detector = cv2.QRCodeDetector()


# ========================
# 行ソート
# ========================

def row_sort(centers,th=40):

    centers = sorted(centers,key=lambda x:x[1])

    rows=[]
    cur=[]
    last=None

    for c in centers:

        if last is None:
            cur.append(c)
            last=c[1]
            continue

        if abs(c[1]-last)<th:
            cur.append(c)
        else:
            rows.append(cur)
            cur=[c]

        last=c[1]

    if cur:
        rows.append(cur)

    out=[]
    for r in rows:
        r=sorted(r,key=lambda x:x[0])
        out.extend(r)

    return out


# ========================
# API
# ========================

class smart_mat_url(Resource):

    def post(self,name):

        if name!="predict":
            return {"error":"invalid api"}

        # ----------------
        # 画像
        # ----------------

        if "image" not in request.files:
            return {"error":"no image"},400

        file=request.files["image"]

        img=cv2.imdecode(
            np.frombuffer(file.read(),np.uint8),
            cv2.IMREAD_COLOR
        )

        orig_h,orig_w=img.shape[:2]

        circle_radius = max(24, int(orig_h/200))

        # ROI
        x1=int(request.form.get("x1",0))
        y1=int(request.form.get("y1",0))
        x2=int(request.form.get("x2",0))
        y2=int(request.form.get("y2",0))
        kind = request.form.get("kind", "pipe/muku")

        x1,x2=sorted([x1,x2])
        y1,y2=sorted([y1,y2])

        x1=max(0,min(x1,orig_w-1))
        x2=max(0,min(x2,orig_w))
        y1=max(0,min(y1,orig_h-1))
        y2=max(0,min(y2,orig_h))

        if x2<=x1 or y2<=y1:
            roi=img
        else:
            roi=img[y1:y2,x1:x2]

        display=request.form.getlist("classes[]")

        # ---------------------
        # YOLO
        # ---------------------
        print(kind)
        print(display)
        if kind == "yari":
            results=model_yari(roi,conf=0.6)[0]
        else:
            results=model_pipe(roi,conf=0.6)[0]

        img_draw=roi.copy()

        centers=[]
        counts={}

        overlay=img_draw.copy()

        # ========================
        # Segmentation
        # ========================

        if results.masks is not None:

            masks=results.masks.data.cpu().numpy()

            for mask,cls in zip(masks,results.boxes.cls):

                cls=int(cls)

                color=(0,0,255) if cls==0 else (255,0,0)

                mask=cv2.resize(mask,(roi.shape[1],roi.shape[0]))
                mask_bool=mask>0.5

                ys,xs=np.where(mask_bool)

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

        # ---------------------
        # BOX
        # ---------------------

        for box,cls,conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):

            x1,y1,x2,y2=map(int,box)

            cls=int(cls)

            name=NAMES[cls]

            counts[name]=counts.get(name,0)+1

            color=(0,0,255) if cls==0 else (255,0,0)

            cx=(x1+x2)//2
            cy=(y1+y2)//2

            if kind == "pipe/muku":
                centers.append((cx,cy,color))

            if "Box" in display:

                cv2.rectangle(
                    img_draw,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

            if "Label" in display:

                label=f"{name} {conf*100:.1f}"

                cv2.putText(
                    img_draw,
                    label,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        # ---------------------
        # 行ソート
        # ---------------------

        centers=row_sort(centers)

        # ---------------------
        # Numbers
        # ---------------------

        if "Numbers" in display:

            for i,(cx,cy,color) in enumerate(centers):

                cv2.putText(
                    img_draw,
                    str(i+1),
                    (cx-10,cy+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        # ---------------------
        # Circle
        # ---------------------

        if "Circle" in display and kind == "pipe/muku":

            for cx,cy,color in centers:

                cv2.circle(
                    img_draw,
                    (cx,cy),
                    6,
                    color,
                    -1
                )
        elif "Circle" in display and kind == "yari":
            for cx,cy,color in centers:

                cv2.circle(
                    img_draw,
                    (cx,cy),
                    circle_radius,
                    color,
                    -1
                )


        # ---------------------
        # QR
        # ---------------------

        qr_text=""

        data,bbox,_=qr_detector.detectAndDecode(roi)

        if data:

            qr_text=data

        # ========================
        # return image
        # ========================

        _,buffer=cv2.imencode(".jpg",img_draw)

        img_base64=base64.b64encode(buffer).decode()

        return jsonify({
            "image":img_base64,
            "counts":counts,
            "qr":[qr_text] if qr_text else []
        })


# ========================
# route
# ========================

api.add_resource(
    smart_mat_url,
    "/api/smart_mat_url/<name>"
)


# ========================
# main
# ========================

if __name__=="__main__":

    print("YOLO server start")

    app.run(
        host="localhost",
        port=5001,
        debug=True
    )