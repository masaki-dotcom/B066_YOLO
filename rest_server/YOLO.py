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
model_square = YOLO(os.path.join(BASE_DIR, "square/best.pt"))
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
#Excel完全グリッド用
def row_sort_grid(centers):
    if not centers:
        return centers
    centers = sorted(centers, key=lambda x:x[1])

    rows=[]
    th=25   # 固定でもOK
    for c in centers:

        placed=False

        for r in rows:

            avg_y = np.mean([p[1] for p in r])

            if abs(c[1]-avg_y) < th:

                r.append(c)
                placed=True
                break

        if not placed:
            rows.append([c])

    # 行ソート
    rows=sorted(rows,key=lambda r:np.mean([p[1] for p in r]))

    out=[]

    for r in rows:

        # 列ソート
        r=sorted(r,key=lambda x:x[0])

        out.extend(r)

    return out
#少し斜め用（傾き補正）
def row_sort_tilt(centers):

    if len(centers)<3:
        return centers

    pts=np.array([[c[0],c[1]] for c in centers])

    vx,vy,x0,y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    angle=np.arctan2(vy,vx)

    rot=[]

    cos=np.cos(-angle)
    sin=np.sin(-angle)

    for c in centers:

        x=c[0]*cos-c[1]*sin
        y=c[0]*sin+c[1]*cos

        rot.append((x,y,c))

    rot=sorted(rot,key=lambda x:x[1])

    out=[r[2] for r in rot]

    return out
#ランダム
def row_sort_basic(centers):

    if not centers:
        return centers

    ys = [c[1] for c in centers]
    h = max(ys) - min(ys)

    th = max(10, int(h * 0.03))

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

class YoloAPI(Resource):

    def post(self,name):

        if name!="predict":
            return {"error":"invalid api"},400

        # --------------------------
        # image load
        # --------------------------

        if "image" not in request.files:
            return {"error":"no image"},400

        file = request.files["image"]

        file_bytes = file.read()

        img = cv2.imdecode(
            np.frombuffer(file_bytes,np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return {"error":"image decode failed"},400   
        # ==========================
        # Resize（メモリ対策）解像度2000以上なら自動縮小する処理
        # ========================== 
        MAX_SIZE = 2000

        h,w = img.shape[:2]

        resize_scale = 1.0   # ★追加

        if max(h,w) > MAX_SIZE:

            resize_scale = MAX_SIZE / max(h,w)

            img = cv2.resize(
                img,
                (int(w*resize_scale),int(h*resize_scale)),
                interpolation=cv2.INTER_AREA   # ★縮小品質向上
            )

        orig_h,orig_w = img.shape[:2]
        # --------------------------
        # ROI
        # --------------------------

        x1 = int(request.form.get("x1",0))
        y1 = int(request.form.get("y1",0))
        x2 = int(request.form.get("x2",0))
        y2 = int(request.form.get("y2",0))
        # ★追加（ROIも縮小）
        x1 = int(x1 * resize_scale)
        y1 = int(y1 * resize_scale)
        x2 = int(x2 * resize_scale)
        y2 = int(y2 * resize_scale)

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
            roi = img[y1:y2,x1:x2].copy()

        if roi.size == 0:
            roi = img

        display = request.form.getlist("classes[]")

        # --------------------------
        # draw parameter
        # --------------------------

        roi_h,roi_w = roi.shape[:2]
        scale = max(roi_h,roi_w)/640.0
        
       
        font_scale = 0.6*scale
        font_th = max(1,int(2*scale))
        box_heights=[]   # ★追加

        # --------------------------
        # YOLO
        # --------------------------

        if kind=="yari":
            results = model_yari(roi,conf=0.6,imgsz=1280)[0]
        elif kind=="square":
            results = model_square(roi,conf=0.6,imgsz=1280)[0]
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

                    centers.append((cx,cy,color,cls))

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

            # ★高さ取得
            box_h = y2-y1
            box_heights.append(box_h)

            cls=int(cls)

            name=NAMES[cls]

            counts[name]=counts.get(name,0)+1

            color=(0,0,255) if cls==0 else (255,0,0)

            cx=(x1+x2)//2
            cy=(y1+y2)//2

            if kind=="pipe/muku":
                centers.append((cx,cy,color,cls))

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
        # サイズ自動調整
        # ==========================

        if box_heights:

            avg_h = np.median(box_heights) # ★中央値

            font_scale = avg_h / 60
            font_th = max(1,int(avg_h/45))

            num_scale_base = avg_h / 75

        # ★circleはROI基準に変更（ここ重要）
        circle_small = max(3,int(min(roi_w,roi_h)/80))
        circle_big   = max(3,int(min(roi_w,roi_h)/80))
        # ==========================
        # row sort
        # ==========================

        if kind=="square":
            centers=row_sort_grid(centers)

        elif kind=="yari":
            centers=row_sort_basic(centers)

        elif kind=="pipe/muku":
            centers=row_sort_grid(centers) 

        if "Numbers" in display:

            for i,(cx,cy,color,cls) in enumerate(centers):

                # カラーとサイズ決定
                if kind == "yari" or kind == "square":

                    num_color = (255,0,0)

                    num_scale = num_scale_base * 0.8

                    num_th = max(1,int(font_th*0.8))

                else:

                    if cls==1:
                        num_color=(255,0,0)
                    else:
                        num_color=(0,0,255)

                    num_scale = num_scale_base

                    num_th = font_th


                # ★文字サイズ取得（これが重要）
                text=str(i+1)

                (text_w,text_h),baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    num_scale,
                    num_th
                )

                # ★完全中央配置
                draw_x = int(cx - text_w/2)
                draw_y = int(cy + text_h/2)


                # ★白縁取り（見やすくするならおすすめ）
                # cv2.putText(
                #     img_draw,
                #     text,
                #     (draw_x,draw_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     num_scale,
                #     (255,255,255),
                #     num_th+1
                # )

                # 本文字
                cv2.putText(
                    img_draw,
                    text,
                    (draw_x,draw_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    num_scale,
                    num_color,
                    num_th+1
                )

        # ==========================
        # circle
        # ==========================

        if "Circle" in display:
            for cx,cy,color,cls in centers:

                if kind=="yari" or kind == "square" :
                    r = circle_small
                    color= (255,0,0)
                else:
                    r = circle_big 

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
    YoloAPI,
    "/api/yolo_server_url/<name>"
)

# ==========================
# main
# ==========================

if __name__=="__main__":

    print("YOLO Factory Inspection Server Start")

    app.run(
        host="localhost",
        port=5005,
        debug=True
    )