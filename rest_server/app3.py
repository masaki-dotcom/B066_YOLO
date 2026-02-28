from flask import Flask, Response, request, jsonify ,send_file
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import select
import json

import cv2
import numpy as np
import onnxruntime as ort
import io
import base64


app = Flask(__name__)
CORS(app)

# =====================
# モデル設定
# =====================
MODEL_PATH = "best.onnx"
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
# 前処理
# =====================
def preprocess(img):
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0

    img_chw = np.transpose(img_rgb, (2, 0, 1))
    blob = np.expand_dims(img_chw, axis=0)

    return blob, w, h



# =====================
# 推論API
# =====================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    # --------------------
    # 画像読み込み
    # --------------------
    file = request.files["image"]
    img_bytes = file.read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "invalid image"}), 400

    orig_h, orig_w = img.shape[:2]
    print("=== Flask: original image ===")
    print("image size:", orig_w, orig_h)

    # --------------------
    # ROI 座標（Nuxtから）
    # --------------------
    try:
        x1 = int(request.form.get("x1"))
        y1 = int(request.form.get("y1"))
        x2 = int(request.form.get("x2"))
        y2 = int(request.form.get("y2"))
    except:
        return jsonify({"error": "invalid roi"}), 400

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    if x2 <= x1 or y2 <= y1:
        return jsonify({"error": "empty roi"}), 400

    roi_w = x2 - x1
    roi_h = y2 - y1

    print("=== Flask: ROI received ===")
    print("roi:", x1, y1, x2, y2)
    print("roi size:", roi_w, roi_h)

    # --------------------
    # ROI 切り出し
    # --------------------
    roi_img = img[y1:y2, x1:x2].copy()

    # --------------------
    # 前処理
    # --------------------
    blob, _, _ = preprocess(roi_img)
    scale_x = roi_w / INPUT_SIZE
    scale_y = roi_h / INPUT_SIZE

    print("=== Flask: preprocess ===")
    print("YOLO input size:", INPUT_SIZE, INPUT_SIZE)
    print("scale_x:", scale_x, "scale_y:", scale_y)

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

        if score < 0.3:
            continue

        # ROI基準 bbox
        x = (xc - bw / 2) * scale_x
        y = (yc - bh / 2) * scale_y
        w_box = bw * scale_x
        h_box = bh * scale_y

        # 元画像座標へ変換 ★超重要
        x = int(x + x1)
        y = int(y + y1)
        w_box = int(w_box)
        h_box = int(h_box)

        boxes.append([x, y, w_box, h_box])
        scores.append(score)
        class_ids.append(cls)

    print("boxes:", len(boxes))
    if scores:
        print("scores min/max:", min(scores), max(scores))

    # --------------------
    # NMS
    # --------------------
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.3,
        nms_threshold=0.5
    )

    img_draw = img.copy()

    # --------------------
    # 元画像に描画
    # --------------------
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            cls = class_ids[i]
            score = scores[i]

            cv2.rectangle(
                img_draw,
                (x, y),
                (x + w_box, y + h_box),
                (0, 255, 0),
                2
            )

            label = f"{NAMES[cls]} {score*100:.1f}%"
            cv2.putText(
                img_draw,
                label,
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print("DRAW:", NAMES[cls], score, x, y, x + w_box, y + h_box)

    # --------------------
    # 元画像サイズで返却
    # --------------------
    _, buf = cv2.imencode(".jpg", img_draw)
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/jpeg"
    )



def get_connection():
    return psycopg2.connect(
        dbname="app_01",
        user="postgres",
        password="tankei001",
        host="localhost",
        port="5432"
    )

def fetch_emails():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM emails ORDER BY id;")
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result

# SSE用ジェネレーター
def event_stream():
    conn = get_connection()
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN emails_channel;")  # チャンネルを LISTEN

    last_data = None
    while True:
        if select.select([conn],[],[],10) == ([],[],[]):  # タイムアウト10秒
            continue
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            data = fetch_emails()
            if data != last_data:
                last_data = data
                yield f"data: {json.dumps(data)}\n\n"

@app.route('/emails')
def get_emails():
    return jsonify(fetch_emails())

@app.route('/emails/stream')
def stream_emails():
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/emails', methods=['POST'])
def add_email():
    payload = request.json
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO emails (name,email) VALUES (%s,%s)",
        (payload['name'], payload['email'])
    )
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "ok"}), 201


##更新
@app.route('/emails/<int:id>', methods=['PUT'])
def update_email(id):
    payload = request.json
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE emails SET name=%s, email=%s WHERE id=%s",
        (payload.get('name'), payload.get('email'), id)
    )
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "ok"})

#削除
@app.route('/emails/<int:id>', methods=['DELETE'])
def delete_email(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM emails WHERE id=%s", (id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="localhost", port=5000, threaded=True, debug=True)
