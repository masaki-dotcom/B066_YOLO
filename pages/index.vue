<template>
  <div class="contan">
      <div>
        <input type="file" @change="onFile" />
          <label><input class="ml-2" type="checkbox" v-model="Box" /> Box</label>
          <label><input class="ml-2" type="checkbox" v-model="Label" /> Label</label>
          <button class="'text-xs rounded px-3 font-bold py-0.5 bg-gray-400 text-blue-700 hover:text-white ml-4" @click="send">推論</button>
      </div>
      
      <div class="count" style="margin-top:8px">        
        pipe: {{ counts.pipe }}
        muku: {{ counts.muku }}
         {{ QR_code.length>0?   `品種: ${QR_code[0]}`:'' }}
      </div>
        
      <br />
      <canvas
        ref="canvas"
        @mousedown="mouseDown"
        @mousemove="mouseMove"
        @mouseup="mouseUp"
      ></canvas>
  </div>  
</template>

<script setup>
import { ref } from "vue"
const url_name=UrlStore() //piniaからグローバル定数を所得

const imageFile = ref(null)
const canvas = ref(null)
const ctx = ref(null)

const imgObj = ref(null)
const scale = ref(1)

const start = ref(null)
const roi = ref(null)
const isDragging = ref(false)

const Box = ref(false)
const Label = ref(false)

const counts = ref({ pipe: 0, muku: 0 })

const QR_code=ref([])

const MAX_SIZE = 1400

// ★ 推論結果用ウインドウ
let resultWindow = null

// --------------------
// 画像読み込み
// --------------------
const onFile = e => {
  imageFile.value = e.target.files[0]
  roi.value = null
  start.value = null
  isDragging.value = false

  imgObj.value = new Image()
  imgObj.value.onload = () => {
    const w = imgObj.value.width
    const h = imgObj.value.height

    scale.value = Math.min(MAX_SIZE / w, MAX_SIZE / h, 1)

    canvas.value.width = Math.round(w * scale.value)
    canvas.value.height = Math.round(h * scale.value)

    ctx.value = canvas.value.getContext("2d")
    ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height)
    ctx.value.drawImage(
      imgObj.value,
      0,
      0,
      canvas.value.width,
      canvas.value.height
    )
  }

  imgObj.value.src = URL.createObjectURL(imageFile.value)
}

// --------------------
// ROI選択
// --------------------
const mouseDown = e => {
  start.value = { x: e.offsetX, y: e.offsetY }
  isDragging.value = true
}

const mouseMove = e => {
  if (!isDragging.value || !start.value) return

  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height)
  ctx.value.drawImage(
    imgObj.value,
    0,
    0,
    canvas.value.width,
    canvas.value.height
  )

  ctx.value.strokeStyle = "lime"
  ctx.value.lineWidth = 2
  ctx.value.strokeRect(
    Math.min(start.value.x, e.offsetX),
    Math.min(start.value.y, e.offsetY),
    Math.abs(e.offsetX - start.value.x),
    Math.abs(e.offsetY - start.value.y)
  )
}

const mouseUp = e => {
  if (!start.value) return

  isDragging.value = false

  roi.value = {
    x1: Math.min(start.value.x, e.offsetX),
    y1: Math.min(start.value.y, e.offsetY),
    x2: Math.max(start.value.x, e.offsetX),
    y2: Math.max(start.value.y, e.offsetY)
  }

  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height)
  ctx.value.drawImage(
    imgObj.value,
    0,
    0,
    canvas.value.width,
    canvas.value.height
  )

  ctx.value.strokeStyle = "lime"
  ctx.value.lineWidth = 2
  ctx.value.strokeRect(
    roi.value.x1,
    roi.value.y1,
    roi.value.x2 - roi.value.x1,
    roi.value.y2 - roi.value.y1
  )
}

// --------------------
// 推論送信
// --------------------
const send = async () => {
  if (!roi.value) {
    alert("ROIを選択してください")
    return
  }

  const sendRoi = {
    x1: Math.round(roi.value.x1 / scale.value),
    y1: Math.round(roi.value.y1 / scale.value),
    x2: Math.round(roi.value.x2 / scale.value),
    y2: Math.round(roi.value.y2 / scale.value)
  }

  const fd = new FormData()
  fd.append("image", imageFile.value)
  fd.append("x1", sendRoi.x1)
  fd.append("y1", sendRoi.y1)
  fd.append("x2", sendRoi.x2)
  fd.append("y2", sendRoi.y2)

  if (Box.value) fd.append("classes[]", "Box")
  if (Label.value) fd.append("classes[]", "Label")

  const base = url_name.smart_mat_url.replace(/\/$/, "")

  const data = await $fetch(`${base}/predict`, {
    method: "POST",
    body: fd
  })

  counts.value.pipe = data.counts.pipe ?? 0
  counts.value.muku = data.counts.muku ?? 0
  QR_code.value = data.qr || []

  // --------------------
  // ★ 別ウインドウ表示
  // --------------------
  if (!resultWindow || resultWindow.closed) {
    resultWindow = window.open("", "resultWindow", "width=900,height=700")
  }
   // ★ 重要：必ずフォーカス
  resultWindow.focus()

  resultWindow.document.open()
  resultWindow.document.write(`
    <html>
      <head>
        <title>推論結果</title>
        <style>
          body {
            margin:0;
            background:#000;
            display:flex;
            justify-content:center;
            align-items:center;
          }
          img { max-width:100%; max-height:100%; }
        </style>
      </head>
      <body>
        <img src="data:image/jpeg;base64,${data.image}" />
      </body>
    </html>
  `)
  resultWindow.document.close()
}
</script>

<style scoped>
.contan{
  padding: 6px;
}
.count{
  font-size: 20px;
  font-weight: bold;
}
canvas {
  border: 1px solid #ccc;
  margin-top: 8px;
}
</style>
