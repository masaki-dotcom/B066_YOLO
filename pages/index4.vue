<template>
  <div class="contan">
      <div style="margin-bottom:6px">
          <label>
          <input type="radio" value="pipe/muku" v-model="kind">
          pipe/muku
          </label>
        
          <label class="ml-4">
          <input type="radio" value="yari" v-model="kind">
          yari
          </label>

      </div>
      <div>
           <input id="fileInput" type="file" class="hidden" @change="onFile"/>
            <label  for="fileInput" class="text-xs rounded px-3 py-1 font-bold bg-gray-400 text-blue-700 hover:text-white  cursor-pointer">
              画像選択
            </label>
          <label><input class="ml-4" type="checkbox" v-model="Box" /> Box</label>
          <label><input class="ml-4" type="checkbox" v-model="Label" /> Label</label>
          <label><input class="ml-4" type="checkbox" v-model="Circle" /> Circle</label>
          <label><input class="ml-4" type="checkbox" v-model="Numbers" /> Number</label>
          <label><input class="ml-4" type="checkbox" v-model="MaskFill" /> Mask Fill</label>
          <button class="text-xs rounded px-3 font-bold py-0.5 bg-gray-400 text-blue-700 hover:text-white ml-4" @click="send">推論</button>
      </div>
      
      <div class="count" style="margin-top:8px">        
        <!-- pipe: {{ counts.pipe }}
        muku: {{ counts.muku }} -->
         {{ kind=='pipe/muku' ?   `pipe: ${counts.pipe}`:'' }}
         {{ kind=='pipe/muku' ?   `muku: ${counts.muku}`: `yari: ${counts.muku}` }}
         {{ QR_code.length>0?   `品種: ${QR_code[0]}`:'' }}
      </div>
        
      <br />
      <canvas v-if="imageFile"
        ref="canvas"
        @mousedown="mouseDown"
        @mousemove="mouseMove"
        @mouseup="mouseUp"
      ></canvas>
  </div>  
</template>

<script setup>
import { ref, watch  } from "vue"
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
const Circle = ref(true)
const Numbers = ref(false)
const MaskFill = ref(false)

const counts = ref({ pipe: 0, muku: 0 })

const QR_code=ref([])

const MAX_SIZE = 900

const kind = ref("pipe/muku")   // pipe / muku / yari


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
// 排他制御
// --------------------
//BoxがONになったら他をOFF
watch(Box, (val) => {
  if (val) {
    Circle.value = false    
    Numbers.value = false
  }
})
// LabelがONになったら他をOFF
watch(Label, (val) => {
  if (val) {
    Circle.value = false    
    Numbers.value = false
  }
})
// NumbersがONになったら他をOFF
watch(Numbers, (val) => {
  if (val) {
    Box.value = false
    Label.value = false
    Circle.value = false
  }
})
// CircleがONになったら他をOFF
watch(Circle, (val) => {
  if (val) {
    Box.value = false
    Label.value = false
    Numbers.value = false
  }
})
// 全部falseなら Circle を true
watch([Numbers, Circle, Box, Label, MaskFill], () => {
  if (
    !Numbers.value &&
    !Circle.value &&
    !Box.value &&
    !Label.value &&
    !MaskFill.value
  ) {
    Circle.value = true
  }
})
watch(kind, (val) => {

  if (val === 'pipe/muku' ) {

    Circle.value = true
    Box.value = false
    Label.value = false
    Numbers.value = false
    MaskFill.value = false

  }

  if (val === 'yari') {

    Circle.value = false
    Box.value = false
    Label.value = false
    Numbers.value = false
    MaskFill.value = true

  }

})
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
  fd.append("kind", kind.value)   // ←追加

  if (Box.value) fd.append("classes[]", "Box")
  if (Label.value) fd.append("classes[]", "Label")
  if (Circle.value) fd.append("classes[]", "Circle")
  if (Numbers.value) fd.append("classes[]", "Numbers")
  if (MaskFill.value) fd.append("classes[]", "MaskFill")

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
resultWindow.focus()
const imgSrc = "data:image/jpeg;base64," + data.image

const html = `
<html>
<head>
<title>推論結果</title>
<style>
html, body {
  margin:0;
  padding:0;
  width:100%;
  height:100%;
  background:#111;
  overflow:hidden;
  display:flex;
  flex-direction:column;
}

#toolbar {
  background:#222;
  color:white;
  padding:6px;
  display:flex;
  align-items:center;
  gap:10px;
}

#viewer {
  flex:1;
  display:flex;
  justify-content:center;
  align-items:center;
  overflow:hidden;
  cursor:default;
}

img {
  width:100%;
  height:100%;
  object-fit:contain;
  transition: transform 0.2s ease;
  transform: scale(1);
}

button {
  padding:4px 10px;
  font-weight:bold;
  cursor:pointer;
}

/* ===== 印刷専用ヘッダー ===== */
#printHeader {
  display:none;
}

/* ===== 印刷設定 ===== */
@media print {

  #toolbar {
    display:none;
  }

  #printHeader {
    display:block;
    text-align:center;
    font-size:18px;
    font-weight:bold;
    margin:10px 0;
  }

  html, body {
    background:white;
  }

  img {
    width:100% !important;
    height:auto !important;
    transform:none !important;
  }
}
</style>
</head>

<body>
<div id="toolbar">
  <button id="printBtn" style="display:none;">印刷</button>
  <span id="countText">
     ${kind.value === 'pipe/muku' ? `pipe:${counts.value.pipe} / muku:${counts.value.muku}` :`yari: ${counts.value.muku}` }
  </span>
</div>

<!-- 印刷時のみ表示 -->
<div id="printHeader"> 
   ${kind.value === 'pipe/muku' ? `pipe:${counts.value.pipe} / muku:${counts.value.muku}` :`yari: ${counts.value.muku}` }
</div>


<div id="viewer">
  <img id="img" src="${imgSrc}" />
</div>
</body>
</html>
`

resultWindow.document.open()
resultWindow.document.write(html)
resultWindow.document.close()

// ===== JSを後から安全に追加 =====
const img = resultWindow.document.getElementById("img")
const viewer = resultWindow.document.getElementById("viewer")
const printBtn = resultWindow.document.getElementById("printBtn")

let zoomScale = 1
let posX = 0
let posY = 0
let isDragging = false
let startX = 0
let startY = 0

function updateTransform() {
  img.style.transform =
    "translate(" + posX + "px," + posY + "px) scale(" + zoomScale + ")"
}

// ===== 印刷ボタン =====
printBtn.addEventListener("click", () => {
  resultWindow.print()
})

// ===== ホイールズーム =====
viewer.addEventListener("wheel", (e) => {
  e.preventDefault()
  const delta = e.deltaY > 0 ? -0.1 : 0.1
  zoomScale += delta
  if (zoomScale < 1) zoomScale = 1
  if (zoomScale > 5) zoomScale = 5
  viewer.style.cursor = zoomScale > 1 ? "grab" : "default"
  updateTransform()
})

// ===== ドラッグ開始（拡大時のみ） =====
viewer.addEventListener("mousedown", (e) => {
  if (zoomScale <= 1) return
  isDragging = true
  startX = e.clientX - posX
  startY = e.clientY - posY
  viewer.style.cursor = "grabbing"
})

// ===== ドラッグ中 =====
viewer.addEventListener("mousemove", (e) => {
  if (!isDragging) return
  posX = e.clientX - startX
  posY = e.clientY - startY
  updateTransform()
})

// ===== ドラッグ終了 =====
viewer.addEventListener("mouseup", () => {
  isDragging = false
  viewer.style.cursor = zoomScale > 1 ? "grab" : "default"
})

viewer.addEventListener("mouseleave", () => {
  isDragging = false
  viewer.style.cursor = zoomScale > 1 ? "grab" : "default"
})

// ===== ダブルクリックでリセット =====
viewer.addEventListener("dblclick", () => {
  zoomScale = 1
  posX = 0
  posY = 0
  updateTransform()
  viewer.style.cursor = "default"
})

const countText = resultWindow.document.getElementById("countText")

// Ctrl + クリックで印刷ボタン表示切替
countText.addEventListener("click", (e) => {
  if (e.ctrlKey) {
    if (printBtn.style.display === "none") {
      printBtn.style.display = "inline-block"
    } else {
      printBtn.style.display = "none"
    }
  }
})

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
