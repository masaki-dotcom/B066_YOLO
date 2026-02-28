<script setup>
import { ref } from "vue"

const imageFile = ref(null)
const canvas = ref(null)
const ctx = ref(null)

const start = ref(null)
const roi = ref(null)

const pipe = ref(true)
const muku = ref(true)

const onFile = e => {
  imageFile.value = e.target.files[0]
  const img = new Image()
  img.onload = () => {
    canvas.value.width = img.width
    canvas.value.height = img.height
    ctx.value = canvas.value.getContext("2d")
    ctx.value.drawImage(img, 0, 0)
  }
  img.src = URL.createObjectURL(imageFile.value)
}

const mouseDown = e => {
  start.value = { x: e.offsetX, y: e.offsetY }
}

const mouseUp = e => {
  roi.value = {
    x1: start.value.x,
    y1: start.value.y,
    x2: e.offsetX,
    y2: e.offsetY
  }
  ctx.value.strokeStyle = "lime"
  ctx.value.strokeRect(
    roi.value.x1,
    roi.value.y1,
    roi.value.x2 - roi.value.x1,
    roi.value.y2 - roi.value.y1
  )
}

const send = async () => {
  const fd = new FormData()
  fd.append("image", imageFile.value)
  fd.append("x1", roi.value.x1)
  fd.append("y1", roi.value.y1)
  fd.append("x2", roi.value.x2)
  fd.append("y2", roi.value.y2)

  if (pipe.value) fd.append("classes[]", "pipe")
  if (muku.value) fd.append("classes[]", "muku")

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    body: fd
  })

  const blob = await res.blob()
  const img = new Image()
  img.onload = () => {
    ctx.value.clearRect(0,0,canvas.value.width,canvas.value.height)
    canvas.value.width = img.width
    canvas.value.height = img.height
    ctx.value.drawImage(img, 0, 0)
  }
  img.src = URL.createObjectURL(blob)
}
</script>

<template>
  <input type="file" @change="onFile" />
  <label><input type="checkbox" v-model="pipe" /> pipe</label>
  <label><input type="checkbox" v-model="muku" /> muku</label>
  <br />
  <canvas ref="canvas" @mousedown="mouseDown" @mouseup="mouseUp"></canvas>
  <br />
  <button @click="send">推論</button>
</template>
