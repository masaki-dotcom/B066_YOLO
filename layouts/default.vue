<script setup>
const pageMeta = usePageMeta()
definePageMeta({
  ssr: true // デフォルトは SSR
})

// ページが ssr:false なら、layout 側でもクライアントレンダリングにする
if (pageMeta.value?.ssr === false) {
  definePageMeta({ ssr: false })
}
</script>

<template>
  <slot />
</template>
