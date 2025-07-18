<!-- <script setup lang="ts">
import { RouterView } from 'vue-router'
import molecule from '@/components/molecularIndex.vue'
</script> -->

<!-- <template>
  <RouterView></RouterView>
  <molecule></molecule>
</template> -->

<template>
  <div class="common-layout">
    <el-container>
      <el-header class="common-header">
        <div class="flex">
          <!-- <img class="logo" src="@/components/icons/happy_lab_logo.svg" alt="happy lab logo" /> -->
          <h1 class="title">PSoSOQY</h1>
          <el-menu
            :default-active="$route.path"
            mode="horizontal"
            @select="handleSelect"
            :router="true"
            background-color="#545c64"
            text-color="#fff"
            active-text-color="#ffd04b"
            ellipsis
            class="common-menu"
          >
            <el-menu-item index="/">Home</el-menu-item>

            <el-menu-item index="/predict">Prediction</el-menu-item>
          </el-menu>
        </div>

        <el-button type="info" size="large" @click="$router.push('/contactUs')"
          >Contact Us</el-button
        >
      </el-header>

      <el-main>
        <RouterView />

        <!-- <molecule></molecule> -->
      </el-main>
      <!-- </el-container> -->
    </el-container>
  </div>
</template>

<script lang="ts" setup>
// import router from '@/router'
import { ref } from 'vue'
import { RouterView, useRouter } from 'vue-router'

// const isCollapse = ref(true)
// const activeIndex = ref('/home')
const handleSelect = (key: string, keyPath: string[]) => {
  console.log(key, keyPath)
}
</script>

<style>
body,
html {
  padding: 0;
  margin: 0;
}
.flex {
  display: flex;
  align-items: center;
}
.el-container {
  height: 100vh;
  overflow: hidden;
}
.common-header {
  background: #545c64;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.common-aside {
  background: #a0cfff;
}
.common-menu {
  width: 500px;
  display: flex;
  align-items: center;
  /* font-size: 50px; */
}
.logo {
  width: 80px;
}
.title {
  color: #fff;
}
</style>
