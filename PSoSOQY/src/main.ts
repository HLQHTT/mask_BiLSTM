import './assets/main.css'

import { createApp} from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import request from '@/util/service'
// import axios from 'axios'

import * as ElementPlusIconsVue from '@element-plus/icons-vue'

const app = createApp(App)

// // 关闭生产模式下的提示
// Vue.config.productionTip = false
 
// // 设置axios为Vue的原型属性
// Vue.prototype.$axios = server

app.use(createPinia())
app.use(router)

app.use(ElementPlus)
app.provide('axios', request)
app.mount('#app')

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}
