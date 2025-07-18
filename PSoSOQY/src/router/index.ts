import { createRouter, createWebHistory } from 'vue-router'
// import layOut from '../views/layOut/layOut.vue'
import App from '@/App.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  // history: createWebHistory('/api/'),
  // history: createWebHistory('http://localhost'),
  routes: [
    {
      path: '/', //路由地址
      //路由名字
      name: 'home',
      //页面组件
      component: () => import('@/views/pages/myHome.vue')
      // redirect: '/home', //重定向

      //子路由/嵌套路由
      // children: [
      //   {
      //     path: '/home',
      //     name: 'home',
      //     component: () => import('../views/pages/myHome.vue')
      //   },
      //   {
      //     path: '/predict',
      //     name: 'predict',
      //     component: () => import('../views/pages/myPrediction.vue')
      //   }
      // ]
    },
    // {
    //   path: '/home',
    //   name: 'home',
    //   component: () => import('../views/pages/myHome.vue')
    // },
    {
      path: '/predict',
      name: 'predict',
      component: () => import('@/views/pages/myPrediction.vue')
    },
    {
      path: '/result',
      name: 'result',
      component: () => import('@/views/pages/myResult.vue')
    },
    {
      path: '/contactUs',
      name: 'contactUs',
      component: () => import('@/views/pages/contactUs.vue')
    }
  ]
})

export default router
