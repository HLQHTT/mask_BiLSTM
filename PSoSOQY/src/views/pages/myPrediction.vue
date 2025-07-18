<template>
  <el-row :gutter="20">
    <el-col :span="12">
      <el-form
        ref="ruleFormRef"
        :label-position="labelPosition"
        label-width="auto"
        :model="ruleForm"
        style="max-width: 700px"
        size="large"
        :rules="rules"
        status-icon
      >
        <el-form-item label="Photosensitizer" prop="Photosensitizer">
          <el-input v-model="ruleForm.Photosensitizer" maxlength="100" show-word-limit clearable />
        </el-form-item>
        <el-form-item label="Solvent" prop="Solvent">
          <el-input v-model="ruleForm.Solvent" maxlength="100" show-word-limit clearable />
        </el-form-item>
        <el-form-item label="Ref_Photosensitizer" prop="Ref_Photosensitizer">
          <el-input
            v-model="ruleForm.Ref_Photosensitizer"
            maxlength="100"
            show-word-limit
            clearable
          />
        </el-form-item>
      </el-form>
      <el-form-item>
        <el-button type="primary" @click="submitForm(ruleFormRef)"> Submit </el-button>
        <el-button @click="resetForm(ruleFormRef)">Reset</el-button>
      </el-form-item>
      <JSME class="jsme"></JSME>
      <!-- <iframe
        id="ifkatcher"
        src="@/components/standalone/index.html"
        width="auto"
        height="auto"
      ></iframe> -->
    </el-col>
    <el-col :span="12">
      <h1>An example of substructure explanation.</h1>
      <img class="logo" src="@/components/icons/example.svg" alt="fg_attribution" />
    </el-col>
  </el-row>
  <!-- <el-row :gutter="20">
    <el-col id="jsme_container" :span="24">JSME</el-col>
  </el-row> -->
</template>

<script lang="ts" setup>
// import { Search } from '@element-plus/icons-vue'
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
const router = useRouter()
import type { FormProps, FormInstance, FormRules } from 'element-plus'
import { ElLoading } from 'element-plus'
import { ElMessage } from 'element-plus'
// import molecularIndex from '@/components/molecularIndex.vue'
import JSME from '@/components/jsmeEditor.vue'
import axios from 'axios'
const labelPosition = ref<FormProps['labelPosition']>('top')
// const openFullScreen = () => {
//   const loading = ElLoading.service({
//     lock: true,
//     text: 'Loading',
//     background: 'rgba(0, 0, 0, 0.5)'
//   })
//   setTimeout(() => {
//     loading.close()
//   }, 2000)
// }
interface RuleForm {
  Photosensitizer: string
  Solvent: string
  Ref_Photosensitizer: string
}
const ruleFormRef = ref<FormInstance>()
const ruleForm = reactive<RuleForm>({
  Photosensitizer: '',
  Solvent: '',
  Ref_Photosensitizer: ''
})

const rules = reactive<FormRules<RuleForm>>({
  Photosensitizer: [{ required: true, message: 'Please input Photosensitizer', trigger: 'blur' }],
  Solvent: [
    {
      required: true,
      message: 'Please input Solvent',
      trigger: 'blur'
    }
  ],
  Ref_Photosensitizer: [
    {
      required: true,
      message: 'Please input Ref_Photosensitizer',
      trigger: 'blur'
    }
  ]
})

const submitForm = async (formEl: FormInstance | undefined) => {
  if (!formEl) return
  // await formEl.validate((valid, fields) => {
  //   if (valid) {
  //     console.log('submit!'), openFullScreen(), router.push('/result')
  //   } else {
  //     console.log('error submit!', fields)
  //   }
  // })
  await formEl.validate(async (valid, fields) => {
    if (valid) {
      console.log('submit!')
      const loading = ElLoading.service({
        lock: true,
        text: 'Loading',
        background: 'rgba(0, 0, 0, 0.7)'
      })
      try {
        // 使用axios提交表单数据到后端
        const response = await axios.post('http://8.138.160.173:80/api/predict', ruleForm) // 假设你的后端API是/predict
        if (response.status === 201 || response.status === 200) {
          // openFullScreen()
          loading.close()
          router.push('/result')
        } else {
          console.error('Server returned an error status:', response.status)
          ElMessage.error('An error has occurred!')
          loading.close()
        }
      } catch (error) {
        console.error('Error submitting form:', error)
        ElMessage.error('An error has occurred!')
        loading.close()
      }
    } else {
      console.log('error submit!', fields)
      ElMessage.error('An error has occurred!')
    }
  })
}

const resetForm = (formEl: FormInstance | undefined) => {
  if (!formEl) return
  formEl.resetFields()
}
</script>

<style scoped>
.search_button {
  display: flex;
  margin: 20px auto;
}
.input_box {
  display: flex;
  margin: 20px auto;
  padding: 0;
}
.el-row {
  margin-bottom: 20px;
}
.el-row:last-child {
  margin-bottom: 0;
}
.el-col {
  border-radius: 4px;
  margin: auto;
}
.logo {
  width: 100%;
  max-width: 100%;
  display: flex;
  margin: auto;
  height: auto;
}
.jsme {
  display: flex;
  margin: auto;
  width: auto;
  /* justify-content: center; */
}
</style>
