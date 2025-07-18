<template>
  <el-row>
    <el-col :span="12">
      <div
        :style="{ width: 'auto', 'align-items': 'center', 'max-width': '100%', display: 'flex' }"
      >
        <img class="img" :src="'data:image/svg+xml;base64,' + svg" alt="fg_attribution" />
      </div>
    </el-col>
    <el-col :span="12">
      <div :style="{ width: 'auto', 'max-width': '100%' }">
        <el-descriptions :column="1" :size="large" border>
          <el-descriptions-item label="SMILES">{{ SMILES }} </el-descriptions-item>
          <el-descriptions-item label="ΦΔsam/ΦΔref">{{ Predicted_value }} </el-descriptions-item>
          <el-descriptions-item label="Solvent">{{ Solvent }} </el-descriptions-item>
          <el-descriptions-item label="Ref_Photosensitizer"
            >{{ Ref_Photosensitizer }}
          </el-descriptions-item>
          <el-descriptions-item label="Excitation Wavelength">509 nm</el-descriptions-item>
        </el-descriptions>
      </div>
    </el-col>
  </el-row>
  <el-row>
    <el-col :span="24">
      <div :style="{ width: 'auto', 'max-width': '100%', margin: '25px 0' }">
        <el-table
          :data="table1"
          border
          stripe
          :header-cell-style="{
            background: '#f5f7fa',
            color: '#000',
            height: '50px'
          }"
          :cell-style="{ color: '#333' }"
          :default-sort="{ prop: 'Attribution', order: 'descending' }"
        >
          <!-- <el-table-column prop="BRICS_substructure" label="BRICS_substructure" width="180" /> -->
          <el-table-column label="Substructure" width="200px">
            <template #default="scope">
              <el-image
                :src="'data:image/svg+xml;base64,' + scope.row.BRICS_substructure"
              ></el-image>
            </template>
          </el-table-column>
          <el-table-column prop="SMILES" label="SMILES" />
          <el-table-column prop="Index" label="Index" />
          <el-table-column prop="Attribution" label="Attribution" sortable />
        </el-table>
      </div>
    </el-col>
  </el-row>
  <el-row>
    <el-col :span="24">
      <div :style="{ width: auto, 'max-width': '100%', margin: '25px 0' }">
        <el-table
          :data="table2"
          border
          stripe
          :header-cell-style="{
            background: '#f5f7fa',
            color: '#000',
            height: '50px'
          }"
          :cell-style="{ color: '#333' }"
          :default-sort="{ prop: 'Attribution', order: 'descending' }"
        >
          <el-table-column label="Functional_group" width="200px">
            <template #default="scope">
              <el-image :src="'data:image/svg+xml;base64,' + scope.row.Functional_group"></el-image>
            </template>
          </el-table-column>
          <el-table-column prop="Functional_index" label="Functional_index" />
          <el-table-column prop="Attribution" label="Attribution" sortable />
        </el-table>
      </div>
    </el-col>
  </el-row>
</template>

<script setup>
import axios from 'axios'
import { ref, onMounted } from 'vue'

const table1 = ref([])
const table2 = ref([])
const SMILES = ref([])
const Predicted_value = ref([])
const Solvent = ref([])
const Ref_Photosensitizer = ref([])
const svg = ref('')

// 组件挂载后获取数据
onMounted(async () => {
  try {
    // 假设你的后端API是http://localhost:5000/data
    const response = await axios.get('http://8.138.160.173:80/api/result')
    // 将响应数据赋值给tableData
    table1.value = response.data.info.table1
    table2.value = response.data.info.table2
    SMILES.value = response.data.info.SMILES
    Predicted_value.value = response.data.info.Predicted_value
    Solvent.value = response.data.info.Solvent
    Ref_Photosensitizer.value = response.data.info.Ref_compound
    // svg.value = response.data.img
    svg.value = response.data.img
  } catch (error) {
    console.error('Error fetching data:', error)
  }
})
</script>

<style>
el-row {
  margin-bottom: 20px;
}
.el-row:last-child {
  margin-bottom: 0;
}
.el-col {
  border-radius: 4px;
}

.img {
  width: auto;
  max-width: 40%;
  margin: auto;
  height: auto;
}
/*设置描述列表整体的样式，这里是加了阴影*/
.el-descriptions {
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}
/*设置描述列表字体大小，行高*/
.el-descriptions__body .el-descriptions__table .el-descriptions__cell {
  font-size: 18px;
  line-height: 30px;
  word-break: break-all;
}
/*设置描述列表标签的样式*/
.el-descriptions__label.el-descriptions__cell.is-bordered-label {
  color: var(--el-text-color-primary);
  width: 220px;
}
/* 表格单元格和表头样式 */
.el-table {
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}
.el-table .cell {
  font-size: 18px; /* 调整字体大小 */
  text-align: center; /* 文本居中 */
  font-weight: bold; /* 加粗 */
}
</style>
