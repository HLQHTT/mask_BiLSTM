<template>
  <el-row :gutter="24">
    <el-col :span="12" class="el-col">
      <div class="styled-paragraph">
        <p class="des_text">
          PSoSOQY is a system that utilizes a bidirectional long short-term memory network (BiLSTM)
          coupled with an attention mechanism to predict the quantum yield of singlet oxygen (SOQY)
          for photosensitizers. Beyond mere prediction, PSoSOQY is capable of providing contribution
          values for BRICS substructures and functional groups regarding the quantum yield of
          singlet oxygen. This capability can offer a reference for the design of photosensitizers
          and expedite the high-throughput screening of photosensitizer candidates.
        </p>
      </div>
      <img class="logo" src="@/components/icons/model overview_vertical.svg" alt="model overview" />
    </el-col>
    <el-col :span="12" class="el-col">
      <div class="styled-paragraph">
        <p class="des_text">
          PSoSOQY predicts the SOQY of photosensitizers at 509 nm using the DPBF consumption method,
          offering relative values as the ratio of ΦΔsam (SOQY of sample photosensitizer) to ΦΔref
          (SOQY of reference compound). Users simply need to input the SMILES of the photosensitizer
          they wish to predict, along with the solvent and the SMILES of the reference compound
          used, to obtain the aforementioned predictive value.
        </p>
        <p class="des_text">
          PSoSOQY can provide the contribution values of different BRICS substructures and
          functional groups to the quantum yield of singlet oxygen for a specific photosensitizer.
          The larger the contribution value, the greater the positive impact on the quantum yield of
          singlet oxygen.
        </p>

        <img class="logo" src="@/components/icons/fg_attribution.jpg" alt="fg_attribution" />
        <span class="caption">functional group attribution of our dataset.</span>
      </div>
    </el-col>
  </el-row>
</template>

<script lang="ts" setup></script>

<style scoped>
.logo {
  width: 80%;
  max-width: 80%;
  display: flex;
  margin: 20px auto;
  height: auto;
}
.styled-paragraph {
  background-color: #f9f9f9; /* 背景颜色 */
  border-radius: 8px; /* 圆角边框 */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 阴影效果 */
  padding: 20px; /* 内边距 */
  margin: 20px auto; /* 外边距 */
  border: 1px solid #eaeaea; /* 边框 */
  width: 100%;
}

.des_text {
  font-size: 18px; /* 字体大小 */
  color: #333; /* 字体颜色 */
  line-height: 1.6; /* 行高 */
  text-align: justify; /* 文本两端对齐 */
  margin-bottom: 20px; /* 段落间距 */
}
.caption {
  display: block;
  font-style: italic;
  margin-top: 10px;
  font-size: 18px;
}
</style>
