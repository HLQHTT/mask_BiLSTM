<template>
  <div class="molecule">
    <div class="flex">
      <div class="left_content">
        <JSME
          height="500px"
          width="700px"
          options="oldlook,star,atommovebutton,hydrogens"
          :model-value="jsmeSmiles"
          :onChange="changeSmiles"
          v-show="activeKey === 'jsme'"
        />
      </div>
      <div class="right_content">
        <p>
          <a-button @click="getSmiles" type="primary" icon="search"> Get SMILES </a-button>
        </p>
        <p style="font-size: 14px">{{ jsmeSmiles }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import JSME from './jsmeEditor.vue'

export default {
  name: 'molecularIndex',
  components: {
    JSME
  },
  data() {
    return {
      activeKey: 'jsme',
      jsmeSmiles: null,
      currentSmiles: null,
      smiles: null
    }
  },
  methods: {
    onSearch() {
      this.$nextTick(() => {
        this.jsmeSmiles = this.smiles
      })
    },
    changeSmiles(val) {
      this.currentSmiles = val
    },
    getSmiles() {
      this.jsmeSmiles = this.currentSmiles
    }
  }
}
</script>

<style scoped>
.molecule {
  padding: 30px;
}

.input-search {
  width: 800px;
}

.flex {
  display: flex;
}

.molecule .left_content {
  width: 800px;
}

.molecule .right_content {
  width: calc(100% - 800px);
  margin-left: 50px;
  margin-top: 50px;
}
.el-login-footer {
  height: 40px;
  line-height: 40px;
  position: fixed;
  bottom: 0;
  width: 100%;
  text-align: center;
  font-family: Arial;
  font-size: 12px;
  letter-spacing: 1px;
}

.radio-group {
  display: grid;
}

.ant-radio-wrapper {
  padding: 10px 0 !important;
}
</style>
