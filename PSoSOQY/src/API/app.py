from flask import Flask, request, jsonify
import sys
sys.path.append('D:\桌面\project\deep learning\mask_BiLSTM\Substructure_contribution.py')
from Substructure_contribution import return_attribution

app = Flask(__name__)

# 定义全局变量来存储result
result = None

@app.route('/submit', methods=['POST'])
def submit():
    # 获取JSON格式的表单数据
    form_data = request.get_json()
    # 处理数据（在这里我们只是将其添加到列表中）
    global result
    result = return_attribution(form_data['Photosensitizer'], form_data['Solvent'], form_data['Ref_Photosensitizer'])
    return jsonify(result), 200

@app.route('/data', methods=['GET'])
def get_data():
    # 返回所有数据
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)