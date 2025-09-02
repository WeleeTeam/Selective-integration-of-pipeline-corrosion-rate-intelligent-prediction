### example目录结构

- `corrosion_data.csv`: 真实腐蚀数据，列包含数值与类别特征以及目标列 `corrosion_rate`
- `data/utils.py`: 数据工具
  - 读取/生成表格数据（CSV）与时序数据（NPZ）
  - 加载腐蚀数据并完成数值标准化、类别独热编码、训练/测试划分
- `bp_mlp/`, `svm/`, `random_forest/`, `elm/`: 传统ML模型示例（表格回归）
- `lstm/`, `gru/`, `cnn/`: 深度学习时序示例（合成序列，演示用）
- `requirements.txt`: 依赖

### 环境依赖

```bash
pip install -r examples/requirements.txt
```

### 数据说明（腐蚀数据）

- 目标列：`corrosion_rate`
- 数值列：`temperature, pressure, flow_rate, Cl_conc, pH, O2_content, CO2_conc, H2S_conc, pipe_age`
- 类别列：`material, coating, corrosion_type`

数据加载流程位于 `examples/data/utils.py`：
- 数值列使用 `StandardScaler`
- 类别列使用 `OneHotEncoder(handle_unknown='ignore')`
- 默认 80%/20% 训练/测试拆分

当找不到或解析失败时，示例会自动回退使用假数据

### 运行方式（传统 ML：表格回归）

以下脚本将优先读取 `examples/corrosion_data.csv`：

输出包含测试集 MSE 与 R2。

### 运行方式（深度学习：时序示例）

当前 LSTM/GRU/CNN 使用合成单变量序列做一步预测，仅作模型结构演示，如需将真实腐蚀数据改造为时序窗口（按时间或管段序号滑窗）

### FAQ

- 找不到 `corrosion_data.csv`？
  - 确认文件存在于 `examples/` 目录下；脚本已使用绝对路径拼接，无需 `cd` 到特定目录。
- 类别列被当成数值？
  - 工具已在加载时强制将类别列转为字符串，仍有异常请检查源CSV是否有空列名或混合类型。
- TensorFlow 安装较慢或不需要深度学习示例？
  - 可仅安装 scikit-learn 相关依赖，或跳过运行 LSTM/GRU/CNN。
