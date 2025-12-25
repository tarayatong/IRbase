# 训练指南 - 加载预训练权重

本指南说明如何使用新增的预训练权重加载功能。

## 🆕 新增功能

### 1. **加载预训练权重进行训练**
### 2. **恢复训练状态（包括optimizer）**
### 3. **冻结encoder只训练decoder**
### 4. **部分加载权重（允许键不匹配）**

---

## 📋 命令行参数

### 预训练权重相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pretrained-weights` | str | None | 预训练权重文件路径 |
| `--load-strict` | flag | False | 严格加载模式（所有键必须匹配） |
| `--freeze-encoder` | flag | False | 冻结encoder权重，只训练decoder |
| `--resume-training` | flag | False | 恢复训练（包括optimizer和epoch状态） |

---

## 🚀 使用示例

### 示例 1: 从头开始训练（不加载任何权重）

```bash
python demo.py \
  --output ./workdirs/experiment1 \
  --learning_rate 1e-4 \
  --epoch_num 200 \
  --batch_size_train 4
```

### 示例 2: 加载预训练权重继续训练

```bash
python demo.py \
  --output ./workdirs/experiment2 \
  --pretrained-weights ./workdirs/experiment1/best.pth \
  --learning_rate 1e-4 \
  --epoch_num 200
```

**说明**：
- 加载 `experiment1` 的最佳模型权重
- 从 epoch 1 开始训练（不恢复训练状态）
- 允许部分加载（默认 `--load-strict=False`）

### 示例 3: 恢复训练（继续上次的训练）

```bash
python demo.py \
  --output ./workdirs/experiment1 \
  --pretrained-weights ./workdirs/experiment1/epoch_100.pth \
  --resume-training \
  --learning_rate 1e-4 \
  --epoch_num 200
```

**说明**：
- 加载 epoch 100 的checkpoint
- `--resume-training` 会恢复optimizer状态和epoch计数
- 会从 epoch 101 继续训练
- 保持之前的最佳IoU记录

### 示例 4: 冻结encoder，只训练decoder（微调）

```bash
python demo.py \
  --output ./workdirs/finetune \
  --pretrained-weights ./workdirs/experiment1/best.pth \
  --freeze-encoder \
  --learning_rate 5e-5 \
  --epoch_num 50
```

**说明**：
- 冻结 `image_encoder` 和 `edge_encoder` 的所有参数
- 只训练 decoder 部分
- 适合在新数据集上微调
- 通常使用更小的学习率

### 示例 5: 严格加载模式（所有键必须完全匹配）

```bash
python demo.py \
  --output ./workdirs/experiment3 \
  --pretrained-weights ./pretrained/model.pth \
  --load-strict \
  --learning_rate 1e-4
```

**说明**：
- `--load-strict` 要求所有参数键完全匹配
- 如果有任何缺失或多余的键，加载会失败
- 适合确保完全相同的模型结构

### 示例 6: 组合使用（恢复训练 + 冻结encoder）

```bash
python demo.py \
  --output ./workdirs/experiment4 \
  --pretrained-weights ./workdirs/experiment4/epoch_50.pth \
  --resume-training \
  --freeze-encoder \
  --learning_rate 1e-5 \
  --epoch_num 100
```

**说明**：
- 从 epoch 50 恢复训练
- 冻结encoder，只训练decoder
- 适合在训练中期切换策略

---

## 📊 Checkpoint文件格式

新的checkpoint保存格式包含更多信息：

```python
checkpoint = {
    'epoch': 当前epoch数,
    'model': 模型的state_dict,
    'optimizer': 优化器的state_dict,
    'best_iou': 当前最佳IoU,
    'eval_metrics': 评估指标字典,
    'train_metrics': 训练指标字典,
    'args': 训练参数配置
}
```

### 保存的文件

1. **best.pth** - 最佳IoU的模型
2. **epoch_N.pth** - 每N个epoch保存一次（N由 `--model_save_fre` 设置）

---

## 🔍 加载过程说明

### 1. 权重加载流程

```
1. 检查文件是否存在
2. 加载checkpoint文件
3. 识别checkpoint格式（支持多种格式）
4. 加载模型权重（严格或非严格模式）
5. 显示加载结果（成功/缺失/多余的keys）
6. 如果指定--freeze-encoder，冻结相应参数
```

### 2. 恢复训练状态流程

```
1. 加载checkpoint
2. 恢复optimizer状态
3. 恢复scheduler状态（如果有）
4. 获取start_epoch
5. 获取best_iou
6. 从正确的epoch继续训练
```

---

## ⚙️ 高级用法

### 从不同结构的模型迁移学习

如果要从结构稍有不同的模型加载权重：

```bash
python demo.py \
  --output ./workdirs/transfer \
  --pretrained-weights ./other_model/checkpoint.pth \
  --learning_rate 1e-4
```

**注意**：默认的 `--load-strict=False` 会自动处理不匹配的keys。

### 只加载特定部分的权重

如果checkpoint中有你不需要的部分，非严格模式会自动跳过：

```bash
# 会自动跳过不匹配的部分
python demo.py \
  --output ./workdirs/partial \
  --pretrained-weights ./full_model.pth
```

---

## 🐛 故障排除

### 问题 1: "缺失的keys"警告

**原因**：当前模型有一些参数在checkpoint中不存在。

**解决方案**：
- 检查模型结构是否改变
- 如果是有意的改变，可以忽略警告（这些参数会随机初始化）
- 如果不是有意的，检查checkpoint文件是否正确

### 问题 2: "多余的keys"警告

**原因**：checkpoint中有一些参数在当前模型中不存在。

**解决方案**：
- 检查模型结构是否改变
- 如果是有意的简化，可以忽略警告
- 如果不是有意的，检查模型定义是否正确

### 问题 3: 加载失败

**检查清单**：
1. 文件路径是否正确
2. 文件是否损坏
3. 是否有足够的内存
4. 如果使用 `--load-strict`，检查模型结构是否完全一致

---

## 📝 最佳实践

### 1. 定期保存checkpoint

```bash
# 每10个epoch保存一次
--model_save_fre 10
```

### 2. 保留多个checkpoint

不要只保留best.pth，定期的epoch checkpoint也很重要：
- 可以回退到之前的训练状态
- 可以分析训练过程
- 避免单点故障

### 3. 微调时使用较小的学习率

```bash
# 原始训练
--learning_rate 1e-4

# 微调时
--learning_rate 5e-5 或 1e-5
```

### 4. 使用恢复训练功能

如果训练中断，使用 `--resume-training` 而不是从头开始：

```bash
python demo.py \
  --output ./workdirs/experiment \
  --pretrained-weights ./workdirs/experiment/epoch_50.pth \
  --resume-training
```

### 5. 记录训练配置

所有训练参数都保存在checkpoint中的 `args` 字段，可以查看历史配置。

---

## 📈 监控训练

### 查看加载信息

运行训练时会显示详细的加载信息：

```
================================================================
加载预训练权重
================================================================
📦 正在加载预训练权重: ./workdirs/experiment1/best.pth
✅ 成功加载权重: 256/256 个参数
📊 Checkpoint 来自 epoch: 100
📊 Checkpoint 的 best IoU: 0.8523
================================================================
```

### 冻结encoder的确认

```
🔒 正在冻结 encoder 权重...
✅ 已冻结 128 个 encoder 参数
```

### 恢复训练状态的确认

```
================================================================
恢复训练状态
================================================================
📦 正在恢复训练状态: ./workdirs/experiment/epoch_50.pth
✅ 已恢复 optimizer 状态
✅ 将从 epoch 51 继续训练
✅ 当前最佳 IoU: 0.8234
================================================================
```

---

## 💡 提示

1. **第一次训练**：不使用任何预训练参数
2. **继续训练**：使用 `--pretrained-weights` + `--resume-training`
3. **迁移学习**：使用 `--pretrained-weights`，不使用 `--resume-training`
4. **微调**：使用 `--pretrained-weights` + `--freeze-encoder`
5. **调试**：使用较小的 `--epoch_num` 和 `--batch_size_train`

---

## 🔗 相关文件

- `demo.py` - 主训练脚本
- `segment_anything_training/build_IRSAM.py` - 模型构建
- `utils/alpha_loss.py` - Alpha损失函数

---

## ❓ 常见问题

**Q: 什么时候使用 `--resume-training`？**

A: 当训练中断，想要从上次的状态继续时使用。它会恢复optimizer状态和epoch计数。

**Q: `--pretrained-weights` 和 `--restore-model` 的区别？**

A: 
- `--pretrained-weights`: 新的推荐方式，功能更强大
- `--restore-model`: 旧的方式，保留用于兼容性

**Q: 可以只加载部分权重吗？**

A: 可以，默认的非严格模式（`--load-strict=False`）会自动处理不匹配的keys。

**Q: checkpoint文件很大，如何减小？**

A: Checkpoint包含model、optimizer等完整信息。如果只需要模型权重：
```python
# 只保存模型
torch.save(net.state_dict(), 'model_only.pth')
```

---

最后更新：2025-12

