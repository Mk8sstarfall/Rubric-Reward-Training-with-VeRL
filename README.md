# Rubric Rewards 论文复现

复现 **"Training AI Co-Scientists Using Rubric Rewards"** (Meta, arXiv:2512.23707)

## 结构
```
├── configs/train_config.yaml   # 训练配置
├── scripts/
│   ├── utils.py                # 工具函数
│   ├── prepare_data.py         # 数据准备
│   ├── reward_function.py      # Rubric奖励函数
│   └── train_grpo.py           # GRPO训练
├── run_train.sh                # 训练脚本
└── start_grader.sh             # Grader服务
```

## 快速开始

```bash
# 1. 安装依赖
pip install verl datasets pandas pyyaml aiohttp

# 2. 启动Grader (Ollama)
ollama serve && ollama pull qwen2.5:7b

# 3. 训练
./run_train.sh --subset ml --n_gpus 4
```

## 核心配置（论文对齐）

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-6 | 论文设置 |
| rollout.n | 8 | GRPO组大小 |
| use_kl_loss | false | **禁用KL惩罚** |
| temperature | 0.7 | 采样温度 |

## Grader配置

```yaml
grader:
  api_base: "http://localhost:11434/v1"
  model_name: "qwen2.5:7b"
```

或环境变量: `GRADER_API_BASE`, `GRADER_MODEL_NAME`

## 简化模式（无Grader）
```bash
./run_train.sh --simple_reward
```

## 数据集

[facebook/research-plan-gen](https://huggingface.co/datasets/facebook/research-plan-gen): ml(6.8k), arxiv(6.5k), pubmed(7.3k)

## 参考
- 论文: [arXiv:2512.23707](https://arxiv.org/abs/2512.23707)
- verl: [verl.readthedocs.io](https://verl.readthedocs.io/)
