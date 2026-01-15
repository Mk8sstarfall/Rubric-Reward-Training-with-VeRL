#!/bin/bash
# =============================================================================
# Grader 模型服务启动脚本
# 支持 ollama / vllm / sglang 后端
# =============================================================================

set -e

# ========================= 默认配置 =========================
BACKEND="vllm"
MODEL="/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507"
PORT=8000
MAX_TOKENS=8192
CONCURRENCY=512          # 最大并发请求数
GPU_COUNT=4              # GPU 数量
GPU_MEMORY_UTIL=0.9      # GPU 显存利用率 (vllm)
# QUANTIZATION=""          # 量化方式: awq, gptq, squeezellm, fp8
TENSOR_PARALLEL=4        # 张量并行度
CONTEXT_LENGTH=16384      # 上下文长度
DTYPE="auto"             # 数据类型: auto, half, float16, bfloat16, float32
HOST="0.0.0.0"
TRUST_REMOTE_CODE=true
CHAT_TEMPLATE=""         # 自定义聊天模板路径
API_KEY=""               # API 密钥 (可选)
LOG_LEVEL="info"

# ========================= 帮助信息 =========================
show_help() {
    cat << EOF
使用方法: $0 [选项]

后端选择:
  -b, --backend <name>      后端类型: ollama, vllm, sglang (默认: ollama)
  -m, --model <name>        模型名称 (默认: qwen2.5:7b)

服务配置:
  -p, --port <port>         服务端口 (默认: 8000)
  -H, --host <host>         绑定地址 (默认: 0.0.0.0)
  -k, --api-key <key>       API 密钥

性能参数:
  -c, --concurrency <n>     最大并发请求数 (默认: 256)
  -t, --max-tokens <n>      最大生成 token 数 (默认: 4096)
  -L, --context-length <n>  上下文长度 (默认: 8192)

GPU 配置:
  -g, --gpu-count <n>       GPU 数量 (默认: 1)
  -u, --gpu-util <0-1>      GPU 显存利用率 (默认: 0.9, 仅 vllm)
  -T, --tensor-parallel <n> 张量并行度 (默认: 1)

模型配置:
  -q, --quantization <type> 量化方式: awq, gptq, squeezellm, fp8
  -d, --dtype <type>        数据类型: auto, half, float16, bfloat16
  --chat-template <path>    自定义聊天模板路径

其他:
  --log-level <level>       日志级别: debug, info, warning, error
  --dry-run                 仅显示配置，不启动服务
  -h, --help                显示帮助信息

示例:
  $0 -b ollama -m qwen2.5:14b
  $0 -b vllm -m Qwen/Qwen2.5-7B-Instruct -g 2 -c 512 -t 8192
  $0 -b sglang -m meta-llama/Llama-3-8B-Instruct -T 2 -q awq
  $0 -b vllm -m TheBloke/Mistral-7B-GPTQ -q gptq --gpu-util 0.95

EOF
    exit 0
}

# ========================= 参数解析 =========================
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--backend)        BACKEND="$2"; shift 2;;
        -m|--model)          MODEL="$2"; shift 2;;
        -p|--port)           PORT="$2"; shift 2;;
        -H|--host)           HOST="$2"; shift 2;;
        -k|--api-key)        API_KEY="$2"; shift 2;;
        -c|--concurrency)    CONCURRENCY="$2"; shift 2;;
        -t|--max-tokens)     MAX_TOKENS="$2"; shift 2;;
        -L|--context-length) CONTEXT_LENGTH="$2"; shift 2;;
        -g|--gpu-count)      GPU_COUNT="$2"; shift 2;;
        -u|--gpu-util)       GPU_MEMORY_UTIL="$2"; shift 2;;
        -T|--tensor-parallel) TENSOR_PARALLEL="$2"; shift 2;;
        -q|--quantization)   QUANTIZATION="$2"; shift 2;;
        -d|--dtype)          DTYPE="$2"; shift 2;;
        --chat-template)     CHAT_TEMPLATE="$2"; shift 2;;
        --log-level)         LOG_LEVEL="$2"; shift 2;;
        --dry-run)           DRY_RUN=true; shift;;
        -h|--help)           show_help;;
        *)
            # 兼容旧版位置参数: ./start_grader.sh backend model port
            if [[ -z "${_POS_BACKEND:-}" ]]; then
                BACKEND="$1"; _POS_BACKEND=1
            elif [[ -z "${_POS_MODEL:-}" ]]; then
                MODEL="$1"; _POS_MODEL=1
            elif [[ -z "${_POS_PORT:-}" ]]; then
                PORT="$1"; _POS_PORT=1
            else
                echo "错误: 未知参数 '$1'"; echo "使用 -h 查看帮助"; exit 1
            fi
            shift;;
    esac
done

# ========================= 打印配置 =========================
print_config() {
    echo "=============================================="
    echo "         Grader 服务配置"
    echo "=============================================="
    echo "后端:           $BACKEND"
    echo "模型:           $MODEL"
    echo "端口:           $PORT"
    echo "主机:           $HOST"
    echo "----------------------------------------------"
    echo "最大并发:       $CONCURRENCY"
    echo "最大 Tokens:    $MAX_TOKENS"
    echo "上下文长度:     $CONTEXT_LENGTH"
    echo "----------------------------------------------"
    echo "GPU 数量:       $GPU_COUNT"
    [[ "$BACKEND" == "vllm" ]] && echo "显存利用率:     $GPU_MEMORY_UTIL"
    echo "张量并行:       $TENSOR_PARALLEL"
    echo "数据类型:       $DTYPE"
    [[ -n "$QUANTIZATION" ]] && echo "量化方式:       $QUANTIZATION"
    [[ -n "$CHAT_TEMPLATE" ]] && echo "聊天模板:       $CHAT_TEMPLATE"
    [[ -n "$API_KEY" ]] && echo "API 密钥:       ****${API_KEY: -4}"
    echo "=============================================="
}

print_config

# 环境变量输出 (供其他程序使用)
print_env() {
    echo ""
    echo "# 环境变量配置 (可添加到 .env 文件):"
    case $BACKEND in
        ollama)
            echo "GRADER_API_BASE=http://localhost:11434/v1";;
        *)
            echo "GRADER_API_BASE=http://${HOST}:${PORT}/v1";;
    esac
    echo "GRADER_MODEL_NAME=$MODEL"
    echo "GRADER_MAX_TOKENS=$MAX_TOKENS"
    [[ -n "$API_KEY" ]] && echo "GRADER_API_KEY=$API_KEY"
    echo ""
}

print_env

if $DRY_RUN; then
    echo "[Dry Run] 仅显示配置，不启动服务"
    exit 0
fi

# ========================= 启动服务 =========================
case $BACKEND in
    ollama)
        echo ">>> 检查 Ollama 服务..."
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "错误: Ollama 服务未运行"
            echo "请先执行: ollama serve"
            exit 1
        fi
        
        echo ">>> 检查模型 $MODEL..."
        if ! ollama list | grep -q "$MODEL"; then
            echo ">>> 拉取模型 $MODEL..."
            ollama pull "$MODEL"
        fi
        
        # Ollama 环境变量配置
        export OLLAMA_NUM_PARALLEL=$CONCURRENCY
        export OLLAMA_MAX_LOADED_MODELS=$GPU_COUNT
        
        echo ">>> Ollama 已就绪"
        echo ">>> API 地址: http://localhost:11434/v1"
        ;;
        
    vllm)
        echo ">>> 启动 vLLM 服务..."
        
        CMD="python -m vllm.entrypoints.openai.api_server"
        CMD+=" --model $MODEL"
        CMD+=" --port $PORT"
        CMD+=" --host $HOST"
        CMD+=" --max-model-len $CONTEXT_LENGTH"
        CMD+=" --max-num-seqs $CONCURRENCY"
        CMD+=" --gpu-memory-utilization $GPU_MEMORY_UTIL"
        CMD+=" --tensor-parallel-size $TENSOR_PARALLEL"
        CMD+=" --dtype $DTYPE"
        
        [[ "$TRUST_REMOTE_CODE" == "true" ]] && CMD+=" --trust-remote-code"
        [[ -n "$QUANTIZATION" ]] && CMD+=" --quantization $QUANTIZATION"
        [[ -n "$CHAT_TEMPLATE" ]] && CMD+=" --chat-template $CHAT_TEMPLATE"
        [[ -n "$API_KEY" ]] && CMD+=" --api-key $API_KEY"
        
        # vLLM 特有配置
        CMD+=" --enable-prefix-caching"
        CMD+=" --disable-log-requests"
        
        echo ">>> 执行命令:"
        echo "$CMD"
        echo ""
        
        eval $CMD
        ;;
        
    sglang)
        echo ">>> 启动 SGLang 服务..."
        
        CMD="python -m sglang.launch_server"
        CMD+=" --model-path $MODEL"
        CMD+=" --port $PORT"
        CMD+=" --host $HOST"
        CMD+=" --context-length $CONTEXT_LENGTH"
        CMD+=" --max-running-requests $CONCURRENCY"
        CMD+=" --tp $TENSOR_PARALLEL"
        CMD+=" --dtype $DTYPE"
        CMD+=" --log-level $LOG_LEVEL"
        
        [[ "$TRUST_REMOTE_CODE" == "true" ]] && CMD+=" --trust-remote-code"
        [[ -n "$QUANTIZATION" ]] && CMD+=" --quantization $QUANTIZATION"
        [[ -n "$CHAT_TEMPLATE" ]] && CMD+=" --chat-template $CHAT_TEMPLATE"
        [[ -n "$API_KEY" ]] && CMD+=" --api-key $API_KEY"
        
        # SGLang 特有配置
        CMD+=" --enable-flashinfer"
        
        echo ">>> 执行命令:"
        echo "$CMD"
        echo ""
        
        eval $CMD
        ;;
        
    *)
        echo "错误: 不支持的后端 '$BACKEND'"
        echo "支持的后端: ollama, vllm, sglang"
        exit 1
        ;;
esac