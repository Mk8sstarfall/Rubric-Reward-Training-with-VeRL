"""Rubric Grading 奖励函数 - 真正的异步版本"""
import os, json, aiohttp
from typing import Dict, Any, Optional
from scripts.utils import (
    load_config, build_grading_prompt, parse_grading_response, 
    check_format, extract_solution, get_project_root
)

DEFAULT_CONFIG = {
    "api_base": "http://localhost:8000/v1",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.7,
    "max_tokens": 8192,
    "timeout": 600
}

def get_grader_config() -> Dict[str, Any]:
    """获取 grader 配置（保持不变）"""
    config = DEFAULT_CONFIG.copy()
    for key, env in [("api_base", "GRADER_API_BASE"), ("model_name", "GRADER_MODEL_NAME"), ("timeout", "GRADER_TIMEOUT")]:
        if os.getenv(env):
            config[key] = os.getenv(env)
    if (temp := os.getenv("GRADER_TEMPERATURE")):
        config["temperature"] = float(temp)
    if (max_tok := os.getenv("GRADER_MAX_TOKENS")):
        config["max_tokens"] = int(max_tok)
    cfg_path = get_project_root() / "configs" / "train_config.yaml"
    if cfg_path.exists():
        try:
            fc = load_config(str(cfg_path)).get("grader", {})
            config.update({k: v for k, v in fc.items() if v is not None})
        except: 
            pass
    return config


async def call_grader_api(
    prompt: str, 
    api_base: str, 
    model_name: str, 
    temperature: float = 0.7, 
    max_tokens: int = 8192, 
    timeout: int = 600
) -> str:
    """异步调用 grader API"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status != 200:
                raise Exception(f"API error {resp.status}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


# ============================================================
# 关键：异步奖励函数，符合 verl Reward Loop 接口
# ============================================================
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
    reward_router_address: str | None = None,  # verl 会传入，但我们用自己的 grader
    reward_model_tokenizer = None,       # verl 会传入，我们不需要
) -> float:
    """
    verl Reward Loop 兼容的异步奖励函数
    
    注意：这是一个 async 函数，verl 的 RewardLoopManager 会自动检测并正确处理
    """
    # 解析 ground_truth
    try:
        gt = json.loads(ground_truth)
    except:
        gt = {}
    
    rubric = gt.get("rubric", [])
    ref = gt.get("reference_solution", "")
    goal = extra_info.get("goal", "") if extra_info else ""
    
    if not rubric or not goal:
        return 0.0
    
    # 格式检查
    format_ok = check_format(solution_str)
    format_penalty = 0.0 if format_ok else 1.0
    
    # 提取 solution
    solution = extract_solution(solution_str) or solution_str
    
    # 获取配置
    cfg = get_grader_config()
    
    # 使用我们自己的 grader API（而不是 reward_router_address）
    api_base = cfg["api_base"]
    
    try:
        # 关键：直接 await，不要用 asyncio.run()
        resp = await call_grader_api(
            build_grading_prompt(goal, solution, rubric, ref),
            api_base,
            cfg["model_name"],
            cfg["temperature"],
            cfg["max_tokens"],
            cfg["timeout"]
        )
        score = parse_grading_response(resp, len(rubric))
    except Exception as e:
        import traceback
        print(f"Grader error: {type(e).__name__}: {e}")
        print(f"  API: {api_base}, Model: {cfg['model_name']}")
        traceback.print_exc()
        score = 0.5 if format_ok else 0.0
    
    return max(0.0, score - format_penalty)


# ============================================================
# 同步版本作为 fallback（用于非 async 模式或测试）
# ============================================================
def compute_score_sync(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """同步版本，用于测试或非 async 模式"""
    import asyncio
    return asyncio.run(compute_score(
        data_source, solution_str, ground_truth, 
        extra_info or {}, None, None
    ))