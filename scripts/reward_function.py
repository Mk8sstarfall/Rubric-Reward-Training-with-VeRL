"""Rubric Grading 奖励函数"""
import os, json, asyncio, aiohttp
from typing import Dict, Any, Optional
from scripts.utils import load_config, build_grading_prompt, parse_grading_response, check_format, extract_solution, get_project_root

DEFAULT_CONFIG = {"api_base": "http://localhost:8000/v1", "model_name": "Qwen/Qwen2.5-7B-Instruct", "temperature": 0.7, "max_tokens": 8192, "timeout": 600}

def get_grader_config() -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    for key, env in [("api_base", "GRADER_API_BASE"), ("model_name", "GRADER_MODEL_NAME"), ("timeout", "GRADER_TIMEOUT")]:
        if os.getenv(env):
            config[key] = os.getenv(env)
    if os.getenv("GRADER_TEMPERATURE"):
        config["temperature"] = float(os.getenv("GRADER_TEMPERATURE"))
    if os.getenv("GRADER_MAX_TOKENS"):
        config["max_tokens"] = int(os.getenv("GRADER_MAX_TOKENS"))
    cfg_path = get_project_root() / "configs" / "train_config.yaml"
    if cfg_path.exists():
        try:
            fc = load_config(str(cfg_path)).get("grader", {})
            config.update({k: v for k, v in fc.items() if v is not None})
        except: pass
    return config

async def call_grader_api(prompt: str, api_base: str, model_name: str, temperature: float = 0.7, max_tokens: int = 8192, timeout: int = 600) -> str:
    payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise Exception(f"API error {resp.status}")
            return (await resp.json())["choices"][0]["message"]["content"]

def call_grader_api_sync(prompt: str, api_base: str, model_name: str, temperature: float = 0.7, max_tokens: int = 8192, timeout: int = 600) -> str:
    return asyncio.run(call_grader_api(prompt, api_base, model_name, temperature, max_tokens, timeout))

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
    """verl兼容的同步奖励函数"""
    try:
        gt = json.loads(ground_truth)
    except:
        gt = {}
    rubric, ref = gt.get("rubric", []), gt.get("reference_solution", "")
    goal = extra_info.get("goal", "") if extra_info else ""
    if not rubric or not goal:
        return 0.0
    format_ok = check_format(solution_str)
    format_penalty = 0.0 if format_ok else 1.0
    solution = extract_solution(solution_str) or solution_str
    cfg = get_grader_config()
    try:
        resp = call_grader_api_sync(build_grading_prompt(goal, solution, rubric, ref), cfg["api_base"], cfg["model_name"], cfg["temperature"], cfg["max_tokens"], cfg["timeout"])
        score = parse_grading_response(resp, len(rubric))
    except Exception as e:
        import traceback
        print(f"Grader error: {type(e).__name__}: {e}")
        print(f"  API: {cfg['api_base']}, Model: {cfg['model_name']}")
        traceback.print_exc()
        score = 0.5 if format_ok else 0.0
    return max(0.0, score - format_penalty)

async def compute_score_async(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any], reward_router_address: str, reward_model_tokenizer=None) -> Dict[str, float]:
    """verl reward loop兼容的异步奖励函数"""
    try:
        gt = json.loads(ground_truth)
    except:
        gt = {}
    rubric, ref = gt.get("rubric", []), gt.get("reference_solution", "")
    goal = extra_info.get("goal", "")
    if not rubric or not goal:
        return {"score": 0.0}
    format_ok = check_format(solution_str)
    format_penalty = 0.0 if format_ok else 1.0
    solution = extract_solution(solution_str) or solution_str
    cfg = get_grader_config()
    api = reward_router_address or cfg["api_base"]
    try:
        resp = await call_grader_api(build_grading_prompt(goal, solution, rubric, ref), api, cfg["model_name"], cfg["temperature"], cfg["max_tokens"], cfg["timeout"])
        score = parse_grading_response(resp, len(rubric))
    except Exception as e:
        print(f"Async grader error: {e}")
        score = 0.5 if format_ok else 0.0
    return {"score": max(0.0, score - format_penalty)}

def compute_score_simple(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
    """简化版奖励函数（不需要grader）"""
    return 1.0 if check_format(solution_str) else 0.0
