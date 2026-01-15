"""工具函数模块"""
import os, re, json, yaml, logging
from pathlib import Path
from typing import Dict, Any, List, Optional

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

RESEARCH_PLAN_PROMPT = """I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.

Here is the research scenario:
{goal}

Guidelines:
- The plan should address the goals of the scenario, and account for all constraints and confounders.
- Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed.
- The phrasing should NOT be verbose, and NOT be in past tense.
- Do not claim to have done any experiments or have results, just provide the plan.

Put the final solution inside <solution> </solution> XML tags. Maximum 750 words."""

def create_prompt(goal: str) -> str:
    return RESEARCH_PLAN_PROMPT.format(goal=goal)

def extract_solution(response: str) -> Optional[str]:
    match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def check_format(response: str, max_words: int = 750) -> bool:
    solution = extract_solution(response)
    return solution is not None and len(solution.split()) <= max_words

GENERAL_GUIDELINES = """**DESIDERATA**
1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item?
2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan include fully specified details on HOW to implement it?
3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws?
4. WELL JUSTIFIED RATIONALE: Is the plan well-motivated and justified?
5. COST AND EFFORT EFFICIENT: Does the plan handle this efficiently without unnecessary complexity?
6. NO ETHICAL ISSUES: Does this part have potential for negative consequences?
7. CONSISTENT WITH OVERALL PLAN: Is this part consistent with the rest of the plan?"""

def build_grading_prompt(goal: str, plan: str, rubric: List[str], reference_solution: str) -> str:
    rubric_text = "\n".join([f"Item {i+1}: {item}" for i, item in enumerate(rubric)])
    return f"""Evaluate if the Proposed Research Plan satisfies the Research Scenario.

# Research Scenario
{goal}

# Rubric
{rubric_text}

# Reference Solution
{reference_solution}

# Proposed Plan
{plan}

# Instructions
For each rubric item, check against:
{GENERAL_GUIDELINES}

Output JSON: {{"item_N": {{"satisfied": true/false, "reason": "..."}}}}
Final: {{"total_satisfied": N, "total_items": 10}}"""

def parse_grading_response(response: str, num_items: int = 10) -> float:
    match = re.search(r'\{"total_satisfied":\s*(\d+),\s*"total_items":\s*(\d+)\}', response)
    if match:
        return int(match.group(1)) / int(match.group(2))
    true_count = len(re.findall(r'"satisfied":\s*true', response, re.IGNORECASE))
    if true_count > 0:
        return true_count / num_items
    matches = re.findall(r'(\d+)\s*/\s*(\d+)', response)
    if matches:
        return int(matches[-1][0]) / int(matches[-1][1])
    return 0.0

def convert_to_verl_format(item: Dict[str, Any], data_source: str = "research_plan_gen") -> Dict[str, Any]:
    goal = item.get('Goal', '')
    rubric = item.get('Rubric', [])
    ref = item.get('Reference solution', '')
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": create_prompt(goal)}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"rubric": rubric, "reference_solution": ref})},
        "extra_info": {"goal": goal, "article_id": item.get('article_id', ''), "q_id": item.get('q_id', ''), "rubric": rubric, "reference_solution": ref}
    }

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_project_root() -> Path:
    return Path(__file__).parent.parent
