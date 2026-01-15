"""GRPO训练脚本"""
import os, sys, argparse, subprocess
from pathlib import Path
from typing import Dict, Any, List
from utils import load_config, setup_logger, get_project_root, ensure_dir

logger = setup_logger("train_grpo")

def build_verl_command(config: Dict[str, Any], args) -> List[str]:
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"]
    root = get_project_root()
    
    algo = config.get("algorithm", {})
    cmd.extend([
        f"algorithm.adv_estimator={algo.get('name', 'grpo')}",
        f"algorithm.use_kl_in_reward={str(algo.get('use_kl_in_reward', False)).lower()}",
        f"algorithm.gamma={algo.get('gamma', 1.0)}",
        f"algorithm.lam={algo.get('lam', 1.0)}",
        f"algorithm.kl_ctrl.type={algo.get('kl_ctrl', {}).get('type', 'fixed')}",
        f"algorithm.kl_ctrl.kl_coef={algo.get('kl_ctrl', {}).get('kl_coef', 0.0)}",
    ])
    
    data = config.get("data", {})
    train_f = data.get("train_file", "data/ml_train.parquet")
    val_f = data.get("val_file", "data/ml_test.parquet")
    if not Path(train_f).is_absolute(): train_f = str(root / train_f)
    if not Path(val_f).is_absolute(): val_f = str(root / val_f)
    cmd.extend([
        f"data.train_files={train_f}", f"data.val_files={val_f}",
        f"data.train_batch_size={data.get('train_batch_size', 64)}",
        f"data.max_prompt_length={data.get('max_prompt_length', 2048)}",
        f"data.max_response_length={data.get('max_response_length', 2048)}",
        "data.filter_overlong_prompts=True", "data.truncation=error", "data.prompt_key=prompt",
    ])
    
    reward_fn = "compute_score"
    cmd.extend([f"custom_reward_function.path={root / 'scripts' / 'reward_function.py'}", f"custom_reward_function.name={reward_fn}"])
    
    model = config.get("model", {})
    cmd.extend([
        f"actor_rollout_ref.model.path={model.get('actor_path', 'Qwen/Qwen2.5-3B-Instruct')}",
        f"actor_rollout_ref.model.trust_remote_code={str(model.get('trust_remote_code', True)).lower()}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
    ])
    
    actor = config.get("actor", {})
    cmd.extend([
        f"actor_rollout_ref.actor.optim.lr={actor.get('learning_rate', 1e-6)}",
        f"actor_rollout_ref.actor.grad_clip={actor.get('grad_clip', 1.0)}",
        f"actor_rollout_ref.actor.use_kl_loss={str(actor.get('use_kl_loss', False)).lower()}",
        f"actor_rollout_ref.actor.entropy_coeff={actor.get('entropy_coeff', 0.0)}",
        f"actor_rollout_ref.actor.clip_ratio={actor.get('clip_ratio', 0.2)}",
        f"actor_rollout_ref.actor.loss_agg_mode={actor.get('loss_agg_mode', 'token-mean')}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor.get('ppo_mini_batch_size', 64)}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor.get('ppo_micro_batch_size_per_gpu', 8)}",
        f"actor_rollout_ref.actor.ppo_epochs={actor.get('ppo_epochs', 1)}",
    ])
    
    rollout = config.get("rollout", {})
    cmd.extend([
        f"actor_rollout_ref.rollout.name={rollout.get('engine', 'vllm')}",
        f"actor_rollout_ref.rollout.n={rollout.get('n', 8)}",
        f"actor_rollout_ref.rollout.temperature={rollout.get('temperature', 0.7)}",
        f"actor_rollout_ref.rollout.top_p={rollout.get('top_p', 1.0)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout.get('tensor_model_parallel_size', 1)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout.get('gpu_memory_utilization', 0.6)}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.get('log_prob_micro_batch_size_per_gpu', 8)}",
        f"actor_rollout_ref.rollout.max_model_len={rollout.get('max_model_len', 32768)}",
    ])
    
    cmd.extend([
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    ])
    
    trainer = config.get("trainer", {})
    ckpt_dir = trainer.get("checkpoint_dir", "output/checkpoints")
    if not Path(ckpt_dir).is_absolute(): ckpt_dir = str(root / ckpt_dir)
    ensure_dir(ckpt_dir)
    n_gpus = args.n_gpus if args.n_gpus else trainer.get("n_gpus_per_node", 1)
    cmd.extend([
        f"trainer.total_epochs={trainer.get('total_epochs', 1)}",
        f"trainer.project_name={trainer.get('project_name', 'rubric-rewards')}",
        f"trainer.experiment_name={trainer.get('experiment_name', 'grpo-ml')}",
        f"trainer.save_freq={trainer.get('save_freq', 20)}",
        f"trainer.test_freq={trainer.get('test_freq', 10)}",
        f"trainer.val_before_train={str(trainer.get('val_before_train', True)).lower()}",
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.nnodes={trainer.get('nnodes', 1)}",
    ])
    loggers = trainer.get("logger", ["console", "wandb"])
    logger_str = "[" + ",".join(loggers) + "]"  # 生成 [console,wandb]
    cmd.append(f"trainer.logger={logger_str}")
    return cmd

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--n_gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--simple_reward", action="store_true")
    args = parser.parse_args()
    
    root = get_project_root()
    cfg_path = args.config if Path(args.config).is_absolute() else str(root / args.config)
    if not Path(cfg_path).exists():
        logger.error(f"配置不存在: {cfg_path}")
        sys.exit(1)
    
    config = load_config(cfg_path)
    logger.info(f"加载配置: {cfg_path}")
    
    train_f = config.get("data", {}).get("train_file", "data/ml_train.parquet")
    if not Path(train_f).is_absolute(): train_f = str(root / train_f)
    if not Path(train_f).exists():
        logger.warning(f"数据不存在: {train_f}")
        logger.info("请运行: python scripts/prepare_data.py")
    
    cmd = build_verl_command(config, args)
    
    env = os.environ.copy()
    grader = config.get("grader", {})
    env["GRADER_API_BASE"] = grader.get("api_base", "http://localhost:8000/v1")
    env["GRADER_MODEL_NAME"] = grader.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    env["GRADER_TIMEOUT"] = str(grader.get("timeout", 600))
    if args.gpus: env["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    logger.info(f"\n命令:\n{' '.join(cmd[:3])} \\\n    " + " \\\n    ".join(cmd[3:]))
    
    if args.dry_run:
        logger.info("\n[Dry Run] 未执行")
        return
    
    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info("训练完成!")
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("请安装verl: pip install verl")
        sys.exit(1)

if __name__ == "__main__":
    main()
