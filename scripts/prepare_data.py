"""数据准备脚本"""
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from utils import convert_to_verl_format, ensure_dir, setup_logger, get_project_root
from typing import cast, Dict, Any

logger = setup_logger("prepare_data")

def download_and_convert(subset: str = "ml", output_dir: str = "data", max_train: int = -1, max_test: int = -1):
    logger.info(f"下载: facebook/research-plan-gen, subset={subset}")
    ds = load_dataset("facebook/research-plan-gen", subset)
    logger.info(f"训练: {len(ds['train'])} | 测试: {len(ds['test'])}")
    
    out = ensure_dir(output_dir)
    
    train = ds['train'].select(range(min(max_train, len(ds['train'])))) if max_train > 0 else ds['train']
    test = ds['test'].select(range(min(max_test, len(ds['test'])))) if max_test > 0 else ds['test']
    
    train_recs = [convert_to_verl_format(cast(Dict[str, Any], item), f"research_plan_gen_{subset}") for item in train]
    test_recs = [convert_to_verl_format(cast(Dict[str, Any], item), f"research_plan_gen_{subset}") for item in test]
    
    pd.DataFrame(train_recs).to_parquet(out / f"{subset}_train.parquet", index=False)
    pd.DataFrame(test_recs).to_parquet(out / f"{subset}_test.parquet", index=False)
    
    logger.info(f"已保存: {out / f'{subset}_train.parquet'} ({len(train_recs)})")
    logger.info(f"已保存: {out / f'{subset}_test.parquet'} ({len(test_recs)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="ml", choices=["ml", "arxiv", "pubmed"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = str(get_project_root() / "data")
    
    download_and_convert(args.subset, args.output_dir, args.max_train_samples, args.max_test_samples)

if __name__ == "__main__":
    main()
