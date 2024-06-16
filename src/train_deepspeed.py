import argparse
import os
import warnings
from typing import Dict, List
import deepspeed
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
import gc
from utils import seed_everything

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 分散処理させるときにwarningが出たため


def preprocess_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, List[int]]:
    """
    与えられたテキストをトークナイズし、ラベルを追加する前処理関数。
    ラベルを追加する理由は、AutoModelForCausalLMの入力を'label'としなければならないため

    Args:
        examples (Dict[str, List[str]]): 前処理するテキストの例。キーはテキストのフィールド名（例えば 'text'）。
        tokenizer (PreTrainedTokenizer): 使用するトークナイザーのインスタンス。
        max_length (int): トークナイズ後の最大シーケンス長。

    Returns:
        Dict[str, List[int]]: トークナイズされたテキストと対応するラベルを含む辞書。
    """
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs["labels"] = inputs.input_ids.copy()
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        "-p",
        type=str,
        default="./configs/train_configs/train_base.yaml",
        help="モデルパラメータのコンフィグ。yamlファイル",
    )
    parser.add_argument(
        # 単一ノードで実行されているプロセスの一意のローカル ID
        "--local_rank",
        "-l",
        type=int,
        default=0,
        help="GPUのランク",
    )
    args = parser.parse_args()
    local_rank = args.local_rank
    # コンフィグ読み込み
    config = OmegaConf.load(args.train_config)

    # distributed learning
    deepspeed.init_distributed()

    # seedの設定
    seed_everything(config.seed)

    # モデルの定義
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model,
        torch_dtype=torch.float16,
        use_cache=config.model.use_cache,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer,
        add_eos_token=True,  # EOSの追加を指示 defaultはFalse
    )

    # データセットの読み込み
    dataset = load_dataset(
        path=config.dataset.path,
        name=config.dataset.subset,
        split=config.dataset.split,
        cache_dir=config.dataset.cache_dir,
    )

    # データをモデルに入力できるように変換
    dataset = dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, config.model.max_length
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=32,
    )

    dataset = dataset.train_test_split(test_size=0.2)

    # 学習
    training_args = TrainingArguments(**config.train)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        # data_collator=data_collator,
    )

    # trainer.train()
    with torch.autocast("cuda"):
        trainer.train()

    del dataset
    del trainer

    gc.collect()
    deepspeed.runtime.utils.empty_cache()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
