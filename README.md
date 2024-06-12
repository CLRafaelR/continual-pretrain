# continual-pretrain

このリポジトリは、LLM（大規模言語モデル）を継続事前学習するために作成しました。 
環境構築は[dev-llmリポジトリ](https://github.com/oriki101/dev-llm)を参考にしてください。

## パッケージのインストール

```bash
sudo apt update
sudo apt upgrade
sudo apt install libaio-dev
```

```bash
# abci/requirements.txtを読み込み、コメントとバージョン情報を除去し、poetry addを実行
# ただし，datasetsとfsspecは互いにバージョンが衝突するため除外
grep -vE '^\s*#' abci/requirements.txt | grep -vE 'datasets|fsspec' | sed 's/==.*//' | xargs -I {} poetry add {}

# バージョン衝突を回避しつつdatasetsとfsspecをインストール
poetry add fsspec==2024.3.1 datasets

# abci/requirements.txtからインストールしたtorchを削除
poetry remove torch

# abci/requirements.txtからインストールしたcu11バージョンのパッケージを削除
poetry remove torch nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-nccl-cu11 nvidia-nvtx-cu11

# CUDA 12.1対応のtorchとllama-cpp-pythonをインストール
poetry source add torch_cu121 --priority=explicit https://download.pytorch.org/whl/cu121
poetry source add llama_cpp_python_cu121 --priority=explicit https://abetlen.github.io/llama-cpp-python/whl/cu121

poetry add llama-cpp-python --source llama_cpp_python_cu121

# abci/requirements.txtにあるcu11バージョンのパッケージをcu12にしてインストール
poetry add torch --source torch_cu121 nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12

# FlashAtteintion-2のインストール
poetry add wheel
poetry run pip install flash-attn --no-build-isolation
```

## シングルノードでの学習

```bash
cd continual-pretrain
poetry run deepspeed src/train_deepspeed.py --train_config ./configs/train_configs/train_base.yaml
# see here to some more workaround: https://github.com/microsoft/DeepSpeed/issues/3961
# poetry run deepspeed --deepspeed --deepspeed_config ./configs/deepspeed/ds_config_zero2.json src/train_deepspeed.py --train_config ./configs/train_configs/train_base.yaml
```

## マルチノードでの学習

国立研究開発法人産業技術総合研究所によって構築・運用されているABCI（AI Bridging Cloud Infrastructure）を利用してマルチノード学習を行います。DeepSpeedはデフォルトでPDSH（Parallel Distributed Shell）を使って分散学習を行いますが、ABCI環境ではSSH経由で接続したノード上でPythonが読み込めないことによりエラーが発生する場合があります。そのため、シングルノード学習のように**`deepspeed`**コマンドを用いるには、ソースコードの修正が必要です。しかし、この作業は環境構築の過程で大きな手間となります。

そこで、Open MPIの**`mpirun`**コマンドを使用して分散学習を行う方法を採用します。これにより、複雑な設定を避けつつ、効率的なマルチノード学習が可能になります。実行コマンドは以下になります。

```bash
cd continual-pretrain
sh script/continual_pretrain_abci.sh
```
