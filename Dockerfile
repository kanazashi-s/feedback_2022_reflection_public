# 開発・デバッグ用イメージ作成
FROM nvcr.io/nvidia/pytorch:22.10-py3

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV TRANSFORMERS_CACHE /workspace/transformers_cache
ENV TOKENIZERS_PARALLELISM=false