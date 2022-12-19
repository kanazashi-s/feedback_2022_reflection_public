import subprocess
import sys
from pathlib import Path
from typing import List
import re


def upload_model(model_name: str, model_path: Path):
    """Upload model to Kaggle Dataset

    Args:
        model_name (str): model name
        model_path (str): model path

    Note:
        You need to set up Kaggle API.
        You need to put your Kaggle API token in /root/.kaggle/kaggle.json.
    """
    create_dataset_metadata(model_name, model_path)
    run_shell(["kaggle", "datasets", "create", "-p", str(model_path)])


def create_dataset_metadata(model_name: str, model_path: Path):
    """Create dataset-metadata.json

    Args:
        model_name (str): model name
        model_path (str): model path
    """

    if re.fullmatch(r'[a-zA-Z0-9-]+', model_name) is None:
        raise Exception(f'Invalid model_name: {model_name}')

    with open(model_path / "dataset-metadata.json", mode="w") as f:
        f.write(f'{{"id": "zashio/{model_name}", "title": "{model_name}", "licenses": [{{"name": "CC0-1.0"}}]}}')


def run_shell(cmd: List[str]):
    """Run shell command
    シェルコマンドを実行しつつ、結果をリアルタイムに標準出力へとパイプしてくれる関数
    https://qiita.com/waterada/items/65209ae58130ec669578
    """

    print(f'cmd: {cmd}')
    with subprocess.Popen(cmd, encoding='UTF-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        for line in p.stdout:
            sys.stdout.write(line)
        p.wait()
        for line in p.stderr:
            sys.stdout.write(line)
        print(f'return: {p.returncode}')
        if p.returncode:
            raise Exception(f'Error! {cmd}')


if __name__ == "__main__":
    model_name = "deberta-v3-base"
    model_path = Path("output", "single", "deberta_v3_base")
    create_dataset_metadata(model_name, model_path)
