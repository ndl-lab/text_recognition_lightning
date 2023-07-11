

# text_recognition_lightning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)


## Description

NDLOCR(ver.2)用の画像中のテキストを認識するモジュールのリポジトリです。

本プログラムは、全文検索用途のテキスト化のために開発した[ver.1](https://github.com/ndl-lab/ndlocr_cli/tree/ver.1)に対して、視覚障害者等の読み上げ用途にも利用できるよう、国立国会図書館が外部委託して追加開発したプログラムです（委託業者：株式会社モルフォAIソリューションズ）。

事業の詳細については、[令和4年度NDLOCR追加開発事業及び同事業成果に対する改善作業](https://lab.ndl.go.jp/data_set/r4ocr/r4_software/)をご覧ください。

本プログラムは、国立国会図書館がCC BY 4.0ライセンスで公開するものです。詳細については LICENSEをご覧ください。



## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ndl-lab/text_recognition_lightning
cd text_recognition_lightning

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```


```bash
# eval
python src/eval.py ckpt_path=logs/your/model/ckpt/path datamodule.dataset.pred=[/your/xml/data/directory1,/your/xml/data/directory2]

# xml
python src/eval.py task=xml ckpt_path=logs/your/model/ckpt/path datamodule.dataset.pred=[/your/xml/data/directory1,/your/xml/data/directory2]

# render(visualize)
python src/eval.py task=render ckpt_path=logs/your/model/ckpt/path datamodule.dataset.pred=[/your/xml/data/directory1,/your/xml/data/directory2]
