# PLM - Fastformer for recommandation (Unofficial)


## Requirements

```bash
Python >= 3.9

# Core ML / DL
torch >= 2.0
transformers >= 4.38
tensorflow >= 2.15

# Utilities
scikit-learn >= 1.2
numpy >= 1.23
tqdm

## Preparing Data
Download data from MIND [link](https://msnews.github.io/) and decompress these files. You will get three files:
`MINDlarge_train`, `MINDlarge_dev`, and `MINDlarge_test`, then put them in the same folder, e.g., `./data/`. 

Script `data_generation.py` can help you to generate the data files which meet the need of SpeedyRec:
```
python data_generation.py --raw_data_path {./data or other path you save the decompressed data}
```
The processed data will be saved to `./data/speedy_data/`.

## Enrich data with LLM
Script `enrich_data.py` 
```
python enrich_data.py 
```
The processed data will be saved to `./data/speedy_data/`.

## Training 
```
python train.py \
--pretreained_model unilm \
--pretrained_model_path {path to ckpt of unilmv2} \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--world_size 4 \
--lr 1e-4 \
--pretrain_lr 8e-6 \
--warmup True \
--schedule_step 240000 \
--warmup_step 1000 \
--batch_size 42 \
--npratio 4 \
--beta_for_cache 0.002 \
--max_step_in_cache 2 \
--savename speedyrec_mind \
--news_dim 256
```
The model will be saved to `./saved_models/`, and validation will be conducted after each epoch.   
The default pretrained model is UniLM v2, and you can get it from [unilm repo](https://github.com/microsoft/unilm). For other pretrained model, you need set `--pretrained_model==others` and give a new path for `--pretrained_model_path`
(like `roberta-base` and `microsoft/deberta-base`, which needs to be supported by [transformers](https://huggingface.co/transformers/model_doc/auto.html?highlight=automodel#transformers.AutoModel)).



## Prediction
Run prediction using saved checkpoint:
```
python submission.py \
--pretrained_model_path {path to ckpt of unilmv2} \
--pretreained_model unilm \
--root_data_dir ./data/speedy_data/ \
--num_hidden_layers 8 \
--load_ckpt_name {path to your saved model} \
--batch_size 256 \
--news_dim 256
```
It will creates a zip file:`predciton.zip`, which can be submitted to the leaderboard of MIND directly.  

