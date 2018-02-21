#!/bin/sh
#SBATCH --time=1-12
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

source activate venv

python unsupervised.py \
--src_lang en \
--tgt_lang ru \
--src_emb /projects/tir1/users/gbhat/data/fasttext/wiki.en.vec \
--tgt_emb /projects/tir1/users/gbhat/data/fasttext/wiki.ru.vec \
--exp_path /projects/tir1/users/gbhat/work/muse_experiments
--cuda True