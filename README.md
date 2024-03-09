# Conformer RNN-T ASR

This directory contains sample implementations of training and evaluation pipelines for a Conformer RNN-T ASR model.

## Setup
### Install PyTorch and TorchAudio nightly or from source
To install fameous Pytorch libraray, follow the directions at <https://pytorch.org/>.

To build TorchAudio from source, refer to the [contributing guidelines](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md).

### Install additional dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training

[`train.py`](./train.py) trains an Conformer-meduim(15M parametes) and RNN-T model (30.2M parameters, 121MB) on CommonVoice using PyTorch Lightning. Note that the script expects users to have the following:
- Access to GPU nodes for training.
- Full CommonVoice dataset.(In my thesis, only Farsi Section of Dataset is used)
- SentencePiece model to be used to encode targets; the model can be generated using [`train_spm.py`](./train_spm.py).
- File (--global_stats_path) that contains training set feature statistics; this file can be generated using [`global_stats.py`](./global_stats.py).

Sample SLURM command:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 4 --ntasks-per-node=8 python train.py --exp-dir ./experiments --dataset-path ./commonvoice-fa/ --global-stats-path ./global_stats.json --sp-model-path ./spm_unigram_1023.model --epochs 160
```

### Evaluation and Inference

[`eval.py`](./eval.py) evaluates a trained Conformer RNN-T model on CommonVoice.

Sample SLURM command:
```
srun python eval.py --checkpoint-path ./experiments/checkpoints/epoch=159.ckpt --librispeech-path ./commonvoice/ --sp-model-path ./spm_unigram_1023.model --use-cuda
```