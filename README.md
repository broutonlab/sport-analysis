<b>FieldNET</b>

Using modified version of MobileNET_V1

<b>Install</b>
First, we need to install packages

```bash
poetry install
poetry shell
```
If there is an EnvCommandError, try to use python3.8.8 version for example
```bash
pyenv install 3.8.8
pyenv local 3.8.8
pyenv local
# 3.8.8
```
For the training process we need to install tensorboard
```bash
pip install tensorboard
```
<b>Train</b>
```bash
python3 train.py
```
<b>Inference</b>
```bash
python test_video.py --video_in <path_to_video> --path_out <path_to_save>
```