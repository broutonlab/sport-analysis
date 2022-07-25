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
Your input dataset need to be that structure:
```bash
data
-- raw
-- -- dir_1
-- -- -- *images format <num.jpg>* and result_<dir_name>.json file
-- -- dir_2
-- -- -- *images format <num.jpg>* and result_<dir_name>.json file
-- -- ...
```
JSON format:
```bash
{
  "0.jpg": {
    "11": [
      517,
      293
    ],
    "15": [
      456,
      215
    ],
    ...
```
Then you can train the network:
```bash
python3 train.py
```
<b>Inference</b>
You can build video from
```bash
python ./src/inference/video_from_images.py --path_to_images <path_to_images> --path_out <path_to_save>
```
Download example video or other video for test, or
```bash
python test_video.py --video_in <path_to_video> --path_out <path_to_save> --weights <path_to_weights>
```
And you'll get
