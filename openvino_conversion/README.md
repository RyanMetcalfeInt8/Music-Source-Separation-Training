# OpenVINO model conversion steps

Clone the repo and prep python env:
```
python -m venv my_env
my_env\Scripts\activate
git clone https://github.com/RyanMetcalfeInt8/Music-Source-Separation-Training.git
cd Music-Source-Separation-Training
git checkout openvino_conversion
pip install -r requirements.txt
pip install onnx openvino onnxscript
```

## Convert MelBandRoformer Models using config (yaml file) and checkpoint:
```
cd openvino_conversion
python convert_to_openvino.py --model_type mel_band_roformer --config_path "dereverb_mel_band_roformer_anvuew.yaml" --start_check_point "dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt"
```

## Convert HTDemucs Models:
```
python convert_to_openvino.py --model_type htdemucs --config_path "config_musdb18_htdemucs.yaml" --start_check_point "955717e8-8726e21a.th"
```

## Convert Apollo Models:
```
python convert_to_openvino.py --model_type apollo --config_path "config_apollo.yaml" --start_check_point "pytorch_model.bin"
```

Note: This also works with this universal apollo model: https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt
But you need to use a `config_apollo.yml` that looks like this:
```
audio:
  chunk_size: 132300
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 0.0

model:
  sr: 44100
  win: 20
  feature_dim: 384
  layer: 6

training:
  instruments: ['restored', 'addition']
  target_instrument: 'restored'
  batch_size: 2
  num_steps: 1000
  num_epochs: 1000
  optimizer: 'prodigy'
  lr: 1.0
  patience: 2
  reduce_factor: 0.95
  coarse_loss_clip: true
  grad_clip: 0
  q: 0.95
  use_amp: true

augmentations:
  enable: false # enable or disable all augmentations (to fast disable if needed)

inference:
  batch_size: 4
  num_overlap: 4
```

(Basically, bumping up feature_dim to 384 as this is how it's defined in: https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/config_apollo_uni.yaml)

## Convert MDX23C Models:

```
python convert_to_openvino.py --model_type mdx23c --config_path "config_mdx23c.yaml" --start_check_point "drumsep_5stems_mdx23c_jarredou.ckpt"
```

For reference, here is the pip list for a successful conversion env (Python 3.10.11):
```
Package                     Version
--------------------------- -----------
absl-py                     2.3.0
accelerate                  1.8.1
aiohappyeyeballs            2.6.1
aiohttp                     3.12.13
aiosignal                   1.3.2
annotated-types             0.7.0
antlr4-python3-runtime      4.9.3
asteroid                    0.7.0
asteroid-filterbanks        0.4.0
async-timeout               5.0.1
attrs                       25.3.0
audiomentations             0.24.0
audioread                   3.0.1
auraloss                    0.4.0
beartype                    0.14.1
bitsandbytes                0.46.0
cached-property             2.0.1
certifi                     2025.6.15
cffi                        1.17.1
charset-normalizer          3.4.2
click                       8.2.1
cloudpickle                 3.1.1
colorama                    0.4.6
contourpy                   1.3.2
cycler                      0.12.1
Cython                      3.1.2
decorator                   5.2.1
demucs                      4.0.0
diffq                       0.2.4
dora_search                 0.1.12
efficientnet-pytorch        0.7.1
einops                      0.8.1
filelock                    3.18.0
fonttools                   4.58.4
frozenlist                  1.7.0
fsspec                      2025.5.1
gitdb                       4.0.12
GitPython                   3.1.44
huggingface-hub             0.33.1
hyper-connections           0.1.11
idna                        3.10
Jinja2                      3.1.6
joblib                      1.5.1
julius                      0.2.7
keyboard                    0.13.5
kiwisolver                  1.4.8
lameenc                     1.8.1
librosa                     0.9.2
lightning-utilities         0.14.3
llvmlite                    0.44.0
loralib                     0.1.2
MarkupSafe                  3.0.2
matplotlib                  3.10.3
mir_eval                    0.8.2
ml_collections              1.1.0
ml_dtypes                   0.5.1
mpmath                      1.3.0
multidict                   6.5.1
munch                       4.0.0
networkx                    3.4.2
numba                       0.61.2
numpy                       2.2.6
omegaconf                   2.2.3
onnx                        1.18.0
onnx-ir                     0.1.3
onnxscript                  0.3.1
openunmix                   1.3.0
openvino                    2025.2.0
openvino-telemetry          2025.1.0
packaging                   25.0
pandas                      2.3.0
pb-bss-eval                 0.0.2
pedalboard                  0.8.9
pesq                        0.0.4
pillow                      11.2.1
pip                         23.0.1
platformdirs                4.3.8
pooch                       1.8.2
pretrainedmodels            0.7.4
primePy                     1.3
prodigyopt                  1.1.2
propcache                   0.3.2
protobuf                    6.31.1
psutil                      7.0.0
PyAudio                     0.2.14
pycparser                   2.22
pydantic                    2.11.7
pydantic_core               2.33.2
pyparsing                   3.2.3
pystoi                      0.4.1
python-dateutil             2.9.0.post0
pytorch-lightning           2.5.2
pytorch-ranger              0.1.1
pytz                        2025.2
PyYAML                      6.0.2
regex                       2024.11.6
requests                    2.32.4
resampy                     0.4.3
retrying                    1.4.0
rotary-embedding-torch      0.3.5
safetensors                 0.5.3
sageattention               1.0.6
scikit-learn                1.7.0
scipy                       1.15.3
segmentation-models-pytorch 0.3.3
sentry-sdk                  2.31.0
setproctitle                1.3.6
setuptools                  65.5.0
six                         1.17.0
smmap                       5.0.2
soundfile                   0.13.1
spafe                       0.3.2
submitit                    1.5.3
sympy                       1.14.0
threadpoolctl               3.6.0
timm                        0.9.2
tokenizers                  0.15.2
torch                       2.7.1
torch-audiomentations       0.12.0
torch-log-wmse              0.3.0
torch-optimizer             0.1.0
torch_pitch_shift           1.2.5
torch-stoi                  0.2.3
torchaudio                  2.7.1
torchmetrics                0.11.4
torchseg                    0.0.1a1
torchvision                 0.22.1
tqdm                        4.67.1
transformers                4.35.2
treetable                   0.2.5
typing_extensions           4.14.0
typing-inspection           0.4.1
tzdata                      2025.2
urllib3                     2.5.0
wandb                       0.20.1
wxPython                    4.2.2
yarl                        1.20.1
```
