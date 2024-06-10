FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg espeak-ng

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    transformers==4.39.2 diffusers==0.27.2 accelerate==0.28.0 omegaconf mmpose mmdet moviepy soundfile \
    https://github.com/camenduru/wheels/releases/download/tost/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl \
    munch pydub phonemizer einops einops-exts git+https://github.com/resemble-ai/monotonic_align.git nltk librosa

RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/TMElyralab/MuseTalk /content/MuseTalk
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/yl4579/StyleTTS2 /content/bucilianus

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco.pth -d /content/MuseTalk/models/dwpose -o dw-ll_ucoco.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.onnx -d /content/MuseTalk/models/dwpose -o dw-ll_ucoco_384.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ll_ucoco_384.pth -d /content/MuseTalk/models/dwpose -o dw-ll_ucoco_384.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-mm_ucoco.pth -d /content/MuseTalk/models/dwpose -o dw-mm_ucoco.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-ss_ucoco.pth -d /content/MuseTalk/models/dwpose -o dw-ss_ucoco.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/dw-tt_ucoco.pth -d /content/MuseTalk/models/dwpose -o dw-tt_ucoco.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-l_ucoco_256-95bb32f5_20230822.pth -d /content/MuseTalk/models/dwpose -o rtm-l_ucoco_256-95bb32f5_20230822.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-x_ucoco_256-05f5bcb7_20230822.pth -d /content/MuseTalk/models/dwpose -o rtm-x_ucoco_256-05f5bcb7_20230822.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/rtm-x_ucoco_384-f5b50679_20230822.pth -d /content/MuseTalk/models/dwpose -o rtm-x_ucoco_384-f5b50679_20230822.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/dwpose/yolox_l.onnx -d /content/MuseTalk/models/dwpose -o yolox_l.onnx && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth -d /content/MuseTalk/models/face-parse-bisent -o 79999_iter.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/resnet18-5c106cde.pth -d /content/MuseTalk/models/face-parse-bisent -o resnet18-5c106cde.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/raw/main/musetalk/musetalk.json -d /content/MuseTalk/models/musetalk -o musetalk.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/musetalk/pytorch_model.bin -d /content/MuseTalk/models/musetalk -o pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/raw/main/sd-vae-ft-mse/config.json -d /content/MuseTalk/models/sd-vae-ft-mse -o config.json && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.bin -d /content/MuseTalk/models/sd-vae-ft-mse -o diffusion_pytorch_model.bin && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.safetensors -d /content/MuseTalk/models/sd-vae-ft-mse -o diffusion_pytorch_model.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/MuseTalk/resolve/main/whisper/tiny.pt -d /content/MuseTalk/models/whisper -o tiny.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/bucilianus-1/resolve/main/epoch_2nd_00030.pth -d /content/bucilianus/Models/LJSpeech -o epoch_2nd_00030.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/bucilianus-1/raw/main/config.yml -d /content/bucilianus/Models/LJSpeech -o config_ft.yml

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py