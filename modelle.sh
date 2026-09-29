#!/bin/bash
# Laedt die MuseTalk-Gewichte (ohne Anmeldung) nach models/
set -e
M=models
mkdir -p $M/musetalk $M/musetalkV15 $M/dwpose $M/face-parse-bisent $M/sd-vae $M/whisper
hf() { huggingface-cli download "$@" >/dev/null; }
hf TMElyralab/MuseTalk --local-dir $M --include "musetalk/musetalk.json" "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
hf stabilityai/sd-vae-ft-mse --local-dir $M/sd-vae --include "config.json" "diffusion_pytorch_model.bin"
hf openai/whisper-tiny --local-dir $M/whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
hf yzd-v/DWPose --local-dir $M/dwpose --include "dw-ll_ucoco_384.pth"
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $M/face-parse-bisent/79999_iter.pth
curl -sL https://download.pytorch.org/models/resnet18-5c106cde.pth -o $M/face-parse-bisent/resnet18-5c106cde.pth
du -sh $M
