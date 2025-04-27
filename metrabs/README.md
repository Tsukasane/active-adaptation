Here we use `uv` to maintain the python environment.

```bash
cd metrabs
uv init .
uv venv --python 3.11
source .venv/bin/activate
```

Install the dependencies.

```bash

uv pip install 'tensorflow[and-cuda]' tensorflow-hub
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install asyncio charset-normalizer aiohttp opencv-python scipy joblib pandas ultralytics==8.3.109
uv pip install -U 'nvidia-cudnn-cu12==9.3.*'

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://bit.ly/metrabs_s -O metrabs_s.tar.gz

mkdir metrabs_dir && tar -xvf metrabs_s.tar.gz -C metrabs_dir
```

For pose estimation, we can run:

```bash
python video2control.py
# or
uv run video2control.py
```

We can also run:

```bash
python video2vis.py
# or
uv run video2vis.py
```
to visualize the pose estimation results by simple 3D visualization.
