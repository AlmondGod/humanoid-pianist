# robopianist-rl

Reinforcement learning code for [RoboPianist](https://github.com/google-research/robopianist).

## Installation

Note: Make sure you are using the same conda environment you created for RoboPianist (see [here](https://github.com/google-research/robopianist/blob/main/README.md#installation)).

```bash
git clone https://github.com/almondgod/robopianist.git
cd robopianist
git submodule init && git submodule update
bash scripts/install_deps.sh

# Install FluidSynth and its dependencies
apt-get install -y fluidsynth

# Install ffmpeg
apt-get install -y ffmpeg

apt-get install python3-dev portaudio19-dev

# Install X11 dependencies (though you might not need this if you're running headless)
apt-get install -y libx11-6 libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libxcursor-dev

#egl
apt-get update && apt-get install -y libegl1

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
export MUJOCO_GL=egl

bash ./Miniconda3-latest-Linux-x86_64.sh

source /root/miniconda3/etc/profile.d/conda.sh

conda create -n pianist python=3.10

conda activate pianist

pip install -e ".[dev]"

cd ../robopianist-rl
pip install -r requirements.txt

pip install "numpy<2.0.0"
```

1. Install [JAX](https://github.com/google/jax#installation) 
 (for NVIDIA: `pip install -U "jax[cuda12]"`)
2. Run `pip install -r requirements.txt`

## Usage

We provide an example bash script to train a policy to play Twinkle Twinkle Little Star with the task parameters used in the paper.

```bash
bash run.sh
```

To look at all the possible command-line flags, run:

```bash
python train.py --help
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{zakka2023robopianist,
  author = {Zakka, Kevin and Smith, Laura and Gileadi, Nimrod and Howell, Taylor and Peng, Xue Bin and Singh, Sumeet and Tassa, Yuval and Florence, Pete and Zeng, Andy and Abbeel, Pieter},
  title = {{RoboPianist: A Benchmark for High-Dimensional Robot Control}},
  journal = {arXiv preprint arXiv:2304.04150},
  year = {2023},
}
```
