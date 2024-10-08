# SegPlant: Semantic Segmentation for plants

Segments branches and leaves of the plants.

# Environment setup

Environment setup using Conda:

Download and install conda using instructions [here](https://docs.anaconda.com/miniconda/miniconda-install/)

```bash
conda env create -f environment.yml
conda activate hug
```

# Installation

You can simply clone the repo and use it directly in the conda environment.
```bash
git clone https://github.com/rashikshrestha/SegPlant.git
cd SegPlant
```

# Download weights
Download models weights for UNet and SAM from [here](https://drive.google.com/drive/folders/180Rc6yY1YKudRl7xdgjf55N-l95lTUSX?usp=sharing), and put it inside `models` directory.

# Inference

To do segmentation of your own plant image, put the image inside `test_data/real_plants/images` and rename the image filename to `00008.png` and so on (i.e 00009.png, 00010.png) for multiple images.

Using UNet model:
```bash
python test_unet.py
```

Using SAM Model
```bash
python test_sam.py 0 # for branch segmentation
python test_sam.py 1 # for leaf segmentation
```

# Train

Download BlenderPlants dataset [here](https://drive.google.com/file/d/1ff38P_HFddPXqAhM7hVbqLdnea2HyGsM/view?usp=sharing)

Train UNet model:
```bash
python train_unet.py /path/to/dataset
```

Fine-tune SAM:
```bash
python train_sam.py /path/to/dataset 0 # for branch segmentation
python train_sam.py /path/to/dataset 1 # for leaf segmentation
```

# For advanced users

For more detail control over code, change the parameters listed at the top of `main` in each python script. The parameters are enclosed as:
```python
# -------- Parameters -------
# parameters here
# ---------------------------
```