# Styled Video Generator
The Styled Video Generator is a project that generates a video from a given set of short video clips. 
The generator can also apply style translations to the video (e.g. painting-like videos). 

## Prerequisites
- A CUDA compatible GPU is highly recommended
- Python 3
- Python packages [torch](https://pypi.org/project/torch/) | [torchvision](https://github.com/pytorch/vision) | [Pandas](https://pandas.pydata.org/) | [Pillow](https://python-pillow.org/) | [NumPy](https://numpy.org/)

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/williamhxy/styled_video_generator
```

- Initialize the submodule "pytorch-CycleGAN-and-pix2pix":
```bash
git submodule init
```

- Update the submodule "pytorch-CycleGAN-and-pix2pix":
```bash
git submodule update
```

### Initialize
Run the "initialize" script in the root and give the path to the short video clips

## Folder Structure
- data: Folder used to keep temporary data and metadata. 
- extern: All submodule repositories are here. 
- models: Pre-trained CycleGAN models for running the image translation. 
- results: The outputs from the algorithm.
- scripts: All supporting scripts are here. 

## Acknowledgments
- An overview and reference to the CycleGAN model can be found [here](https://junyanz.github.io/CycleGAN/).
- The CycleGAN model repository can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- The video to image processing in this project uses the [decord](https://github.com/dmlc/decord) code. 


