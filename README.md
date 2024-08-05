<div align="center">
<h1>DepthLight</h1>

[**Raphaël Manus**](https://raphaelmanus.com)<sup>1, 2</sup> · [**Marc Christie**](https://people.irisa.fr/Marc.Christie/)<sup>1</sup> · [**Samuel Boivin**](https://www.linkedin.com/in/samuel-boivin-90951a3/)<sup>1</sup> · [**Pascal Guehl**](https://pascalguehl.jimdofree.com/)<sup>2</sup>

<sup>1</sup>Inria, IRISA, CNRS, Univ. Rennes&emsp;&emsp;<sup>2</sup>LIX, Ecole Polytechnique, IP Paris
<br>

<a href="https://arxiv.org/"><img src='https://img.shields.io/badge/arXiv-DepthLight-red' alt='Paper PDF'></a>
<a href='https://depthlight.github.io'><img src='https://img.shields.io/badge/Project_Page-DepthLight-green' alt='Project Page'></a>
</div>

## WIP page...

## Installation

Update `depthlight.yml` with correct CUDA version for PyTorch if needed.
Checkpoints download is done using wget.

```bash
git clone --recurse-submodules https://github.com/RaphaelManus/DepthLight
cd DepthLight

conda env create -f depthlight.yml
conda env create -f LANet.yml

conda activate depthlight
```

## Usage

We encourage users to structure their data directories in the following way:
```
- data
    '- input
    |   - img1.jpg
    |   - img2.png
    |   - ...
    '- ldr_pano
    |   - ...
    '- hdr_pano
    |   - ...
    '- usd
    |   - ...
```

### Running the script
```bash
python run.py \
  --input <path>
  --type <ldr_lfov | ldr_pano | hdr_pano> \
  --output <path> \
  --fov <fov> \
  --prompt <"optional prompt">
```
Options:
- `--input` or `-i`: Point it to an image directory storing all interested images
- `--type` or `-t` (optional): By default, expected input type is LDR LFOV images.
- `--output` or `-o` (optional): You can point it to a different directory than the input if needed.
- `--fov` or `-f` (optional): Specify the fov of the inputs, default is `90°`.
- `--prompt` or `-p` (optional): Specify a prompt to guide the generation, default is `"indoor"`.

For example:
```bash
python run.py -i ./data/input -t ldr_lfov -f 90 -p "indoor"
```
