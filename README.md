<p align="center">
  This is the official implementation of the paper
</p>

<div id="user-content-toc" display="inline">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Diverse Score Distillation</h1>
    </summary>
  </ul>

<p align="center">
    <a class="active text-decoration-none" href="https://yanbo-xu.netlify.app/">Yanbo Xu</a><sup> 1</sup>,  &nbsp;
    <a class="active text-decoration-none" href="https://scholar.google.com/citations?user=HtNfeKYAAAAJ&hl=en">Jayanth Srinivasa</a><sup> 2</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://scholar.google.com/citations?user=NIv_aeQAAAAJ&hl=en">Gaowen Liu</a><sup> 2</sup>, &nbsp;
    <a class="active text-decoration-none" href="https://shubhtuls.github.io/">Shubham Tulsiani</a><sup> 1</sup>
</p>
<p align="center">
  <span class="author-block"><sup>1 </sup>Carnegie Mellon University,</span>&nbsp;
  <span class="author-block"><sup>2 </sup>Cisco Research</span>&nbsp;

</p>

<p align="center">
 <a href="https://arxiv.org/abs/2412.06780">
    <img src="https://img.shields.io/badge/arXiv-2412.06780-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://billyxyb.github.io/Diverse-Score-Distillation/">
    <img src="https://img.shields.io/badge/dsd-project page-b78601.svg">
  </a>
</p>

<p align="center">
  <img alt="sample generation" src="./docs/teaser_captioned.gif" width="100%">
<!-- <img alt="sample generation" src="https://lukoianov.com/static/media/A_DSLR_photo_of_a_freshly_baked_round_loaf_of_sourdough_bread.8bfaaad1.gif" width="70%">
<br/> -->
</p>

> **Abstract:** Score distillation of 2D diffusion models has proven to be a powerful mechanism to guide 3D optimization, for example enabling text-based 3D generation or single-view reconstruction. A common limitation of existing score distillation formulations, however, is that the outputs of the (mode-seeking) optimization are limited in diversity despite the underlying diffusion model being capable of generating diverse samples. In this work, inspired by the sampling process in denoising diffusion, we propose a score formulation that guides the optimization to follow generation paths defined by random initial seeds, thus ensuring diversity. We then present an approximation to adopt this formulation for scenarios where the optimization may not precisely follow the generation paths (e.g. a 3D representation whose renderings evolve in a co-dependent manner). We showcase the applications of our `Diverse Score Distillation' (DSD) formulation across tasks such as 2D optimization, text-based 3D inference, and single-view reconstruction. We also empirically validate DSD against prior score distillation formulations and show that it significantly improves sample diversity while preserving fidelity.

## Installation

This project is based on [Threestudio](https://github.com/threestudio-project/threestudio). Please see the installation gudie from [Threestudio](https://github.com/threestudio-project/threestudio).

## Run 3D Generation

If more than 40GB of VRAM is avaliable, the results from full resolution (512) can be run as below:

```sh
python launch.py --config configs/dsd.yaml --train --gpu 0 system.prompt_processor.prompt="a toy robot" --seed 0

python launch.py --config configs/dsd.yaml --train --gpu 0 system.prompt_processor.prompt="pumpkin head zombie, skinny, highly detailed, photorealistic" --seed 0

python launch.py --config configs/dsd.yaml --train --gpu 0 system.prompt_processor.prompt="Mini Garden, highly detailed, 8K, HD." --seed 0
```
The results will be saved to `outputs/diverse-score-distillation/`.

## Generating Diverse 3D Shapes

To generate diverse samples, just change the seeds:

```sh
python launch.py --config configs/dsd.yaml --train --gpu 0 system.prompt_processor.prompt="a toy robot" --seed 0

python launch.py --config configs/dsd.yaml --train --gpu 0 system.prompt_processor.prompt="a toy robot" --seed 1000
```

If VRM is limitted, we recommend reducing the rendering resolution by running command (tested on GPUs with 24GB of VRAM):
```sh
python launch.py --config configs/dsd_low_res.yaml --train --gpu 0 system.prompt_processor.prompt="a toy robot" --seed 0
```

## Citing

If you find our project useful, please consider citing it:

```
@article{xu2024diversescoredistillation,
  title={Diverse Score Distillation}, 
  author={Yanbo Xu and Jayanth Srinivasa and Gaowen Liu and Shubham Tulsiani},
  journal={arXiv preprint arXiv:2412.06780},
  year={2024}
}

```