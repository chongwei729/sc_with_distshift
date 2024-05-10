# sc_with_distshift

This repo contains the source code for paper [Selective Classification Under Distribution Shifts](https://arxiv.org/abs/2405.05160).

If you find our work interesting or helpful, please consider a generous citation of our paper:

```
@article{liang2024selective,
  title={Selective Classification Under Distribution Shifts},
  author={Hengyue Liang and Le Peng and Ju Sun},
  journal={arXiv preprint 	arXiv:2405.05160},
  year={2024}
}
```

## Paper Summary

## How to use this code
This code provides essential funtionalities to

    1. Collect info needed to evaluate the selective classification (SC) performance of a pretrained model

    2. A script to plot risk-coverage (RC) curves for all scores considered in our paper.

#### Collecting data using pretrained models
```collect_feature_logits.py``` provides an example of using EVA model (pretrained for ImageNet-1K classification task from ```timm```) to collect necessary data from ```ImageNet-1k (2012)``` validation set to plot the RC curve.

Reader may use this script as a pointer and experiment with dataset(e.g., shifted version)/models that they are interested.


#### Plotting the RC curve
```plot_rc_curve.py``` includes plotting RC curves using all scores tested in our paper. The example given in ```Demon-Vis``` demonstrates the RC curves drawn by ```EVA``` model and ```ImageNet-1k (2012)``` validation set.

To test the SC performance on shifted data, one can simply collect the necessary data by modifying ```collect_feature_logits.py``` and combine the files (collected data) with similar names into one single file and pass them into ```plot_rc_curve.py```.

## Environment Setup

We tested this demo code with conda env:

```bash
python==3.12
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

but i guess newer versions should also work perfectly.


Then, install the dependencies by:
```bash
pip install -r requirement.txt
```

## Acknowledgement
