# MsDD

## File Structure

- **train_MD.py**: Run this file to train and test our MsDD.
- **config**: Configuration parameters of MsDD.
- **definition**: Initial definitions of all modules.
- **utils**: Preprocess datasets.
- **tools**: Statistical analysis of datasets.

## Details

- **entropy.py** in `utils` provides the process for calculating graph entropy in the propagation network with each time unit.
- **test_entropy.ipynb** in `tools` demonstrates the process of constructing the multi-stage propagation sub-graph. This file also provides some sample constructions of truth and misinformation.
- **MD_Dynamic_net.py** records the dynamic analysis part.

## Citation

Please cite our paper:
@article{hao2024multi,
  title={Multi-stage dynamic disinformation detection with graph entropy guidance},
  author={Hao, Xiaorong and Liu, Bo and Yang, Xinyan and Sun, Xiangguo and Meng, Qing and Cao, Jiuxin},
  journal={World Wide Web},
  volume={27},
  number={2},
  pages={8},
  year={2024},
  publisher={Springer}
}
