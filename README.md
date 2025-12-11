# SFMambaNet implementation

This paper focuses on pruning correspondences from an initial set of correspondences with a low inlier ratio.

If you find this project useful, please cite:

```
 @ to do
```

## Requirements

Please use Python 3.10, opencv-python (4.7.0.72), Pytorch (>= 2.1.1+cu118), and mamba-ssm (1.2.0.post1). Other dependencies should be easily installed through pip or conda.

## Example scripts

### Run the demo

For a quick start, clone the repo and download the pretrained model.

```bash
git clone https://github.com/Kirito14IT/SFMambaNet
```

### Generate training and testing data

First download YFCC100M dataset.

```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
```

Download SUN3D testing (1.1G) and training (31G) dataset if you need.

```bash
bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```

Then generate matches for YFCC100M and SUN3D (only testing) with SIFT.

```bash
cd ../dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```

Generate SUN3D training data if you need by following the same procedure and uncommenting corresponding lines in `sun3d.py`.

### Train model on YFCC100M or SUN3D

After generating dataset for YFCC100M, run the tranining script.

```bash
cd ./core 
python main.py
```

You can change the default settings for network structure and training process in `./core/config.py`.

### Train with your own local feature or data

The provided models are trained using SIFT. You had better retrain the model if you want to use CGR-Net with your own local feature, such as RootSIFT, SuperPoint and etc.

You can follow the provided example scirpts in `./dump_match` to generate dataset for your own local feature or data.

## Acknowledgement

This code is borrowed from [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet) 和 [MatchMamba](https://github.com/Mrwyb/MatchMamba). If using the part of code related to data generation, testing and evaluation, please cite these papers.

```
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={5845--5854},
  year={2019}
}
@inproceedings{zhao2021progressive,
  title={Progressive correspondence pruning by consensus learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6464--6473},
  year={2021}
}
@article{wu2025matchmamba,
  title={MatchMamba: Correspondence Pruning via Selective State Space Model},
  author={Wu, Yubin and Li, Xiaojie and Chen, Hao and Yang, Changcai and Wei, Lifang and Chen, Riqing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

