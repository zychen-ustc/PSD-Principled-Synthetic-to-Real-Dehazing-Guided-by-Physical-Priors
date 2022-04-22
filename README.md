# Principled S2R Dehazing

This repository contains the official implementation for PSD Framework introduced in the following paper:

[**PSD: Principled Synthetic to Real Dehazing Guided by Physical Priors**](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.pdf)
<br>
[Zeyuan Chen](https://zychen-ustc.github.io/), Yangchao Wang, [Yang Yang](https://cfm.uestc.edu.cn/~yangyang/), [Dong Liu](http://staff.ustc.edu.cn/~dongeliu/)
<br>
CVPR 2021 (Oral)

### Citation

If you find our work useful in your research, please cite:

```
@InProceedings{Chen_2021_CVPR,
    author    = {Chen, Zeyuan and Wang, Yangchao and Yang, Yang and Liu, Dong},
    title     = {PSD: Principled Synthetic-to-Real Dehazing Guided by Physical Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7180-7189}
}
```

### Environment
- Python 3.6
- Pytorch 1.3.0

## Pre-trained Model

Model|File size|Download
:-:|:-:|:-:
PSD-MSBDN|126M|[Google Drive](https://drive.google.com/file/d/1kHdjj8p_-CzGfcF0bGUpiTrPBUeONYVU/view?usp=sharing)
PSD-FFANET|24M|[Google Drive](https://drive.google.com/file/d/1sRlVJgCZck7y9yYrWRwJ61O75ikFMwg-/view?usp=sharing)
PSD-GCANET|9M|[Google Drive](https://drive.google.com/file/d/1M7fwAcBzsJ3RcBF6HW3x1MSpmX2NuMv6/view?usp=sharing)

百度网盘链接: https://pan.baidu.com/s/1M1RO5AZaYcZtckb-OzfXgw (提取码: ixcz)

In the paper, all the qualitative results and most visual comparisons are produced by **PSD-MSBDN** model.

## Testing 
```
python test.py
```
- Note that the test.py file is hard coded, and the default code is for the testing of PSD-FFANET model. If you want to test the other two models, you need to modify the code. See annotations in [test.py](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/blob/main/PSD/test.py) and it would only take seconds.
- If the program reports an error when going through A-Net, please make sure that your PyTorch version is 1.3.0. You could also solve the problem by resize the input of A-Net to 512×512 or delete A-Net (only for testing). See [issue #5](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/issues/5) for more information.
- If you want to evaluete metrics including NIQE and BRISQUE, you should use the official implementations in matlab.


## Train Custom Model by PSD
### Modify the network: 

As most existing dehazing models are end-to-end, you are supposed to modify the network to make it a physics-baesd one. 

To be specific, take [GCANet](https://github.com/cddlyf/GCANet) as an example. In its GCANet.py file, the [variable y in Line 96](https://github.com/cddlyf/GCANet/blob/23846ffa2ead27b5c2dd27c96498722385f216a7/GCANet.py#L96) is the final feature map. You should replace the final deconv layer by two branches for transmission maps and dehazing results, separately. The branch can be consisted of two simple convolutional layers. In addition, you should also add an A-Net to generate atmosphere light.

### Pre-Training: 

With the modified Network, you can do the pre-train phase with synthetic data. In our settings, we use OTS from [RESIDE dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) as the data for pre-training. 

In [main.py](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/blob/main/PSD/main.py), we present the pipeline and loss settings for the pre-training of PSD-FFANet, you can take it as an example and modify it to fit your own model. 

Based on our observations, the pre-train models usually have similar performance (sometimes suffer slight drops) on PSNR and SSIM compared with the original models.

### Fine-tuning: 

Start from a pre-trained model, you can fine-tune it with real-world data in an unsupervised manner. We use RTTS from [RESIDE dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) as our fine-tuning data. We also process all hazy images in RTTS by CLAHE for convenience. 

You can find **both RTTS and our pre-processed data in this [Link](https://pan.baidu.com/s/1_YJUObKmDxoncbr8WF8_5g) (code: wxty)**. Code for the fine-tuning of the three provided models is included in [finetune.py](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/blob/main/PSD/finetune.py).
