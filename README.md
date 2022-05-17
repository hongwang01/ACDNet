# Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction (IJCAI2022)
[Hong Wang](https://hongwang01.github.io/), Yuexiang Li, [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng), [Yefeng Zheng](https://sites.google.com/site/yefengzheng/)

[Arxiv&&SM](https://arxiv.org/pdf/2205.07471.pdf)

## Abstract
Inspired by the great success of deep neural networks, learning-based methods have gained promising performances for metal artifact reduction (MAR) in computed tomography (CT) images. However, most of the existing approaches put less emphasis on modelling and embedding the intrinsic prior knowledge underlying this specific MAR task into their network designs. Against this issue, we propose an adaptive convolutional dictionary network (ACDNet), which leverages both model-based and learning-based methods. Specifically, we explore the prior structures of metal artifacts, e.g., non-local repetitive streaking patterns, and encode them as an explicit weighted convolutional dictionary model. Then, a simple-yet-effective algorithm is carefully designed to solve the model. By unfolding every iterative substep of the proposed algorithm into a network module, we explicitly embed the prior structure into a deep network, i.e., a clear interpretability for the MAR task. Furthermore, our ACDNet can automatically learn the prior for artifact-free CT images via training data and adaptively adjust the representation kernels for each input CT image based on its content. Hence, our method inherits the clear interpretability of model-based methods and maintains the powerful representation ability of learning-based methods. Comprehensive experiments executed on synthetic and clinical datasets show the superiority of our ACDNet in terms of effectiveness and model generalization

## Dependicies

This repository is tested under the following system settings:

Python 3.6

Pytorch 1.4.0

CUDA 10.1

GPU NVIDIA Tesla V100-SMX2


## Benchmark Dataset

Please refer to [InDuDoNet](https://github.com/hongwang01/InDuDoNet)
 
## Training
```
python train.py --gpu_id 0  --data_path "data/train/" --batchSize 1 --batchnum 1 --log_dir "logs/" --model_dir "models/"
```
*Please note that for the demo, “batchnum=1, batchSize=1". Please change it according to your own training set.*

## Testing
```
python test.py  --gpu_id 0 --data_path "data/test/" --model_dir "models/ACDNet_latest.pt" --save_path "save_results/"
```

## Metric
[PSNR/SSIM](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation) 


## Citations

```
@inproceedings{wang2022ada,
  title={Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction},
  author={Wang, Hong and Li, Yuexiang and Meng, Deyu and Zheng, Yefeng},
  booktitle={The 31st International Joint Conference on Artificial Intelligence},
  year={2022},
  organization={IEEE}
}
```
## References

[1] Hong Wang, Yuexiang Li, Haimiao Zhang, Jiawei Chen, Kai Ma, Deyu Meng, and Yefeng Zheng. InDuDoNet: An interpretable dual domain network for CT metal artifact reduction. In International Conference on Medical Image Computing and ComputerAssisted Intervention, pages 107–118, 2021.

[2] Hong Wang, Yuexiang Li, Nanjun He, Kai Ma, Deyu Meng, and Yefeng Zheng. DICDNet: Deep interpretable convolutional dictionary network for metal
artifact reduction in CT images. IEEE Transactions on Medical Imaging, 2021.

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang9209@hotmail.com)
