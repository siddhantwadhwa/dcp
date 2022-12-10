# Deep Closest Point

## Prerequisites 
PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5py

tqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Introduction

## Code organization
This repository is a fork of https://github.com/WangYueFt/dcp. The overall code organization can be
summarized as follows: 
- `main.py` - The main training code that is overloaded to also perform validation.
- `model.py` - The model definition for DCP, further separated into the `encoder` and `head` definition.
- `data.py` - The dataset definition, along with newly implemented augmentation procedure for the data.
- Along with miscellaneous utility and visualization scripts for development 

## Augmentation Procedure

Augmentation procedures that simulates occlusions and subtractive noise are added in the `data.py` section
of the code. Each augmentation procedures are implemented in a separate class as follows
1. `RandomRemoveTwoQuadrant` - Divides the point cloud into two quadrants, remove a single portion from one quadrant.
2. `RandomRemoveFourQuadrant` - Divides the point cloud into four quadrants, remove a single portion from one quadrant.
3. `RandomRemoveEightQuadrant` - Divides the point cloud into eight quadrants, remove a single portion form one quadrant.
4. `RandomRemove` - Removes the point with a uniform random probability across all points.

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

## License
MIT License
