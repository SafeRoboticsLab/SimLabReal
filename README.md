# Generalization Guarantees for Safe RL with Shielding
#### Requirements
 - Python 3.5 or above
 - PyTorch 1.3
 - Gym 0.15
TODO

## Usage

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with MAML. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on HalfCheetah-Vel:
```
python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --config maml-halfcheetah-vel/config.json --policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz --meta-batch-size 20 --num-batches 10  --num-workers 8
```

## References
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

https://github.com/tristandeleu/pytorch-maml-rl
