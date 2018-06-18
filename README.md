# cvi-poisson-lds
Implementation of Khan and Lin's [Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models
](https://arxiv.org/abs/1703.04265) for linear dynamical systems w/ poisson likelihood.


# Getting Started
Before running our code, create a [conda](https://conda.io/docs/user-guide/getting-started.html "Getting started with conda") environment using the file `environment.yml`. To do so, open a terminal and run:
```conda env create -f environment.yml```

Then, activate the created environment:
```source activate cvi-poisson-lds```

If you don't want to use conda, just make sure to use the libraries listed in `environment.yml` in their specified version. Additionally, the latest versions of Lasagna and Theano  are required to generate the data. They are not available on PyPI:

```bash
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Running the Code
```bash
python main.py
```
Hyperparameters can be tuned on lines 88 - 96. To use CVI for your own problem, write your own non-conjugate likelihood in _cvi\_helpers.non\_conjugate\_likelihood\(\)_.

## Plots
Plots are saved in CVI_PLDS/plots (see screenshot below). ![Fig](https://github.com/vmasrani/CVI_PLDS/blob/master/plots/cvi_results.png)

![Fig](https://github.com/vmasrani/CVI_PLDS/blob/master/plots/baselines.png)

![Fig](https://github.com/vmasrani/CVI_PLDS/blob/master/plots/elbo_loglik.png)

## Acknowledgements
We use the synthetic data code and the VILDS model from [Black Box Variational Inference for Linear Dynamical Systems](https://github.com/earcher/vilds) by Archer et al.

## Citing

If you use the code, please cite the original paper. Bibtex:
```
@article{khan2017conjugate,
  title={Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models},
  author={Khan, Mohammad Emtiyaz and Lin, Wu},
  journal={arXiv preprint arXiv:1703.04265},
  year={2017}
}
