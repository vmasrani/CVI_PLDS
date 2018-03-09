Implementation of Khan and Lin's [Conjugate-Computation Variational Inference : Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models
](https://arxiv.org/abs/1703.04265) for linear dynamical systems w/ poisson likelihood.

To install, first get the requirements from requirements.txt, then install the latest versions of Lasagna and Theano (not available on PyPI):

```bash
pip install -r requirements.txt
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

Run with:
```python
python cvi.py
```
