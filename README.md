# predictive-forward-forward (Bio-plausible forward only alternative to train neural networks)
Implementation of the proposed predictive forward-forward learning algorithm for training a recurrent neural network.
This works combines predictive coding with recently proposed forward-forward algorithm

# Requirements
Our implementation is easy to follow, with basic linear-alegbra knowledge one can decode the working of the algorithm. Please look at algorithm-1 in our paper to better understand the overall working. In this framework we have provided simple modules; thus it is very convenient to extend our framework.

You will only need following basic packages
1. TensorFlow (version >= 2.0)
2. Numpy
3. Matplotlib
4. Python (version >=3.5)
5. [Ngc-learn][https://github.com/ago109/ngc-learn] (Some modules depends responsible for generating components are dependent on ngc-learn)

# Citations

Please cite our paper if it is helpful in your work:

```bibtex
@article{ororbia2023predictive,
  title={The Predictive Forward-Forward Algorithm},
  author={Ororbia, Alexander and Mali, Ankur},
  journal={arXiv preprint arXiv:2301.01452},
  year={2023}
}
```
