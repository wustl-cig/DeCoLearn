# DeCoLearn: Deformation-Compensated Learning

This is the official repository of [DeCoLearn: Deformation-Compensated Learning for Image Reconstruction without Ground Truth](https://arxiv.org/abs/2107.05533).

![](./img/GIF.gif)

## 0. Abstract
Deep neural networks for medical image reconstruction are traditionally trained using high-quality ground-truth images as training targets. Recent work on Noise2Noise (N2N) has shown the potential of using multiple noisy measurements of the same object as an alternative to having a ground-truth. However, existing N2N-based methods are not suitable for learning from the measurements of an object undergoing nonrigid deformation. This paper addresses this issue by proposing the deformation-compensated learning (DeCoLearn) method for training deep reconstruction networks by compensating for object deformations. A key component of DeCoLearn is a deep registration module, which is jointly trained with the deep reconstruction network without any ground-truth supervision. We validate DeCoLearn on both simulated and experimentally collected magnetic resonance imaging (MRI) data and show that it significantly improves imaging quality.

## 1. Supplementary Materials

Here, we provide [a supplementary document](./supplemental_documents.pdf) showing (a) an illustration of simulated sampling masks, (b) validation with additional levels of deformation, (c) validation with additional sub-sampling rates, (d) an illustration of the influence of the trade-off parameter \gamma, and (e) validation on MRI measurements simulated using complex-value ground-truth images.

## 2. Code

The code will be uploaded soon.

## 3. Citation

```
@article{gan2021deformation,
  title={Deformation-Compensated Learning for Image Reconstruction without Ground Truth},
  author={Gan, Weijie and Sun, Yu and Eldeniz, Cihat and Liu, Jiaming and An, Hongyu and Kamilov, Ulugbek S},
  journal={arXiv preprint arXiv:2107.05533},
  year={2021}
}
```