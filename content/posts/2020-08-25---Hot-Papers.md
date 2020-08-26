---
title: Hot Papers 2020-08-25
date: 2020-08-26T09:12:07.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement

William Peebles, John Peebles, Jun-Yan Zhu, Alexei Efros, Antonio Torralba

- retweets: 51, favorites: 216 (08/26/2020 09:12:07)

- links: [abs](https://arxiv.org/abs/2008.10599) | [pdf](https://arxiv.org/pdf/2008.10599)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Existing disentanglement methods for deep generative models rely on hand-picked priors and complex encoder-based architectures. In this paper, we propose the Hessian Penalty, a simple regularization term that encourages the Hessian of a generative model with respect to its input to be diagonal. We introduce a model-agnostic, unbiased stochastic approximation of this term based on Hutchinson's estimator to compute it efficiently during training. Our method can be applied to a wide range of deep generators with just a few lines of code. We show that training with the Hessian Penalty often causes axis-aligned disentanglement to emerge in latent space when applied to ProGAN on several datasets. Additionally, we use our regularization term to identify interpretable directions in BigGAN's latent space in an unsupervised fashion. Finally, we provide empirical evidence that the Hessian Penalty encourages substantial shrinkage when applied to over-parameterized latent spaces.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement<br>pdf: <a href="https://t.co/hDferVdWG4">https://t.co/hDferVdWG4</a><br>abs: <a href="https://t.co/MrD6jIsTQ0">https://t.co/MrD6jIsTQ0</a><br>project page: <a href="https://t.co/XMMU5OTedj">https://t.co/XMMU5OTedj</a><br>github: <a href="https://t.co/YlgT4C5UzV">https://t.co/YlgT4C5UzV</a> <a href="https://t.co/VMTBkBs5UG">pic.twitter.com/VMTBkBs5UG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1298083597547638784?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Semantic View Synthesis

Hsin-Ping Huang, Hung-Yu Tseng, Hsin-Ying Lee, Jia-Bin Huang

- retweets: 22, favorites: 74 (08/26/2020 09:12:07)

- links: [abs](https://arxiv.org/abs/2008.10598) | [pdf](https://arxiv.org/pdf/2008.10598)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We tackle a new problem of semantic view synthesis -- generating free-viewpoint rendering of a synthesized scene using a semantic label map as input. We build upon recent advances in semantic image synthesis and view synthesis for handling photographic image content generation and view extrapolation. Direct application of existing image/view synthesis methods, however, results in severe ghosting/blurry artifacts. To address the drawbacks, we propose a two-step approach. First, we focus on synthesizing the color and depth of the visible surface of the 3D scene. We then use the synthesized color and depth to impose explicit constraints on the multiple-plane image (MPI) representation prediction process. Our method produces sharp contents at the original view and geometrically consistent renderings across novel viewpoints. The experiments on numerous indoor and outdoor images show favorable results against several strong baselines and validate the effectiveness of our approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Semantic View Synthesis<br>pdf: <a href="https://t.co/AlCJZezD8d">https://t.co/AlCJZezD8d</a><br>abs: <a href="https://t.co/bL0nnp9oIh">https://t.co/bL0nnp9oIh</a><br>project page: <a href="https://t.co/gthjh8tf6R">https://t.co/gthjh8tf6R</a><br>github: <a href="https://t.co/oUR10YksOw">https://t.co/oUR10YksOw</a> <a href="https://t.co/iw9XlRMPXc">pic.twitter.com/iw9XlRMPXc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1298081936800448512?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Generate High Resolution Images With Generative Variational Autoencoder

Abhinav Sagar

- retweets: 14, favorites: 68 (08/26/2020 09:12:07)

- links: [abs](https://arxiv.org/abs/2008.10399) | [pdf](https://arxiv.org/pdf/2008.10399)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this work, we present a novel neural network to generate high resolution images. We replace the decoder of VAE with a discriminator while using the encoder as it is. The encoder uses data from a normal distribution while the generator from a gaussian distribution. The combination from both is given to a discriminator which tells whether the generated images are correct or not. We evaluate our network on 3 different datasets: MNIST, LSUN and CelebA-HQ dataset. Our network beats the previous state of the art using MMD, SSIM, log likelihood, reconstruction error, ELBO and KL divergence as the evaluation metrics while generating much sharper images. This work is potentially very exciting as we are able to combine the advantages of generative models and inference models in a principled bayesian manner.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generate High Resolution Images With Generative Variational Autoencoder<br>pdf: <a href="https://t.co/krzKlKCphQ">https://t.co/krzKlKCphQ</a><br>abs: <a href="https://t.co/Yq7GRebhVg">https://t.co/Yq7GRebhVg</a><br>github: <a href="https://t.co/pSsAmgD30B">https://t.co/pSsAmgD30B</a> <a href="https://t.co/vaen2HrF85">pic.twitter.com/vaen2HrF85</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1298095908308099074?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. CA-GAN: Weakly Supervised Color Aware GAN for Controllable Makeup  Transfer

Robin Kips, Pietro Gori, Matthieu Perrot, Isabelle Bloch

- retweets: 17, favorites: 53 (08/26/2020 09:12:08)

- links: [abs](https://arxiv.org/abs/2008.10298) | [pdf](https://arxiv.org/pdf/2008.10298)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

While existing makeup style transfer models perform an image synthesis whose results cannot be explicitly controlled, the ability to modify makeup color continuously is a desirable property for virtual try-on applications. We propose a new formulation for the makeup style transfer task, with the objective to learn a color controllable makeup style synthesis. We introduce CA-GAN, a generative model that learns to modify the color of specific objects (e.g. lips or eyes) in the image to an arbitrary target color while preserving background. Since color labels are rare and costly to acquire, our method leverages weakly supervised learning for conditional GANs. This enables to learn a controllable synthesis of complex objects, and only requires a weak proxy of the image attribute that we desire to modify. Finally, we present for the first time a quantitative analysis of makeup style transfer and color control performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CA-GAN: Weakly Supervised Color Aware GAN for Controllable Makeup Transfer<br>pdf: <a href="https://t.co/koLLPww9nM">https://t.co/koLLPww9nM</a><br>abs: <a href="https://t.co/f7EJiC44iU">https://t.co/f7EJiC44iU</a><br>project page: <a href="https://t.co/xWiA4QVV1t">https://t.co/xWiA4QVV1t</a> <a href="https://t.co/iiWlqUxS8K">pic.twitter.com/iiWlqUxS8K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1298108237108187136?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Self-Supervised Learning for Large-Scale Unsupervised Image Clustering

Evgenii Zheltonozhskii, Chaim Baskin, Alex M. Bronstein, Avi Mendelson

- retweets: 15, favorites: 42 (08/26/2020 09:12:08)

- links: [abs](https://arxiv.org/abs/2008.10312) | [pdf](https://arxiv.org/pdf/2008.10312)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Unsupervised learning has always been appealing to machine learning researchers and practitioners, allowing them to avoid an expensive and complicated process of labeling the data. However, unsupervised learning of complex data is challenging, and even the best approaches show much weaker performance than their supervised counterparts. Self-supervised deep learning has become a strong instrument for representation learning in computer vision. However, those methods have not been evaluated in a fully unsupervised setting. In this paper, we propose a simple scheme for unsupervised classification based on self-supervised representations. We evaluate the proposed approach with several recent self-supervised methods showing that it achieves competitive results for ImageNet classification (39% accuracy on ImageNet with 1000 clusters and 46% with overclustering). We suggest adding the unsupervised evaluation to a set of standard benchmarks for self-supervised learning. The code is available at https://github.com/Randl/kmeans_selfsuper

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised learning is really hot now. In our new paper (<a href="https://t.co/n8pyI5CS3V">https://t.co/n8pyI5CS3V</a>) with <a href="https://twitter.com/ChaimBaskin?ref_src=twsrc%5Etfw">@ChaimBaskin</a> Alex Bronstein and Avi Mendelson we study self-supervised learning in unsupervised clustering settings. The code is available at <a href="https://t.co/dGekFTW962">https://t.co/dGekFTW962</a> 1/n</p>&mdash; Evgenii Zheltonozhskii (@evgeniyzhe) <a href="https://twitter.com/evgeniyzhe/status/1298136113639497729?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Hierarchical Style-based Networks for Motion Synthesis

Jingwei Xu, Huazhe Xu, Bingbing Ni, Xiaokang Yang, Xiaolong Wang, Trevor Darrell

- retweets: 9, favorites: 44 (08/26/2020 09:12:08)

- links: [abs](https://arxiv.org/abs/2008.10162) | [pdf](https://arxiv.org/pdf/2008.10162)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generating diverse and natural human motion is one of the long-standing goals for creating intelligent characters in the animated world. In this paper, we propose a self-supervised method for generating long-range, diverse and plausible behaviors to achieve a specific goal location. Our proposed method learns to model the motion of human by decomposing a long-range generation task in a hierarchical manner. Given the starting and ending states, a memory bank is used to retrieve motion references as source material for short-range clip generation. We first propose to explicitly disentangle the provided motion material into style and content counterparts via bi-linear transformation modelling, where diverse synthesis is achieved by free-form combination of these two components. The short-range clips are then connected to form a long-range motion sequence. Without ground truth annotation, we propose a parameterized bi-directional interpolation scheme to guarantee the physical validity and visual naturalness of generated results. On large-scale skeleton dataset, we show that the proposed method is able to synthesise long-range, diverse and plausible motion, which is also generalizable to unseen motion data during testing. Moreover, we demonstrate the generated sequences are useful as subgoals for actual physical execution in the animated world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical Style-based Networks for Motion Synthesis<br>pdf: <a href="https://t.co/uaPtSFuA2F">https://t.co/uaPtSFuA2F</a><br>abs: <a href="https://t.co/DVaBZEdXGo">https://t.co/DVaBZEdXGo</a><br>project page: <a href="https://t.co/wIuyxXo37E">https://t.co/wIuyxXo37E</a> <a href="https://t.co/Q5DPBSTgHV">pic.twitter.com/Q5DPBSTgHV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1298075154157895681?ref_src=twsrc%5Etfw">August 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



