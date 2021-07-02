---
title: Hot Papers 2021-07-02
date: 2021-07-03T07:45:53.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Focal Self-attention for Local-Global Interactions in Vision  Transformers

Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, Jianfeng Gao

- retweets: 2546, favorites: 259 (07/03/2021 07:45:53)

- links: [abs](https://arxiv.org/abs/2107.00641) | [pdf](https://arxiv.org/pdf/2107.00641)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently, Vision Transformer and its variants have shown great promise on various computer vision tasks. The ability of capturing short- and long-range visual dependencies through self-attention is arguably the main source for the success. But it also brings challenges due to quadratic computational overhead, especially for the high-resolution vision tasks (e.g., object detection). In this paper, we present focal self-attention, a new mechanism that incorporates both fine-grained local and coarse-grained global interactions. Using this new mechanism, each token attends the closest surrounding tokens at fine granularity but the tokens far away at coarse granularity, and thus can capture both short- and long-range visual dependencies efficiently and effectively. With focal self-attention, we propose a new variant of Vision Transformer models, called Focal Transformer, which achieves superior performance over the state-of-the-art vision Transformers on a range of public image classification and object detection benchmarks. In particular, our Focal Transformer models with a moderate size of 51.1M and a larger size of 89.8M achieve 83.5 and 83.8 Top-1 accuracy, respectively, on ImageNet classification at 224x224 resolution. Using Focal Transformers as the backbones, we obtain consistent and substantial improvements over the current state-of-the-art Swin Transformers for 6 different object detection methods trained with standard 1x and 3x schedules. Our largest Focal Transformer yields 58.7/58.9 box mAPs and 50.9/51.3 mask mAPs on COCO mini-val/test-dev, and 55.4 mIoU on ADE20K for semantic segmentation, creating new SoTA on three of the most challenging computer vision tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Focal Self-attention for Local-Global Interactions in<br>Vision Transformers<br>pdf: <a href="https://t.co/2mFN1OQzVG">https://t.co/2mFN1OQzVG</a><br><br>largest Focal Transformer yields 58.7/58.9 box mAPs and 50.9/51.3 mask mAPs on COCO mini-val/test-dev, and 55.4 mIoU on ADE20K for semantic segmentation <a href="https://t.co/ij7VYIbcQR">pic.twitter.com/ij7VYIbcQR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410764142680608768?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. CLIP-It! Language-Guided Video Summarization

Medhini Narasimhan, Anna Rohrbach, Trevor Darrell

- retweets: 723, favorites: 179 (07/03/2021 07:45:53)

- links: [abs](https://arxiv.org/abs/2107.00650) | [pdf](https://arxiv.org/pdf/2107.00650)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

A generic video summary is an abridged version of a video that conveys the whole story and features the most important scenes. Yet the importance of scenes in a video is often subjective, and users should have the option of customizing the summary by using natural language to specify what is important to them. Further, existing models for fully automatic generic summarization have not exploited available language models, which can serve as an effective prior for saliency. This work introduces CLIP-It, a single framework for addressing both generic and query-focused video summarization, typically approached separately in the literature. We propose a language-guided multimodal transformer that learns to score frames in a video based on their importance relative to one another and their correlation with a user-defined query (for query-focused summarization) or an automatically generated dense video caption (for generic video summarization). Our model can be extended to the unsupervised setting by training without ground-truth supervision. We outperform baselines and prior work by a significant margin on both standard video summarization datasets (TVSum and SumMe) and a query-focused video summarization dataset (QFVS). Particularly, we achieve large improvements in the transfer setting, attesting to our method's strong generalization capabilities.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CLIP-It! Language-Guided Video Summarization<br>pdf: <a href="https://t.co/xEKaeXw9qk">https://t.co/xEKaeXw9qk</a><br>abs: <a href="https://t.co/Qges7bG37d">https://t.co/Qges7bG37d</a><br>project page: <a href="https://t.co/6786FcFeg3">https://t.co/6786FcFeg3</a> <a href="https://t.co/m5vyTAhcgJ">pic.twitter.com/m5vyTAhcgJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410760332432261122?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Want to create short summaries of long videos? Check out our work, “CLIP-It! Language-Guided Video Summarization”. Joint work with Anna Rohrbach and <a href="https://twitter.com/trevordarrell?ref_src=twsrc%5Etfw">@trevordarrell</a> <br><br>Paper: <a href="https://t.co/OrY6IFceHO">https://t.co/OrY6IFceHO</a><br>Project Page: <a href="https://t.co/3hcTfn7mcK">https://t.co/3hcTfn7mcK</a> <br>Results: <a href="https://t.co/q6RcLV6NP3">https://t.co/q6RcLV6NP3</a></p>&mdash; Medhini Narasimhan (@medhini_n) <a href="https://twitter.com/medhini_n/status/1410771917875732485?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Variational Diffusion Models

Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho

- retweets: 704, favorites: 188 (07/03/2021 07:45:53)

- links: [abs](https://arxiv.org/abs/2107.00630) | [pdf](https://arxiv.org/pdf/2107.00630)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Diffusion-based generative models have demonstrated a capacity for perceptually impressive synthesis, but can they also be great likelihood-based models? We answer this in the affirmative, and introduce a family of diffusion-based generative models that obtain state-of-the-art likelihoods on standard image density estimation benchmarks. Unlike other diffusion-based models, our method allows for efficient optimization of the noise schedule jointly with the rest of the model. We show that the variational lower bound (VLB) simplifies to a remarkably short expression in terms of the signal-to-noise ratio of the diffused data, thereby improving our theoretical understanding of this model class. Using this insight, we prove an equivalence between several models proposed in the literature. In addition, we show that the continuous-time VLB is invariant to the noise schedule, except for the signal-to-noise ratio at its endpoints. This enables us to learn a noise schedule that minimizes the variance of the resulting VLB estimator, leading to faster optimization. Combining these advances with architectural improvements, we obtain state-of-the-art likelihoods on image density estimation benchmarks, outperforming autoregressive models that have dominated these benchmarks for many years, with often significantly faster optimization. In addition, we show how to turn the model into a bits-back compression scheme, and demonstrate lossless compression rates close to the theoretical optimum.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Variational Diffusion Models<br><br>Obtains SotA likelihoods on image density estimation benchmarks, outperforming autoregressive models that have dominated these benchmarks for many years, with often significantly faster optimization.<a href="https://t.co/wjHVJuVatj">https://t.co/wjHVJuVatj</a> <a href="https://t.co/opWtNVUzFc">pic.twitter.com/opWtNVUzFc</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1410761239962157060?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Variational Diffusion Models<br>pdf: <a href="https://t.co/VvSYLZ3Cyz">https://t.co/VvSYLZ3Cyz</a><br>abs: <a href="https://t.co/XLsUDN5VJQ">https://t.co/XLsUDN5VJQ</a><br><br>sota likelihoods on image density estimation benchmarks, outperforming autoregressive models that have dominated these benchmarks for many years, with often significantly faster optimization <a href="https://t.co/MfrzrDq52y">pic.twitter.com/MfrzrDq52y</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410761766175481860?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. CSWin Transformer: A General Vision Transformer Backbone with  Cross-Shaped Windows

Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo

- retweets: 506, favorites: 64 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00652) | [pdf](https://arxiv.org/pdf/2107.00652)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks. A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. We provide a detailed mathematical analysis of the effect of the stripe width and vary the stripe width for different layers of the Transformer network which achieves strong modeling capability while limiting the computation cost. We also introduce Locally-enhanced Positional Encoding (LePE), which handles the local positional information better than existing encoding schemes. LePE naturally supports arbitrary input resolutions, and is thus especially effective and friendly for downstream tasks. Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks. Specifically, it achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, and 51.7 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and +2.0 respectively under the similar FLOPs setting. By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and state-of-the-art segmentation performance on ADE20K with 55.2 mIoU. The code and models will be available at https://github.com/microsoft/CSWin-Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows<br>pdf: <a href="https://t.co/6KuG5MRGPM">https://t.co/6KuG5MRGPM</a><br><br>85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task <a href="https://t.co/pHZdSI0RBa">pic.twitter.com/pHZdSI0RBa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410773728905121796?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Global Filter Networks for Image Classification

Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou

- retweets: 360, favorites: 72 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00645) | [pdf](https://arxiv.org/pdf/2107.00645)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent advances in self-attention and pure multi-layer perceptrons (MLP) models for vision have shown great potential in achieving promising performance with fewer inductive biases. These models are generally based on learning interaction among spatial locations from raw data. The complexity of self-attention and MLP grows quadratically as the image size increases, which makes these models hard to scale up when high-resolution features are required. In this paper, we present the Global Filter Network (GFNet), a conceptually simple yet computationally efficient architecture, that learns long-term spatial dependencies in the frequency domain with log-linear complexity. Our architecture replaces the self-attention layer in vision transformers with three key operations: a 2D discrete Fourier transform, an element-wise multiplication between frequency-domain features and learnable global filters, and a 2D inverse Fourier transform. We exhibit favorable accuracy/complexity trade-offs of our models on both ImageNet and downstream tasks. Our results demonstrate that GFNet can be a very competitive alternative to transformer-style models and CNNs in efficiency, generalization ability and robustness. Code is available at https://github.com/raoyongming/GFNet

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Global Filter Networks for Image Classification<br>pdf: <a href="https://t.co/dIeGFqtllM">https://t.co/dIeGFqtllM</a><br>abs: <a href="https://t.co/48uTA872An">https://t.co/48uTA872An</a><br>project page: <a href="https://t.co/LyAIupelxl">https://t.co/LyAIupelxl</a><br>github: <a href="https://t.co/0BcTRgg4pJ">https://t.co/0BcTRgg4pJ</a> <a href="https://t.co/aVds2AwhCC">pic.twitter.com/aVds2AwhCC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410853648146567175?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. AutoFormer: Searching Transformers for Visual Recognition

Minghao Chen, Houwen Peng, Jianlong Fu, Haibin Ling

- retweets: 324, favorites: 53 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00651) | [pdf](https://arxiv.org/pdf/2107.00651)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, pure transformer-based models have shown great potentials for vision tasks such as image classification and detection. However, the design of transformer networks is challenging. It has been observed that the depth, embedding dimension, and number of heads can largely affect the performance of vision transformers. Previous models configure these dimensions based upon manual crafting. In this work, we propose a new one-shot architecture search framework, namely AutoFormer, dedicated to vision transformer search. AutoFormer entangles the weights of different blocks in the same layers during supernet training. Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch. Besides, the searched models, which we refer to AutoFormers, surpass the recent state-of-the-arts such as ViT and DeiT. In particular, AutoFormer-tiny/small/base achieve 74.7%/81.7%/82.4% top-1 accuracy on ImageNet with 5.7M/22.9M/53.7M parameters, respectively. Lastly, we verify the transferability of AutoFormer by providing the performance on downstream benchmarks and distillation experiments. Code and models are available at https://github.com/microsoft/AutoML.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AutoFormer: Searching Transformers for Visual Recognition<br>pdf: <a href="https://t.co/BfcLzNpd2I">https://t.co/BfcLzNpd2I</a><br>abs: <a href="https://t.co/pFSpFDrBOZ">https://t.co/pFSpFDrBOZ</a><br>github: <a href="https://t.co/SBeDmRhmET">https://t.co/SBeDmRhmET</a><br><br>AutoFormer-tiny/small/base achieve 74.7%/81.7%/82.4% top-1 accuracy on ImageNet with 5.7M/22.9M/53.7M parameters, respectively <a href="https://t.co/kC8DykvoiM">pic.twitter.com/kC8DykvoiM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410761103626293250?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under  Data Augmentation

Nicklas Hansen, Hao Su, Xiaolong Wang

- retweets: 196, favorites: 49 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00644) | [pdf](https://arxiv.org/pdf/2107.00644)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

While agents trained by Reinforcement Learning (RL) can solve increasingly challenging tasks directly from visual observations, generalizing learned skills to novel environments remains very challenging. Extensive use of data augmentation is a promising technique for improving generalization in RL, but it is often found to decrease sample efficiency and can even lead to divergence. In this paper, we investigate causes of instability when using data augmentation in common off-policy RL algorithms. We identify two problems, both rooted in high-variance Q-targets. Based on our findings, we propose a simple yet effective technique for stabilizing this class of algorithms under augmentation. We perform extensive empirical evaluation of image-based RL using both ConvNets and Vision Transformers (ViT) on a family of benchmarks based on DeepMind Control Suite, as well as in robotic manipulation tasks. Our method greatly improves stability and sample efficiency of ConvNets under augmentation, and achieves generalization results competitive with state-of-the-art methods for image-based RL. We further show that our method scales to RL with ViT-based architectures, and that data augmentation may be especially important in this setting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation<br>pdf: <a href="https://t.co/bc9OKZIyuJ">https://t.co/bc9OKZIyuJ</a><br>abs: <a href="https://t.co/72lkOCSkqi">https://t.co/72lkOCSkqi</a><br>project page: <a href="https://t.co/eJWIlifY9x">https://t.co/eJWIlifY9x</a> <a href="https://t.co/P9Xo1oGcbs">pic.twitter.com/P9Xo1oGcbs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410768129752510464?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Mandoline: Model Evaluation under Distribution Shift

Mayee Chen, Karan Goel, Nimit Sohoni, Fait Poms, Kayvon Fatahalian, Christopher Ré

- retweets: 182, favorites: 23 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00643) | [pdf](https://arxiv.org/pdf/2107.00643)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine learning models are often deployed in different settings than they were trained and validated on, posing a challenge to practitioners who wish to predict how well the deployed model will perform on a target distribution. If an unlabeled sample from the target distribution is available, along with a labeled sample from a possibly different source distribution, standard approaches such as importance weighting can be applied to estimate performance on the target. However, importance weighting struggles when the source and target distributions have non-overlapping support or are high-dimensional. Taking inspiration from fields such as epidemiology and polling, we develop Mandoline, a new evaluation framework that mitigates these issues. Our key insight is that practitioners may have prior knowledge about the ways in which the distribution shifts, which we can use to better guide the importance weighting procedure. Specifically, users write simple "slicing functions" - noisy, potentially correlated binary functions intended to capture possible axes of distribution shift - to compute reweighted performance estimates. We further describe a density ratio estimation framework for the slices and show how its estimation error scales with slice quality and dataset size. Empirical validation on NLP and vision tasks shows that \name can estimate performance on the target distribution up to $3\times$ more accurately compared to standard baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper appearing in <a href="https://twitter.com/hashtag/ICML2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2021</a>! Mandoline: Model Evaluation under Distribution Shift:<br><br>Paper: <a href="https://t.co/JmvzPJtRtK">https://t.co/JmvzPJtRtK</a><br>Code: <a href="https://t.co/pcB61ObekF">https://t.co/pcB61ObekF</a><br><br>work done w/ equal contribution from <a href="https://twitter.com/krandiash?ref_src=twsrc%5Etfw">@krandiash</a> and <a href="https://twitter.com/nimit_sohoni?ref_src=twsrc%5Etfw">@nimit_sohoni</a> , as well as <a href="https://twitter.com/faitpoms?ref_src=twsrc%5Etfw">@faitpoms</a>, <a href="https://twitter.com/kayvonf?ref_src=twsrc%5Etfw">@kayvonf</a>, and <a href="https://twitter.com/HazyResearch?ref_src=twsrc%5Etfw">@HazyResearch</a> 1/6 <a href="https://t.co/87ir29r5ti">pic.twitter.com/87ir29r5ti</a></p>&mdash; Mayee Chen (@MayeeChen) <a href="https://twitter.com/MayeeChen/status/1411043189125955586?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Simple Training Strategies and Model Scaling for Object Detection

Xianzhi Du, Barret Zoph, Wei-Chih Hung, Tsung-Yi Lin

- retweets: 72, favorites: 26 (07/03/2021 07:45:54)

- links: [abs](https://arxiv.org/abs/2107.00057) | [pdf](https://arxiv.org/pdf/2107.00057)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

The speed-accuracy Pareto curve of object detection systems have advanced through a combination of better model architectures, training and inference methods. In this paper, we methodically evaluate a variety of these techniques to understand where most of the improvements in modern detection systems come from. We benchmark these improvements on the vanilla ResNet-FPN backbone with RetinaNet and RCNN detectors. The vanilla detectors are improved by 7.7% in accuracy while being 30% faster in speed. We further provide simple scaling strategies to generate family of models that form two Pareto curves, named RetinaNet-RS and Cascade RCNN-RS. These simple rescaled detectors explore the speed-accuracy trade-off between the one-stage RetinaNet detectors and two-stage RCNN detectors. Our largest Cascade RCNN-RS models achieve 52.9% AP with a ResNet152-FPN backbone and 53.6% with a SpineNet143L backbone. Finally, we show the ResNet architecture, with three minor architectural changes, outperforms EfficientNet as the backbone for object detection and instance segmentation systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Simple Training Strategies and Model Scaling for Object Detection<br>pdf: <a href="https://t.co/0frwnBjkUp">https://t.co/0frwnBjkUp</a><br>abs: <a href="https://t.co/B6dugtbSkC">https://t.co/B6dugtbSkC</a><br><br>largest Cascade RCNN-RS models achieve 52.9% AP with a ResNet152-FPN backbone and 53.6% with a SpineNet143L backbone <a href="https://t.co/QK1pBCQEVG">pic.twitter.com/QK1pBCQEVG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410775752610660353?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Secure Quantized Training for Deep Learning

Marcel Keller, Ke Sun

- retweets: 56, favorites: 33 (07/03/2021 07:45:55)

- links: [abs](https://arxiv.org/abs/2107.00501) | [pdf](https://arxiv.org/pdf/2107.00501)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We have implemented training of neural networks in secure multi-party computation (MPC) using quantization commonly used in the said setting. To the best of our knowledge, we are the first to present an MNIST classifier purely trained in MPC that comes within 0.2 percent of the accuracy of the same convolutional neural network trained via plaintext computation. More concretely, we have trained a network with two convolution and two dense layers to 99.2% accuracy in 25 epochs. This took 3.5 hours in our MPC implementation (under one hour for 99% accuracy).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have trained an MNIST model purely in MPC to 99% accuracy in under an hour: <a href="https://t.co/11sRRjX7pt">https://t.co/11sRRjX7pt</a></p>&mdash; Marcel Keller 🏳️‍🌈 (@mkskeller) <a href="https://twitter.com/mkskeller/status/1410912953700737026?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Attention Bottlenecks for Multimodal Fusion

Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun

- retweets: 56, favorites: 27 (07/03/2021 07:45:55)

- links: [abs](https://arxiv.org/abs/2107.00135) | [pdf](https://arxiv.org/pdf/2107.00135)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Humans perceive the world by concurrently processing and fusing high-dimensional inputs from multiple modalities such as vision and audio. Machine perception models, in stark contrast, are typically modality-specific and optimised for unimodal benchmarks, and hence late-stage fusion of final representations or predictions from each modality (`late-fusion') is still a dominant paradigm for multimodal video classification. Instead, we introduce a novel transformer based architecture that uses `fusion bottlenecks' for modality fusion at multiple layers. Compared to traditional pairwise self-attention, our model forces information between different modalities to pass through a small number of bottleneck latents, requiring the model to collate and condense the most relevant information in each modality and only share what is necessary. We find that such a strategy improves fusion performance, at the same time reducing computational cost. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple audio-visual classification benchmarks including Audioset, Epic-Kitchens and VGGSound. All code and models will be released.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Attention Bottlenecks for Multimodal Fusion<br>pdf: <a href="https://t.co/mHXLVsXQPZ">https://t.co/mHXLVsXQPZ</a><br>abs: <a href="https://t.co/7DFP9VNLa8">https://t.co/7DFP9VNLa8</a><br><br>achieve sota results on multiple audio-visual classification benchmarks including Audioset, Epic-Kitchens and VGGSound <a href="https://t.co/In4KoIggYJ">pic.twitter.com/In4KoIggYJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410763304209948674?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. All That's 'Human' Is Not Gold: Evaluating Human Evaluation of Generated  Text

Elizabeth Clark, Tal August, Sofia Serrano, Nikita Haduong, Suchin Gururangan, Noah A. Smith

- retweets: 36, favorites: 33 (07/03/2021 07:45:55)

- links: [abs](https://arxiv.org/abs/2107.00061) | [pdf](https://arxiv.org/pdf/2107.00061)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Human evaluations are typically considered the gold standard in natural language generation, but as models' fluency improves, how well can evaluators detect and judge machine-generated text? We run a study assessing non-experts' ability to distinguish between human- and machine-authored text (GPT2 and GPT3) in three domains (stories, news articles, and recipes). We find that, without training, evaluators distinguished between GPT3- and human-authored text at random chance level. We explore three approaches for quickly training evaluators to better identify GPT3-authored text (detailed instructions, annotated examples, and paired examples) and find that while evaluators' accuracy improved up to 55%, it did not significantly improve across the three domains. Given the inconsistent results across text domains and the often contradictory reasons evaluators gave for their judgments, we examine the role untrained human evaluations play in NLG evaluation and provide recommendations to NLG researchers for improving human evaluations of text generated from state-of-the-art models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">All That’s ‘Human’ Is Not Gold: Evaluating Human Evaluation of Generated Text<br>pdf: <a href="https://t.co/hcssxuD1xe">https://t.co/hcssxuD1xe</a><br><br>approaches for training evaluators to better identify GPT3-authored text, evaluators’ accuracy improved up to 55%, it did not significantly improve across the three domains <a href="https://t.co/hwn8KHEk4S">pic.twitter.com/hwn8KHEk4S</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410766347336830977?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Uncertainty-Aware Learning for Improvements in Image Quality of the  Canada-France-Hawaii Telescope

Sankalp Gilda, Stark C. Draper, Sebastien Fabbro, William Mahoney, Simon Prunet, Kanoa Withington, Matthew Wilson, Yuan-Sen Ting, Andrew Sheinis

- retweets: 29, favorites: 36 (07/03/2021 07:45:55)

- links: [abs](https://arxiv.org/abs/2107.00048) | [pdf](https://arxiv.org/pdf/2107.00048)
- [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We leverage state-of-the-art machine learning methods and a decade's worth of archival data from the Canada-France-Hawaii Telescope (CFHT) to predict observatory image quality (IQ) from environmental conditions and observatory operating parameters. Specifically, we develop accurate and interpretable models of the complex dependence between data features and observed IQ for CFHT's wide field camera, MegaCam. Our contributions are several-fold. First, we collect, collate and reprocess several disparate data sets gathered by CFHT scientists. Second, we predict probability distribution functions (PDFs) of IQ, and achieve a mean absolute error of $\sim0.07''$ for the predicted medians. Third, we explore data-driven actuation of the 12 dome ``vents'', installed in 2013-14 to accelerate the flushing of hot air from the dome. We leverage epistemic and aleatoric uncertainties in conjunction with probabilistic generative modeling to identify candidate vent adjustments that are in-distribution (ID) and, for the optimal configuration for each ID sample, we predict the reduction in required observing time to achieve a fixed SNR. On average, the reduction is $\sim15\%$. Finally, we rank sensor data features by Shapley values to identify the most predictive variables for each observation. Our long-term goal is to construct reliable and real-time models that can forecast optimal observatory operating parameters for optimization of IQ. Such forecasts can then be fed into scheduling protocols and predictive maintenance routines. We anticipate that such approaches will become standard in automating observatory operations and maintenance by the time CFHT's successor, the Maunakea Spectroscopic Explorer (MSE), is installed in the next decade.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Machine Learning + Astronomy ==&gt; Magic :) Check out our latest paper: <a href="https://t.co/4YwE89EkXk">https://t.co/4YwE89EkXk</a>. We demonstrate a way to increase efficiency of ground-based astronomical observatories by about 15 %!<a href="https://twitter.com/OpenAcademics?ref_src=twsrc%5Etfw">@OpenAcademics</a> <a href="https://twitter.com/AcademicChatter?ref_src=twsrc%5Etfw">@AcademicChatter</a><br> <a href="https://twitter.com/hashtag/phdfriend?src=hash&amp;ref_src=twsrc%5Etfw">#phdfriend</a> <a href="https://twitter.com/hashtag/phdchat?src=hash&amp;ref_src=twsrc%5Etfw">#phdchat</a> <a href="https://twitter.com/hashtag/phdvoice?src=hash&amp;ref_src=twsrc%5Etfw">#phdvoice</a> <a href="https://twitter.com/hashtag/academictwitter?src=hash&amp;ref_src=twsrc%5Etfw">#academictwitter</a></p>&mdash; Sankalp Gilda (@wutwutman1) <a href="https://twitter.com/wutwutman1/status/1410776607690838022?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Audiovisual Singing Voice Separation

Bochen Li, Yuxuan Wang, Zhiyao Duan

- retweets: 30, favorites: 35 (07/03/2021 07:45:55)

- links: [abs](https://arxiv.org/abs/2107.00231) | [pdf](https://arxiv.org/pdf/2107.00231)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Separating a song into vocal and accompaniment components is an active research topic, and recent years witnessed an increased performance from supervised training using deep learning techniques. We propose to apply the visual information corresponding to the singers' vocal activities to further improve the quality of the separated vocal signals. The video frontend model takes the input of mouth movement and fuses it into the feature embeddings of an audio-based separation framework. To facilitate the network to learn audiovisual correlation of singing activities, we add extra vocal signals irrelevant to the mouth movement to the audio mixture during training. We create two audiovisual singing performance datasets for training and evaluation, respectively, one curated from audition recordings on the Internet, and the other recorded in house. The proposed method outperforms audio-based methods in terms of separation quality on most test recordings. This advantage is especially pronounced when there are backing vocals in the accompaniment, which poses a great challenge for audio-only methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Audiovisual Singing Voice Separation<br>pdf: <a href="https://t.co/op6o32TiPr">https://t.co/op6o32TiPr</a><br>abs: <a href="https://t.co/aB3JxEbi0I">https://t.co/aB3JxEbi0I</a><br><br>proposed an audiovisual approach to address the solo singing voice separation problem by analyzing both the auditory signal and mouth movement of the solo singer in the visual signal <a href="https://t.co/qS5Q13KIqr">pic.twitter.com/qS5Q13KIqr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1410800112427847681?ref_src=twsrc%5Etfw">July 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



