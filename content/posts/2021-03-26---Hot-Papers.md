---
title: Hot Papers 2021-03-26
date: 2021-03-27T09:23:34.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-26"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-26"
socialImage: "/media/flying-marine.jpg"

---

# 1. KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs

Christian Reiser, Songyou Peng, Yiyi Liao, Andreas Geiger

- retweets: 4622, favorites: 336 (03/27/2021 09:23:34)

- links: [abs](https://arxiv.org/abs/2103.13744) | [pdf](https://arxiv.org/pdf/2103.13744)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

NeRF synthesizes novel views of a scene with unprecedented quality by fitting a neural radiance field to RGB images. However, NeRF requires querying a deep Multi-Layer Perceptron (MLP) millions of times, leading to slow rendering times, even on modern GPUs. In this paper, we demonstrate that significant speed-ups are possible by utilizing thousands of tiny MLPs instead of one single large MLP. In our setting, each individual MLP only needs to represent parts of the scene, thus smaller and faster-to-evaluate MLPs can be used. By combining this divide-and-conquer strategy with further optimizations, rendering is accelerated by two orders of magnitude compared to the original NeRF model without incurring high storage costs. Further, using teacher-student distillation for training, we show that this speed-up can be achieved without sacrificing visual quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs<br>pdf: <a href="https://t.co/pF27zlTvz7">https://t.co/pF27zlTvz7</a><br>abs: <a href="https://t.co/qfVYGZrakR">https://t.co/qfVYGZrakR</a> <a href="https://t.co/ZihuaoIA4J">pic.twitter.com/ZihuaoIA4J</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375252238449410051?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. PlenOctrees for Real-time Rendering of Neural Radiance Fields

Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, Angjoo Kanazawa

- retweets: 1525, favorites: 218 (03/27/2021 09:23:34)

- links: [abs](https://arxiv.org/abs/2103.14024) | [pdf](https://arxiv.org/pdf/2103.14024)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We introduce a method to render Neural Radiance Fields (NeRFs) in real time using PlenOctrees, an octree-based 3D representation which supports view-dependent effects. Our method can render 800x800 images at more than 150 FPS, which is over 3000 times faster than conventional NeRFs. We do so without sacrificing quality while preserving the ability of NeRFs to perform free-viewpoint rendering of scenes with arbitrary geometry and view-dependent effects. Real-time performance is achieved by pre-tabulating the NeRF into a PlenOctree. In order to preserve view-dependent effects such as specularities, we factorize the appearance via closed-form spherical basis functions. Specifically, we show that it is possible to train NeRFs to predict a spherical harmonic representation of radiance, removing the viewing direction as an input to the neural network. Furthermore, we show that PlenOctrees can be directly optimized to further minimize the reconstruction loss, which leads to equal or better quality compared to competing methods. Moreover, this octree optimization step can be used to reduce the training time, as we no longer need to wait for the NeRF training to converge fully. Our real-time neural rendering approach may potentially enable new applications such as 6-DOF industrial and product visualizations, as well as next generation AR/VR systems. PlenOctrees are amenable to in-browser rendering as well; please visit the project page for the interactive online demo, as well as video and code: https://alexyu.net/plenoctrees

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PlenOctrees for Real-time Rendering of Neural Radiance Fields<br>pdf: <a href="https://t.co/Q46tEKYENv">https://t.co/Q46tEKYENv</a><br>abs: <a href="https://t.co/dnfpO6k1kJ">https://t.co/dnfpO6k1kJ</a><br>project page: <a href="https://t.co/JaLiuDY6Tk">https://t.co/JaLiuDY6Tk</a><br>github: <a href="https://t.co/O9BmKhvA2s">https://t.co/O9BmKhvA2s</a> <a href="https://t.co/waXq2HEXbz">pic.twitter.com/waXq2HEXbz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375261785629061121?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. An Image is Worth 16x16 Words, What is a Video Worth?

Gilad Sharir, Asaf Noy, Lihi Zelnik-Manor

- retweets: 828, favorites: 219 (03/27/2021 09:23:34)

- links: [abs](https://arxiv.org/abs/2103.13915) | [pdf](https://arxiv.org/pdf/2103.13915)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Leading methods in the domain of action recognition try to distill information from both the spatial and temporal dimensions of an input video. Methods that reach State of the Art (SotA) accuracy, usually make use of 3D convolution layers as a way to abstract the temporal information from video frames. The use of such convolutions requires sampling short clips from the input video, where each clip is a collection of closely sampled frames. Since each short clip covers a small fraction of an input video, multiple clips are sampled at inference in order to cover the whole temporal length of the video. This leads to increased computational load and is impractical for real-world applications. We address the computational bottleneck by significantly reducing the number of frames required for inference. Our approach relies on a temporal transformer that applies global attention over video frames, and thus better exploits the salient information in each frame. Therefore our approach is very input efficient, and can achieve SotA results (on Kinetics dataset) with a fraction of the data (frames per video), computation and latency. Specifically on Kinetics-400, we reach 78.8 top-1 accuracy with $\times 30$ less frames per video, and $\times 40$ faster inference than the current leading method. Code is available at: https://github.com/Alibaba-MIIL/STAM

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Image is Worth 16x16 Words, What is a Video Worth?<br><br>Reaches 78.8 top-1 acc. with 30x less frames per video, and ×40 faster inference than the current leading method on Kinetics-400 with a temporal transformer.<br><br>abs: <a href="https://t.co/hPxN7FfFg9">https://t.co/hPxN7FfFg9</a><br>code: <a href="https://t.co/2m18Mws0wv">https://t.co/2m18Mws0wv</a> <a href="https://t.co/5fRZkVTEen">pic.twitter.com/5fRZkVTEen</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1375249901651550209?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Image is Worth 16x16 Words, What is a Video Worth?<br>pdf: <a href="https://t.co/vf53DlyFwy">https://t.co/vf53DlyFwy</a><br>abs: <a href="https://t.co/2wG2qCT53o">https://t.co/2wG2qCT53o</a><br>github: <a href="https://t.co/ZGpmrFK7qy">https://t.co/ZGpmrFK7qy</a> <a href="https://t.co/2yUWK0tP5g">pic.twitter.com/2yUWK0tP5g</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375248366855143424?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo

- retweets: 584, favorites: 135 (03/27/2021 09:23:35)

- links: [abs](https://arxiv.org/abs/2103.14030) | [pdf](https://arxiv.org/pdf/2103.14030)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (86.4 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The code and models will be made publicly available at~\url{https://github.com/microsoft/Swin-Transformer}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows<br><br>Proposes Swin Transformer, which surpasses the previous obj. dection SotA by a large margin of +2.7 box AP and +2.6 mask AP on COCO.<br><br>abs: <a href="https://t.co/M4cGbInTkO">https://t.co/M4cGbInTkO</a><br>code: <a href="https://t.co/PiXLQAjD2i">https://t.co/PiXLQAjD2i</a> <a href="https://t.co/UJfu42roXk">pic.twitter.com/UJfu42roXk</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1375247799298662401?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows<br>pdf: <a href="https://t.co/QCiZdqejvq">https://t.co/QCiZdqejvq</a><br>abs: <a href="https://t.co/M6ramAWo35">https://t.co/M6ramAWo35</a><br>github: <a href="https://t.co/50c5fvyGqc">https://t.co/50c5fvyGqc</a> <a href="https://t.co/rk0DNOh6wo">pic.twitter.com/rk0DNOh6wo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375249309978013700?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Contrasting Contrastive Self-Supervised Representation Learning Models

Klemen Kotar, Gabriel Ilharco, Ludwig Schmidt, Kiana Ehsani, Roozbeh Mottaghi

- retweets: 504, favorites: 145 (03/27/2021 09:23:35)

- links: [abs](https://arxiv.org/abs/2103.14005) | [pdf](https://arxiv.org/pdf/2103.14005)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In the past few years, we have witnessed remarkable breakthroughs in self-supervised representation learning. Despite the success and adoption of representations learned through this paradigm, much is yet to be understood about how different training methods and datasets influence performance on downstream tasks. In this paper, we analyze contrastive approaches as one of the most successful and popular variants of self-supervised representation learning. We perform this analysis from the perspective of the training algorithms, pre-training datasets and end tasks. We examine over 700 training experiments including 30 encoders, 4 pre-training datasets and 20 diverse downstream tasks. Our experiments address various questions regarding the performance of self-supervised models compared to their supervised counterparts, current benchmarks used for evaluation, and the effect of the pre-training data on end task performance. We hope the insights and empirical evidence provided by this work will help future research in learning better visual representations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contrasting Contrastive Self-Supervised Representation Learning Models<br><br>Examines over 30 encoders, 4 pre-training datasets and 20 downstream tasks to address various questions such as the perf of SSL models compared to their supervised counterparts.<a href="https://t.co/jvA8GAenHB">https://t.co/jvA8GAenHB</a> <a href="https://t.co/fBiNbzMAZA">pic.twitter.com/fBiNbzMAZA</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1375254299312148482?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Vision Transformers for Dense Prediction

René Ranftl, Alexey Bochkovskiy, Vladlen Koltun

- retweets: 366, favorites: 119 (03/27/2021 09:23:35)

- links: [abs](https://arxiv.org/abs/2103.13413) | [pdf](https://arxiv.org/pdf/2103.13413)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art. Our models are available at https://github.com/intel-isl/DPT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vision Transformers for Dense Prediction<br>pdf: <a href="https://t.co/gxEsxTdUTo">https://t.co/gxEsxTdUTo</a><br>abs: <a href="https://t.co/8dAaoibQW6">https://t.co/8dAaoibQW6</a><br>github: <a href="https://t.co/tHq8wqRLQN">https://t.co/tHq8wqRLQN</a> <a href="https://t.co/gUYMTKaxGY">pic.twitter.com/gUYMTKaxGY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375247681300402176?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vision Transformers for Dense Prediction<br><br>Achieves the new SotA on monocular depth estimation and image segmentation by improving Vision Transformer. <br><br>abs: <a href="https://t.co/jG0m1jW6eG">https://t.co/jG0m1jW6eG</a><br>code: <a href="https://t.co/1FFgUyWon5">https://t.co/1FFgUyWon5</a> <a href="https://t.co/5JdhEdiKQv">pic.twitter.com/5JdhEdiKQv</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1375248871664746499?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. High-Fidelity Pluralistic Image Completion with Transformers

Ziyu Wan, Jingbo Zhang, Dongdong Chen, Jing Liao

- retweets: 323, favorites: 83 (03/27/2021 09:23:35)

- links: [abs](https://arxiv.org/abs/2103.14031) | [pdf](https://arxiv.org/pdf/2103.14031)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Image completion has made tremendous progress with convolutional neural networks (CNNs), because of their powerful texture modeling capacity. However, due to some inherent properties (e.g., local inductive prior, spatial-invariant kernels), CNNs do not perform well in understanding global structures or naturally support pluralistic completion. Recently, transformers demonstrate their power in modeling the long-term relationship and generating diverse results, but their computation complexity is quadratic to input length, thus hampering the application in processing high-resolution images. This paper brings the best of both worlds to pluralistic image completion: appearance prior reconstruction with transformer and texture replenishment with CNN. The former transformer recovers pluralistic coherent structures together with some coarse textures, while the latter CNN enhances the local texture details of coarse priors guided by the high-resolution masked images. The proposed method vastly outperforms state-of-the-art methods in terms of three aspects: 1) large performance boost on image fidelity even compared to deterministic completion methods; 2) better diversity and higher fidelity for pluralistic completion; 3) exceptional generalization ability on large masks and generic dataset, like ImageNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Fidelity Pluralistic Image Completion with Transformers<br>pdf: <a href="https://t.co/39h0sEFPeu">https://t.co/39h0sEFPeu</a><br>abs: <a href="https://t.co/NqtgQ6ZFHe">https://t.co/NqtgQ6ZFHe</a><br>project page: <a href="https://t.co/y0p6t9GMnR">https://t.co/y0p6t9GMnR</a> <a href="https://t.co/AO43jF8SVj">pic.twitter.com/AO43jF8SVj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375249934811824134?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Scaling-up Disentanglement for Image Translation

Aviv Gabbay, Yedid Hoshen

- retweets: 251, favorites: 120 (03/27/2021 09:23:35)

- links: [abs](https://arxiv.org/abs/2103.14017) | [pdf](https://arxiv.org/pdf/2103.14017)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Image translation methods typically aim to manipulate a set of labeled attributes (given as supervision at training time e.g. domain label) while leaving the unlabeled attributes intact. Current methods achieve either: (i) disentanglement, which exhibits low visual fidelity and can only be satisfied where the attributes are perfectly uncorrelated. (ii) visually-plausible translations, which are clearly not disentangled. In this work, we propose OverLORD, a single framework for disentangling labeled and unlabeled attributes as well as synthesizing high-fidelity images, which is composed of two stages; (i) Disentanglement: Learning disentangled representations with latent optimization. Differently from previous approaches, we do not rely on adversarial training or any architectural biases. (ii) Synthesis: Training feed-forward encoders for inferring the learned attributes and tuning the generator in an adversarial manner to increase the perceptual quality. When the labeled and unlabeled attributes are correlated, we model an additional representation that accounts for the correlated attributes and improves disentanglement. We highlight that our flexible framework covers multiple image translation settings e.g. attribute manipulation, pose-appearance translation, segmentation-guided synthesis and shape-texture transfer. In an extensive evaluation, we present significantly better disentanglement with higher translation quality and greater output diversity than state-of-the-art methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling-up Disentanglement for Image Translation<br>pdf: <a href="https://t.co/cDLI1ihNdA">https://t.co/cDLI1ihNdA</a><br>abs: <a href="https://t.co/sbt9NraAgq">https://t.co/sbt9NraAgq</a><br>project page: <a href="https://t.co/QKa0DEEptI">https://t.co/QKa0DEEptI</a> <a href="https://t.co/KaLrul3MxE">pic.twitter.com/KaLrul3MxE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375306077013676032?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The proposed model (bottom row) shows that Elon Musk will age worse than the baseline model (top row) shows.<a href="https://t.co/WKgVo6WWK7">https://t.co/WKgVo6WWK7</a> <a href="https://t.co/9n0blWf61a">pic.twitter.com/9n0blWf61a</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1375256817668792323?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Mask Attention Networks: Rethinking and Strengthen Transformer

Zhihao Fan, Yeyun Gong, Dayiheng Liu, Zhongyu Wei, Siyuan Wang, Jian Jiao, Nan Duan, Ruofei Zhang, Xuanjing Huang

- retweets: 143, favorites: 101 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.13597) | [pdf](https://arxiv.org/pdf/2103.13597)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformer is an attention-based neural network, which consists of two sublayers, namely, Self-Attention Network (SAN) and Feed-Forward Network (FFN). Existing research explores to enhance the two sublayers separately to improve the capability of Transformer for text representation. In this paper, we present a novel understanding of SAN and FFN as Mask Attention Networks (MANs) and show that they are two special cases of MANs with static mask matrices. However, their static mask matrices limit the capability for localness modeling in text representation learning. We therefore introduce a new layer named dynamic mask attention network (DMAN) with a learnable mask matrix which is able to model localness adaptively. To incorporate advantages of DMAN, SAN, and FFN, we propose a sequential layered structure to combine the three types of layers. Extensive experiments on various tasks, including neural machine translation and text summarization demonstrate that our model outperforms the original Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mask Attention Networks: Rethinking and Strengthen Transformer<br>pdf: <a href="https://t.co/ZqSmVCWEnc">https://t.co/ZqSmVCWEnc</a><br>abs: <a href="https://t.co/1I3v3qiMBp">https://t.co/1I3v3qiMBp</a> <a href="https://t.co/NWExoKJgmy">pic.twitter.com/NWExoKJgmy</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375278685008322564?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Contrast to Divide: Self-Supervised Pre-Training for Learning with Noisy  Labels

Evgenii Zheltonozhskii, Chaim Baskin, Avi Mendelson, Alex M. Bronstein, Or Litany

- retweets: 159, favorites: 43 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.13646) | [pdf](https://arxiv.org/pdf/2103.13646)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The success of learning with noisy labels (LNL) methods relies heavily on the success of a warm-up stage where standard supervised training is performed using the full (noisy) training set. In this paper, we identify a "warm-up obstacle": the inability of standard warm-up stages to train high quality feature extractors and avert memorization of noisy labels. We propose "Contrast to Divide" (C2D), a simple framework that solves this problem by pre-training the feature extractor in a self-supervised fashion. Using self-supervised pre-training boosts the performance of existing LNL approaches by drastically reducing the warm-up stage's susceptibility to noise level, shortening its duration, and increasing extracted feature quality. C2D works out of the box with existing methods and demonstrates markedly improved performance, especially in the high noise regime, where we get a boost of more than 27% for CIFAR-100 with 90% noise over the previous state of the art. In real-life noise settings, C2D trained on mini-WebVision outperforms previous works both in WebVision and ImageNet validation sets by 3% top-1 accuracy. We perform an in-depth analysis of the framework, including investigating the performance of different pre-training approaches and estimating the effective upper bound of the LNL performance with semi-supervised learning. Code for reproducing our experiments is available at https://github.com/ContrastToDivide/C2D

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper, C2D (<a href="https://t.co/AhrDVP8C0I">https://t.co/AhrDVP8C0I</a>, <a href="https://t.co/UcdS4nYTqH">https://t.co/UcdS4nYTqH</a>) shows how self-supervised pre-training boosts learning with noisy labels, achieves SOTA performance and provides in-depth analysis. Authors <a href="https://twitter.com/evgeniyzhe?ref_src=twsrc%5Etfw">@evgeniyzhe</a> <a href="https://twitter.com/ChaimBaskin?ref_src=twsrc%5Etfw">@ChaimBaskin</a> Avi Mendelson, Alex Bronstein, <a href="https://twitter.com/orlitany?ref_src=twsrc%5Etfw">@orlitany</a> 1/n</p>&mdash; Evgenii Zheltonozhskii (@evgeniyzhe) <a href="https://twitter.com/evgeniyzhe/status/1375486632728616969?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. TagMe: GPS-Assisted Automatic Object Annotation in Videos

Songtao He, Favyen Bastani, Mohammad Alizadeh, Hari Balakrishnan, Michael Cafarella, Tim Kraska, Sam Madden

- retweets: 156, favorites: 25 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.13428) | [pdf](https://arxiv.org/pdf/2103.13428)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Training high-accuracy object detection models requires large and diverse annotated datasets. However, creating these data-sets is time-consuming and expensive since it relies on human annotators. We design, implement, and evaluate TagMe, a new approach for automatic object annotation in videos that uses GPS data. When the GPS trace of an object is available, TagMe matches the object's motion from GPS trace and the pixels' motions in the video to find the pixels belonging to the object in the video and creates the bounding box annotations of the object. TagMe works using passive data collection and can continuously generate new object annotations from outdoor video streams without any human annotators. We evaluate TagMe on a dataset of 100 video clips. We show TagMe can produce high-quality object annotations in a fully-automatic and low-cost way. Compared with the traditional human-in-the-loop solution, TagMe can produce the same amount of annotations at a much lower cost, e.g., up to 110x.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TagMe: GPS-Assisted Automatic Object Annotation in<br>Videos<br>pdf: <a href="https://t.co/j2aA0ca9fB">https://t.co/j2aA0ca9fB</a><br>abs: <a href="https://t.co/L7MZstZNwJ">https://t.co/L7MZstZNwJ</a><br>project page: <a href="https://t.co/XluSez9bau">https://t.co/XluSez9bau</a> <a href="https://t.co/yCzeXWFyZN">pic.twitter.com/yCzeXWFyZN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375290131452600320?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Designing a Practical Degradation Model for Deep Blind Image  Super-Resolution

Kai Zhang, Jingyun Liang, Luc Van Gool, Radu Timofte

- retweets: 35, favorites: 40 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.14006) | [pdf](https://arxiv.org/pdf/2103.14006)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

It is widely acknowledged that single image super-resolution (SISR) methods would not perform well if the assumed degradation model deviates from those in real images. Although several degradation models take additional factors into consideration, such as blur, they are still not effective enough to cover the diverse degradations of real images. To address this issue, this paper proposes to design a more complex but practical degradation model that consists of randomly shuffled blur, downsampling and noise degradations. Specifically, the blur is approximated by two convolutions with isotropic and anisotropic Gaussian kernels; the downsampling is randomly chosen from nearest, bilinear and bicubic interpolations; the noise is synthesized by adding Gaussian noise with different noise levels, adopting JPEG compression with different quality factors, and generating processed camera sensor noise via reverse-forward camera image signal processing (ISP) pipeline model and RAW image noise model. To verify the effectiveness of the new degradation model, we have trained a deep blind ESRGAN super-resolver and then applied it to super-resolve both synthetic and real images with diverse degradations. The experimental results demonstrate that the new degradation model can help to significantly improve the practicability of deep super-resolvers, thus providing a powerful alternative solution for real SISR applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Designing a Practical Degradation Model for Deep Blind Image Super-Resolution<br>pdf: <a href="https://t.co/qqJEp0fIgX">https://t.co/qqJEp0fIgX</a><br>abs: <a href="https://t.co/ecjlOGEN2j">https://t.co/ecjlOGEN2j</a> <a href="https://t.co/f4iZCJdm8r">pic.twitter.com/f4iZCJdm8r</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375264476765769729?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance  Fields

Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan

- retweets: 30, favorites: 39 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.13415) | [pdf](https://arxiv.org/pdf/2103.13415)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

The rendering procedure used by neural radiance fields (NeRF) samples a scene with a single ray per pixel and may therefore produce renderings that are excessively blurred or aliased when training or testing images observe scene content at different resolutions. The straightforward solution of supersampling by rendering with multiple rays per pixel is impractical for NeRF, because rendering each ray requires querying a multilayer perceptron hundreds of times. Our solution, which we call "mip-NeRF" (a la "mipmap"), extends NeRF to represent the scene at a continuously-valued scale. By efficiently rendering anti-aliased conical frustums instead of rays, mip-NeRF reduces objectionable aliasing artifacts and significantly improves NeRF's ability to represent fine details, while also being 7% faster than NeRF and half the size. Compared to NeRF, mip-NeRF reduces average error rates by 16% on the dataset presented with NeRF and by 60% on a challenging multiscale variant of that dataset that we present. Mip-NeRF is also able to match the accuracy of a brute-force supersampled NeRF on our multiscale dataset while being 22x faster.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields<br>pdf: <a href="https://t.co/kNvcHQSsLj">https://t.co/kNvcHQSsLj</a><br>abs: <a href="https://t.co/5x2LRdINSt">https://t.co/5x2LRdINSt</a> <a href="https://t.co/zMUO3kxEmv">pic.twitter.com/zMUO3kxEmv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375251789491154945?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. The ThreeDWorld Transport Challenge: A Visually Guided Task-and-Motion  Planning Benchmark for Physically Realistic Embodied AI

Chuang Gan, Siyuan Zhou, Jeremy Schwartz, Seth Alter, Abhishek Bhandwaldar, Dan Gutfreund, Daniel L.K. Yamins, James J DiCarlo, Josh McDermott, Antonio Torralba, Joshua B. Tenenbaum

- retweets: 30, favorites: 36 (03/27/2021 09:23:36)

- links: [abs](https://arxiv.org/abs/2103.14025) | [pdf](https://arxiv.org/pdf/2103.14025)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We introduce a visually-guided and physics-driven task-and-motion planning benchmark, which we call the ThreeDWorld Transport Challenge. In this challenge, an embodied agent equipped with two 9-DOF articulated arms is spawned randomly in a simulated physical home environment. The agent is required to find a small set of objects scattered around the house, pick them up, and transport them to a desired final location. We also position containers around the house that can be used as tools to assist with transporting objects efficiently. To complete the task, an embodied agent must plan a sequence of actions to change the state of a large number of objects in the face of realistic physical constraints. We build this benchmark challenge using the ThreeDWorld simulation: a virtual 3D environment where all objects respond to physics, and where can be controlled using fully physics-driven navigation and interaction API. We evaluate several existing agents on this benchmark. Experimental results suggest that: 1) a pure RL model struggles on this challenge; 2) hierarchical planning-based agents can transport some objects but still far from solving this task. We anticipate that this benchmark will empower researchers to develop more intelligent physics-driven robots for the physical world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The ThreeDWorld Transport Challenge: A Visually Guided Task-and-Motion Planning Benchmark for Physically Realistic Embodied AI<a href="https://t.co/83LNGiSjBy">https://t.co/83LNGiSjBy</a><a href="https://t.co/q16iNl2ceR">https://t.co/q16iNl2ceR</a> <a href="https://t.co/8IlJgUOVWY">pic.twitter.com/8IlJgUOVWY</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1375254384339156993?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Fairness in Ranking: A Survey

Meike Zehlike, Ke Yang, Julia Stoyanovich

- retweets: 32, favorites: 23 (03/27/2021 09:23:37)

- links: [abs](https://arxiv.org/abs/2103.14000) | [pdf](https://arxiv.org/pdf/2103.14000)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent)

In the past few years, there has been much work on incorporating fairness requirements into algorithmic rankers, with contributions coming from the data management, algorithms, information retrieval, and recommender systems communities. In this survey we give a systematic overview of this work, offering a broad perspective that connects formalizations and algorithmic approaches across subfields. An important contribution of our work is in developing a common narrative around the value frameworks that motivate specific fairness-enhancing interventions in ranking. This allows us to unify the presentation of mitigation objectives and of algorithmic techniques to help meet those objectives or identify trade-offs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Honored to be giving a keynote at <a href="https://twitter.com/edbticdt2021?ref_src=twsrc%5Etfw">@edbticdt2021</a> in less than an hour on fairness &amp; diversity in ranking. Come to the talk or read the accompanying survey with <a href="https://twitter.com/MilkaLichtblau?ref_src=twsrc%5Etfw">@MilkaLichtblau</a> &amp; Ke Yang published just hours ago, a year in the making <a href="https://t.co/d5ISNqYrPY">https://t.co/d5ISNqYrPY</a> <a href="https://twitter.com/hashtag/ranking?src=hash&amp;ref_src=twsrc%5Etfw">#ranking</a> <a href="https://twitter.com/AIResponsibly?ref_src=twsrc%5Etfw">@AIResponsibly</a></p>&mdash; Julia Stoyanovich (@stoyanoj) <a href="https://twitter.com/stoyanoj/status/1375421605380296707?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Robust and Accurate Object Detection via Adversarial Learning

Xiangning Chen, Cihang Xie, Mingxing Tan, Li Zhang, Cho-Jui Hsieh, Boqing Gong

- retweets: 30, favorites: 24 (03/27/2021 09:23:37)

- links: [abs](https://arxiv.org/abs/2103.13886) | [pdf](https://arxiv.org/pdf/2103.13886)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Data augmentation has become a de facto component for training high-performance deep image classifiers, but its potential is under-explored for object detection. Noting that most state-of-the-art object detectors benefit from fine-tuning a pre-trained classifier, we first study how the classifiers' gains from various data augmentations transfer to object detection. The results are discouraging; the gains diminish after fine-tuning in terms of either accuracy or robustness. This work instead augments the fine-tuning stage for object detectors by exploring adversarial examples, which can be viewed as a model-dependent data augmentation. Our method dynamically selects the stronger adversarial images sourced from a detector's classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant. This model-dependent augmentation generalizes to different object detectors better than AutoAugment, a model-agnostic augmentation policy searched based on one particular detector. Our approach boosts the performance of state-of-the-art EfficientDets by +1.1 mAP on the COCO object detection benchmark. It also improves the detectors' robustness against natural distortions by +3.8 mAP and against domain shift by +1.3 mAP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robust and Accurate Object Detection via Adversarial Learning<br><br>Det-AdvProp boosts EfficientDet&#39;s accuracy by 1.1 mAP on COCO &amp; robustness by 3.8 mAP on COCO-C (natural corruption) and 1.3 mAP on VOC (domain shift). <a href="https://twitter.com/cihangxie?ref_src=twsrc%5Etfw">@cihangxie</a> <a href="https://twitter.com/tanmingxing?ref_src=twsrc%5Etfw">@tanmingxing</a> <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> <a href="https://twitter.com/hashtag/CVPR?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR</a><a href="https://t.co/jKed7zc9rI">https://t.co/jKed7zc9rI</a> <a href="https://t.co/c1y8xoNuPe">pic.twitter.com/c1y8xoNuPe</a></p>&mdash; Boqing Gong (@BoqingGo) <a href="https://twitter.com/BoqingGo/status/1375320972165713925?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Vectorization and Rasterization: Self-Supervised Learning for Sketch and  Handwriting

Ayan Kumar Bhunia, Pinaki Nath Chowdhury, Yongxin Yang, Timothy M. Hospedales, Tao Xiang, Yi-Zhe Song

- retweets: 20, favorites: 31 (03/27/2021 09:23:37)

- links: [abs](https://arxiv.org/abs/2103.13716) | [pdf](https://arxiv.org/pdf/2103.13716)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-supervised learning has gained prominence due to its efficacy at learning powerful representations from unlabelled data that achieve excellent performance on many challenging downstream tasks. However supervision-free pre-text tasks are challenging to design and usually modality specific. Although there is a rich literature of self-supervised methods for either spatial (such as images) or temporal data (sound or text) modalities, a common pre-text task that benefits both modalities is largely missing. In this paper, we are interested in defining a self-supervised pre-text task for sketches and handwriting data. This data is uniquely characterised by its existence in dual modalities of rasterized images and vector coordinate sequences. We address and exploit this dual representation by proposing two novel cross-modal translation pre-text tasks for self-supervised feature learning: Vectorization and Rasterization. Vectorization learns to map image space to vector coordinates and rasterization maps vector coordinates to image space. We show that the our learned encoder modules benefit both raster-based and vector-based downstream approaches to analysing hand-drawn data. Empirical evidence shows that our novel pre-text tasks surpass existing single and multi-modal self-supervision methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vectorization and Rasterization: Self-Supervised Learning for Sketch and Handwriting<br>pdf: <a href="https://t.co/XMpftC5rOW">https://t.co/XMpftC5rOW</a><br>abs: <a href="https://t.co/LZQFx3ssSG">https://t.co/LZQFx3ssSG</a> <a href="https://t.co/WQeslIhuW3">pic.twitter.com/WQeslIhuW3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1375251257770840064?ref_src=twsrc%5Etfw">March 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



