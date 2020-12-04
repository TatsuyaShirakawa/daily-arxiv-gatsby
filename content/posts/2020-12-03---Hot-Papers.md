---
title: Hot Papers 2020-12-03
date: 2020-12-04T09:25:51.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. Learning from others' mistakes: Avoiding dataset biases without modeling  them

Victor Sanh, Thomas Wolf, Yonatan Belinkov, Alexander M. Rush

- retweets: 870, favorites: 139 (12/04/2020 09:25:51)

- links: [abs](https://arxiv.org/abs/2012.01300) | [pdf](https://arxiv.org/pdf/2012.01300)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

State-of-the-art natural language processing (NLP) models often learn to model dataset biases and surface form correlations instead of features that target the intended underlying task. Previous work has demonstrated effective methods to circumvent these issues when knowledge of the bias is available. We consider cases where the bias issues may not be explicitly identified, and show a method for training models that learn to ignore these problematic correlations. Our approach relies on the observation that models with limited capacity primarily learn to exploit biases in the dataset. We can leverage the errors of such limited capacity models to train a more robust model in a product of experts, thus bypassing the need to hand-craft a biased model. We show the effectiveness of this method to retain improvements in out-of-distribution settings even if no particular bias is targeted by the biased model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨New pre-print on avoiding dataset biases<br><br>We show a method to train a model to ignore dataset biases without explicitly identifying/modeling them by learning from the errors of a “dumb” model.<br><br>Link: <a href="https://t.co/UqodTR58P1">https://t.co/UqodTR58P1</a><br>W/ 🤩 collaborators <a href="https://twitter.com/Thom_Wolf?ref_src=twsrc%5Etfw">@Thom_Wolf</a>, <a href="https://twitter.com/boknilev?ref_src=twsrc%5Etfw">@boknilev</a> &amp; <a href="https://twitter.com/srush_nlp?ref_src=twsrc%5Etfw">@srush_nlp</a> <a href="https://t.co/RWcNscxdmF">pic.twitter.com/RWcNscxdmF</a></p>&mdash; Victor Sanh (@SanhEstPasMoi) <a href="https://twitter.com/SanhEstPasMoi/status/1334553191934021637?ref_src=twsrc%5Etfw">December 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Learning Spatial Attention for Face Super-Resolution

Chaofeng Chen, Dihong Gong, Hao Wang, Zhifeng Li, Kwan-Yee K. Wong

- retweets: 182, favorites: 135 (12/04/2020 09:25:52)

- links: [abs](https://arxiv.org/abs/2012.01211) | [pdf](https://arxiv.org/pdf/2012.01211)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

General image super-resolution techniques have difficulties in recovering detailed face structures when applying to low resolution face images. Recent deep learning based methods tailored for face images have achieved improved performance by jointly trained with additional task such as face parsing and landmark prediction. However, multi-task learning requires extra manually labeled data. Besides, most of the existing works can only generate relatively low resolution face images (e.g., $128\times128$), and their applications are therefore limited. In this paper, we introduce a novel SPatial Attention Residual Network (SPARNet) built on our newly proposed Face Attention Units (FAUs) for face super-resolution. Specifically, we introduce a spatial attention mechanism to the vanilla residual blocks. This enables the convolutional layers to adaptively bootstrap features related to the key face structures and pay less attention to those less feature-rich regions. This makes the training more effective and efficient as the key face structures only account for a very small portion of the face image. Visualization of the attention maps shows that our spatial attention network can capture the key face structures well even for very low resolution faces (e.g., $16\times16$). Quantitative comparisons on various kinds of metrics (including PSNR, SSIM, identity similarity, and landmark detection) demonstrate the superiority of our method over current state-of-the-arts. We further extend SPARNet with multi-scale discriminators, named as SPARNetHD, to produce high resolution results (i.e., $512\times512$). We show that SPARNetHD trained with synthetic data cannot only produce high quality and high resolution outputs for synthetically degraded face images, but also show good generalization ability to real world low quality face images. Codes are available at \url{https://github.com/chaofengc/Face-SPARNet}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Spatial Attention for Face Super-Resolution<br>pdf: <a href="https://t.co/dhnPMOuNod">https://t.co/dhnPMOuNod</a><br>abs: <a href="https://t.co/G0kYSNbPMb">https://t.co/G0kYSNbPMb</a> <a href="https://t.co/8NbwEXqwhh">pic.twitter.com/8NbwEXqwhh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334332339099471873?ref_src=twsrc%5Etfw">December 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers

Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen

- retweets: 216, favorites: 83 (12/04/2020 09:25:52)

- links: [abs](https://arxiv.org/abs/2012.00759) | [pdf](https://arxiv.org/pdf/2012.00759)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present MaX-DeepLab, the first end-to-end model for panoptic segmentation. Our approach simplifies the current pipeline that depends heavily on surrogate sub-tasks and hand-designed components, such as box detection, non-maximum suppression, thing-stuff merging, etc. Although these sub-tasks are tackled by area experts, they fail to comprehensively solve the target task. By contrast, our MaX-DeepLab directly predicts class-labeled masks with a mask transformer, and is trained with a panoptic quality inspired loss via bipartite matching. Our mask transformer employs a dual-path architecture that introduces a global memory path in addition to a CNN path, allowing direct communication with any CNN layers. As a result, MaX-DeepLab shows a significant 7.1% PQ gain in the box-free regime on the challenging COCO dataset, closing the gap between box-based and box-free methods for the first time. A small variant of MaX-DeepLab improves 3.0% PQ over DETR with similar parameters and M-Adds. Furthermore, MaX-DeepLab, without test time augmentation, achieves new state-of-the-art 51.3% PQ on COCO test-dev set.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers<br>pdf: <a href="https://t.co/9ON8lHTegA">https://t.co/9ON8lHTegA</a><br>abs: <a href="https://t.co/wVExYKlHE2">https://t.co/wVExYKlHE2</a> <a href="https://t.co/2ITfKS2ZVb">pic.twitter.com/2ITfKS2ZVb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334342476816117762?ref_src=twsrc%5Etfw">December 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A Photogrammetry-based Framework to Facilitate Image-based Modeling and  Automatic Camera Tracking

Sebastian Bullinger, Christoph Bodensteiner, Michael Arens

- retweets: 102, favorites: 52 (12/04/2020 09:25:52)

- links: [abs](https://arxiv.org/abs/2012.01044) | [pdf](https://arxiv.org/pdf/2012.01044)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We propose a framework that extends Blender to exploit Structure from Motion (SfM) and Multi-View Stereo (MVS) techniques for image-based modeling tasks such as sculpting or camera and motion tracking. Applying SfM allows us to determine camera motions without manually defining feature tracks or calibrating the cameras used to capture the image data. With MVS we are able to automatically compute dense scene models, which is not feasible with the built-in tools of Blender. Currently, our framework supports several state-of-the-art SfM and MVS pipelines. The modular system design enables us to integrate further approaches without additional effort. The framework is publicly available as an open source software package.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Photogrammetry-based Framework to Facilitate Image-based Modeling and Automatic Camera Tracking<br>pdf: <a href="https://t.co/LIBQAIaMGY">https://t.co/LIBQAIaMGY</a><br>abs: <a href="https://t.co/kaCccnIdXR">https://t.co/kaCccnIdXR</a><br>github: <a href="https://t.co/Y3HncPAeUk">https://t.co/Y3HncPAeUk</a> <a href="https://t.co/MhYEt6gM05">pic.twitter.com/MhYEt6gM05</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334359724670783491?ref_src=twsrc%5Etfw">December 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware  Image Synthesis

Eric R. Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, Gordon Wetzstein

- retweets: 42, favorites: 56 (12/04/2020 09:25:52)

- links: [abs](https://arxiv.org/abs/2012.00926) | [pdf](https://arxiv.org/pdf/2012.00926)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We have witnessed rapid progress on 3D-aware image synthesis, leveraging recent advances in generative visual models and neural rendering. Existing approaches however fall short in two ways: first, they may lack an underlying 3D representation or rely on view-inconsistent rendering, hence synthesizing images that are not multi-view consistent; second, they often depend upon representation network architectures that are not expressive enough, and their results thus lack in image quality. We propose a novel generative model, named Periodic Implicit Generative Adversarial Networks ($\pi$-GAN or pi-GAN), for high-quality 3D-aware image synthesis. $\pi$-GAN leverages neural representations with periodic activation functions and volumetric rendering to represent scenes as view-consistent 3D representations with fine detail. The proposed approach obtains state-of-the-art results for 3D-aware image synthesis with multiple real and synthetic datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis<br>pdf: <a href="https://t.co/LP716AJNfm">https://t.co/LP716AJNfm</a><br>abs: <a href="https://t.co/a6ylN0EEEy">https://t.co/a6ylN0EEEy</a> <a href="https://t.co/MQysrSAs53">pic.twitter.com/MQysrSAs53</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334322260191141889?ref_src=twsrc%5Etfw">December 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



