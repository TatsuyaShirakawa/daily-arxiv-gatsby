---
title: Hot Papers 2021-05-05
date: 2021-05-06T09:06:46.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. MLP-Mixer: An all-MLP Architecture for Vision

Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy

- retweets: 7620, favorites: 108 (05/06/2021 09:06:46)

- links: [abs](https://arxiv.org/abs/2105.01601) | [pdf](https://arxiv.org/pdf/2105.01601)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Convolutional Neural Networks (CNNs) are the go-to model for computer vision. Recently, attention-based networks, such as the Vision Transformer, have also become popular. In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models. We hope that these results spark further research beyond the realms of well established CNNs and Transformers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper from Brain Zurich and Berlin!<br><br>We try a conv and attention free vision architecture: MLP-Mixer (<a href="https://t.co/5tRZ7fRikn">https://t.co/5tRZ7fRikn</a>)<br><br>Simple is good, so we went as minimalist as possible (just MLPs!) to see whether modern training methods &amp; data is sufficient... <a href="https://t.co/cZG5E0IGXQ">pic.twitter.com/cZG5E0IGXQ</a></p>&mdash; Neil Houlsby (@neilhoulsby) <a href="https://twitter.com/neilhoulsby/status/1389822601741144066?ref_src=twsrc%5Etfw">May 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Where and When: Space-Time Attention for Audio-Visual Explanations

Yanbei Chen, Thomas Hummel, A. Sophia Koepke, Zeynep Akata

- retweets: 169, favorites: 71 (05/06/2021 09:06:46)

- links: [abs](https://arxiv.org/abs/2105.01517) | [pdf](https://arxiv.org/pdf/2105.01517)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Explaining the decision of a multi-modal decision-maker requires to determine the evidence from both modalities. Recent advances in XAI provide explanations for models trained on still images. However, when it comes to modeling multiple sensory modalities in a dynamic world, it remains underexplored how to demystify the mysterious dynamics of a complex multi-modal model. In this work, we take a crucial step forward and explore learnable explanations for audio-visual recognition. Specifically, we propose a novel space-time attention network that uncovers the synergistic dynamics of audio and visual data over both space and time. Our model is capable of predicting the audio-visual video events, while justifying its decision by localizing where the relevant visual cues appear, and when the predicted sounds occur in videos. We benchmark our model on three audio-visual video event datasets, comparing extensively to multiple recent multi-modal representation learners and intrinsic explanation models. Experimental results demonstrate the clear superior performance of our model over the existing methods on audio-visual video event recognition. Moreover, we conduct an in-depth study to analyze the explainability of our model based on robustness analysis via perturbation tests and pointing games using human annotations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Where and When: Space-Time Attention for Audio-Visual Explanations<br>pdf: <a href="https://t.co/h1lmTtmLbf">https://t.co/h1lmTtmLbf</a><br>abs: <a href="https://t.co/w7QRb8g4M1">https://t.co/w7QRb8g4M1</a><br><br>STAN is a strong audio-visual representation learners and offers impressive model performance on audio-visual event recognition <a href="https://t.co/lX0ZUFhDR4">pic.twitter.com/lX0ZUFhDR4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1389778704709083138?ref_src=twsrc%5Etfw">May 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. COMISR: Compression-Informed Video Super-Resolution

Yinxiao Li, Pengchong Jin, Feng Yang, Ce Liu, Ming-Hsuan Yang, Peyman Milanfar

- retweets: 64, favorites: 33 (05/06/2021 09:06:46)

- links: [abs](https://arxiv.org/abs/2105.01237) | [pdf](https://arxiv.org/pdf/2105.01237)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Most video super-resolution methods focus on restoring high-resolution video frames from low-resolution videos without taking into account compression. However, most videos on the web or mobile devices are compressed, and the compression can be severe when the bandwidth is limited. In this paper, we propose a new compression-informed video super-resolution model to restore high-resolution content without introducing artifacts caused by compression. The proposed model consists of three modules for video super-resolution: bi-directional recurrent warping, detail-preserving flow estimation, and Laplacian enhancement. All these three modules are used to deal with compression properties such as the location of the intra-frames in the input and smoothness in the output frames. For thorough performance evaluation, we conducted extensive experiments on standard datasets with a wide range of compression rates, covering many real video use cases. We showed that our method not only recovers high-resolution content on uncompressed frames from the widely-used benchmark datasets, but also achieves state-of-the-art performance in super-resolving compressed videos based on numerous quantitative metrics. We also evaluated the proposed method by simulating streaming from YouTube to demonstrate its effectiveness and robustness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">COMISR: Compression-Informed Video Super-Resolution<br>pdf: <a href="https://t.co/cN2ArvyXVD">https://t.co/cN2ArvyXVD</a><br>abs: <a href="https://t.co/jstRrHgyDa">https://t.co/jstRrHgyDa</a><br><br>model consists of three modules for video super-resolution: bi-directional recurrent warping, detail-preserving flow estimation, and Laplacian enhancement <a href="https://t.co/zRG76ieaEd">pic.twitter.com/zRG76ieaEd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1389776625441591302?ref_src=twsrc%5Etfw">May 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. VQCPC-GAN: Variable-length Adversarial Audio Synthesis using  Vector-Quantized Contrastive Predictive Coding

Javier Nistal, Cyran Aouameur, Stefan Lattner, Gaël Richard

- retweets: 72, favorites: 23 (05/06/2021 09:06:46)

- links: [abs](https://arxiv.org/abs/2105.01531) | [pdf](https://arxiv.org/pdf/2105.01531)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Influenced by the field of Computer Vision, Generative Adversarial Networks (GANs) are often adopted for the audio domain using fixed-size two-dimensional spectrogram representations as the "image data". However, in the (musical) audio domain, it is often desired to generate output of variable duration. This paper presents VQCPC-GAN, an adversarial framework for synthesizing variable-length audio by exploiting Vector-Quantized Contrastive Predictive Coding (VQCPC). A sequence of VQCPC tokens extracted from real audio data serves as conditional input to a GAN architecture, providing step-wise time-dependent features of the generated content. The input noise z (characteristic in adversarial architectures) remains fixed over time, ensuring temporal consistency of global features. We evaluate the proposed model by comparing a diverse set of metrics against various strong baselines. Results show that, even though the baselines score best, VQCPC-GAN achieves comparable performance even when generating variable-length audio. Numerous sound examples are provided in the accompanying website, and we release the code for reproducibility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper submitted to WASPAA21!<br><br>VQCPC-GAN: Variable-length Adversarial Audio Synthesis Using Vector-Quantized Contrastive Predictive Coding<br><br>📑<a href="https://t.co/CZ9FNsX2c7">https://t.co/CZ9FNsX2c7</a><br>💻<a href="https://t.co/OCrTZi33Ad">https://t.co/OCrTZi33Ad</a> (soon)<br>🔊<a href="https://t.co/JndzKzSSUb">https://t.co/JndzKzSSUb</a><a href="https://twitter.com/cyranaouameur?ref_src=twsrc%5Etfw">@cyranaouameur</a>  <a href="https://twitter.com/deeplearnmusic?ref_src=twsrc%5Etfw">@deeplearnmusic</a> <a href="https://twitter.com/RichardGal8?ref_src=twsrc%5Etfw">@RichardGal8</a></p>&mdash; Javier Nistal (@latentspaces) <a href="https://twitter.com/latentspaces/status/1389892003123679236?ref_src=twsrc%5Etfw">May 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Consensus Dynamics and Opinion Formation on Hypergraphs

Leonie Neuhäuser, Renaud Lambiotte, Michael T. Schaub

- retweets: 42, favorites: 20 (05/06/2021 09:06:47)

- links: [abs](https://arxiv.org/abs/2105.01369) | [pdf](https://arxiv.org/pdf/2105.01369)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

In this chapter, we derive and analyse models for consensus dynamics on hypergraphs. As we discuss, unless there are nonlinear node interaction functions, it is always possible to rewrite the system in terms of a new network of effective pairwise node interactions, regardless of the initially underlying multi-way interaction structure. We thus focus on dynamics based on a certain class of non-linear interaction functions, which can model different sociological phenomena such as peer pressure and stubbornness. Unlike for linear consensus dynamics on networks, we show how our nonlinear model dynamics can cause shifts away from the average system state. We examine how these shifts are influenced by the distribution of the initial states, the underlying hypergraph structure and different forms of non-linear scaling of the node interaction function.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Interested in consensus dynamics on hypergraphs, in the effects of peer-pressure and stubbornness and why non-linearity is essential for higher-order effects? We revise this in: &quot;Consensus Dynamics and Opinion Formation on Hypergraphs&quot;(<a href="https://t.co/DyJ8q8CjCT">https://t.co/DyJ8q8CjCT</a>). Enjoy the read!</p>&mdash; Leonie Neuhäuser (@leoneuhaeuser) <a href="https://twitter.com/leoneuhaeuser/status/1389895330431488004?ref_src=twsrc%5Etfw">May 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



