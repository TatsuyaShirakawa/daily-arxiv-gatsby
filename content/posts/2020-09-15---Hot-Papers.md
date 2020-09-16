---
title: Hot Papers 2020-09-15
date: 2020-09-16T09:18:44.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. High-Resolution Deep Image Matting

Haichao Yu, Ning Xu, Zilong Huang, Yuqian Zhou, Humphrey Shi

- retweets: 14, favorites: 84 (09/16/2020 09:18:44)

- links: [abs](https://arxiv.org/abs/2009.06613) | [pdf](https://arxiv.org/pdf/2009.06613)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image matting is a key technique for image and video editing and composition. Conventionally, deep learning approaches take the whole input image and an associated trimap to infer the alpha matte using convolutional neural networks. Such approaches set state-of-the-arts in image matting; however, they may fail in real-world matting applications due to hardware limitations, since real-world input images for matting are mostly of very high resolution. In this paper, we propose HDMatt, a first deep learning based image matting approach for high-resolution inputs. More concretely, HDMatt runs matting in a patch-based crop-and-stitch manner for high-resolution inputs with a novel module design to address the contextual dependency and consistency issues between different patches. Compared with vanilla patch-based inference which computes each patch independently, we explicitly model the cross-patch contextual dependency with a newly-proposed Cross-Patch Contextual module (CPC) guided by the given trimap. Extensive experiments demonstrate the effectiveness of the proposed method and its necessity for high-resolution inputs. Our HDMatt approach also sets new state-of-the-art performance on Adobe Image Matting and AlphaMatting benchmarks and produce impressive visual results on more real-world high-resolution images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Resolution Deep Image Matting<br>pdf: <a href="https://t.co/LLcfeu9QuX">https://t.co/LLcfeu9QuX</a><br>abs: <a href="https://t.co/kGFKFpR95U">https://t.co/kGFKFpR95U</a> <a href="https://t.co/563JzJPAyG">pic.twitter.com/563JzJPAyG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1305688687813054466?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. YOLObile: Real-Time Object Detection on Mobile Devices via  Compression-Compilation Co-Design

Yuxuan Cai, Hongjia Li, Geng Yuan, Wei Niu, Yanyu Li, Xulong Tang, Bin Ren, Yanzhi Wang

- retweets: 20, favorites: 67 (09/16/2020 09:18:45)

- links: [abs](https://arxiv.org/abs/2009.05697) | [pdf](https://arxiv.org/pdf/2009.05697)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The rapid development and wide utilization of object detection techniques have aroused attention on both accuracy and speed of object detectors. However, the current state-of-the-art object detection works are either accuracy-oriented using a large model but leading to high latency or speed-oriented using a lightweight model but sacrificing accuracy. In this work, we propose YOLObile framework, a real-time object detection on mobile devices via compression-compilation co-design. A novel block-punched pruning scheme is proposed for any kernel size. To improve computational efficiency on mobile devices, a GPU-CPU collaborative scheme is adopted along with advanced compiler-assisted optimizations. Experimental results indicate that our pruning scheme achieves 14$\times$ compression rate of YOLOv4 with 49.0 mAP. Under our YOLObile framework, we achieve 17 FPS inference speed using GPU on Samsung Galaxy S20. By incorporating our proposed GPU-CPU collaborative scheme, the inference speed is increased to 19.1 FPS, and outperforms the original YOLOv4 by 5$\times$ speedup.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">YOLObile: Real-Time Object Detection on Mobile Devices via Compression-Compilation Co-Design<br>pdf: <a href="https://t.co/7XLGMwc2ZK">https://t.co/7XLGMwc2ZK</a><br>abs: <a href="https://t.co/5lFjfG1lYx">https://t.co/5lFjfG1lYx</a> <a href="https://t.co/kza4bT4rvA">pic.twitter.com/kza4bT4rvA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1305696239456055299?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Improving Inversion and Generation Diversity in StyleGAN using a  Gaussianized Latent Space

Jonas Wulff, Antonio Torralba

- retweets: 16, favorites: 67 (09/16/2020 09:18:45)

- links: [abs](https://arxiv.org/abs/2009.06529) | [pdf](https://arxiv.org/pdf/2009.06529)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Modern Generative Adversarial Networks are capable of creating artificial, photorealistic images from latent vectors living in a low-dimensional learned latent space. It has been shown that a wide range of images can be projected into this space, including images outside of the domain that the generator was trained on. However, while in this case the generator reproduces the pixels and textures of the images, the reconstructed latent vectors are unstable and small perturbations result in significant image distortions. In this work, we propose to explicitly model the data distribution in latent space. We show that, under a simple nonlinear operation, the data distribution can be modeled as Gaussian and therefore expressed using sufficient statistics. This yields a simple Gaussian prior, which we use to regularize the projection of images into the latent space. The resulting projections lie in smoother and better behaved regions of the latent space, as shown using interpolation performance for both real and generated images. Furthermore, the Gaussian model of the distribution in latent space allows us to investigate the origins of artifacts in the generator output, and provides a method for reducing these artifacts while maintaining diversity of the generated images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improving Inversion and Generation Diversity in<br>StyleGAN using a Gaussianized Latent Space<br>pdf: <a href="https://t.co/wMmhurV32T">https://t.co/wMmhurV32T</a><br>abs: <a href="https://t.co/ToiiTdTIfg">https://t.co/ToiiTdTIfg</a> <a href="https://t.co/4avTQAeTnV">pic.twitter.com/4avTQAeTnV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1305683022646317058?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Unit Test Case Generation with Transformers

Michele Tufano, Dawn Drain, Alexey Svyatkovskiy, Shao Kun Deng, Neel Sundaresan

- retweets: 16, favorites: 52 (09/16/2020 09:18:45)

- links: [abs](https://arxiv.org/abs/2009.05617) | [pdf](https://arxiv.org/pdf/2009.05617)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Automated Unit Test Case generation has been the focus of extensive literature within the research community. Existing approaches are usually guided by the test coverage criteria, generating synthetic test cases that are often difficult to read or understand for developers. In this paper we propose AthenaTest, an approach that aims at generating unit test cases by learning from real-world, developer-written test cases. Our approach relies on a state-of-the-art sequence-to-sequence transformer model which is able to write useful test cases for a given method under test (i.e., focal method). We also introduce methods2test - the largest publicly available supervised parallel corpus of unit test case methods and corresponding focal methods in Java, which comprises 630k test cases mined from 70k open-source repositories hosted on GitHub. We use this dataset to train a transformer model to translate focal methods into the corresponding test cases. We evaluate the ability of our model in generating test cases using natural language processing as well as code-specific criteria. First, we assess the quality of the translation compared to the target test case, then we analyze properties of the test case such as syntactic correctness and number and variety of testing APIs (e.g., asserts). We execute the test cases, collect test coverage information, and compare them with test cases generated by EvoSuite and GPT-3. Finally, we survey professional developers on their preference in terms of readability, understandability, and testing effectiveness of the generated test cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unit Test Case Generation with Transformers<br>pdf: <a href="https://t.co/nHpqrkmr0f">https://t.co/nHpqrkmr0f</a><br>abs: <a href="https://t.co/zV6QaTSEQV">https://t.co/zV6QaTSEQV</a> <a href="https://t.co/DsQN4sKZ7v">pic.twitter.com/DsQN4sKZ7v</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1305672502383988736?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. QED: A Framework and Dataset for Explanations in Question Answering

Matthew Lamm, Jennimaria Palomaki, Chris Alberti, Daniel Andor, Eunsol Choi, Livio Baldini Soares, Michael Collins

- retweets: 11, favorites: 54 (09/16/2020 09:18:45)

- links: [abs](https://arxiv.org/abs/2009.06354) | [pdf](https://arxiv.org/pdf/2009.06354)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

A question answering system that in addition to providing an answer provides an explanation of the reasoning that leads to that answer has potential advantages in terms of debuggability, extensibility and trust. To this end, we propose QED, a linguistically informed, extensible framework for explanations in question answering. A QED explanation specifies the relationship between a question and answer according to formal semantic notions such as referential equality, sentencehood, and entailment. We describe and publicly release an expert-annotated dataset of QED explanations built upon a subset of the Google Natural Questions dataset, and report baseline models on two tasks -- post-hoc explanation generation given an answer, and joint question answering and explanation generation. In the joint setting, a promising result suggests that training on a relatively small amount of QED data can improve question answering. In addition to describing the formal, language-theoretic motivations for the QED approach, we describe a large user study showing that the presence of QED explanations significantly improves the ability of untrained raters to spot errors made by a strong neural QA baseline.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce our paper “QED: A Framework and Dataset for Explanations in Question Answering.” Work done as an intern at <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> with some great colleagues, including <a href="https://twitter.com/chris_alberti?ref_src=twsrc%5Etfw">@chris_alberti</a>,  <a href="https://twitter.com/eunsolc?ref_src=twsrc%5Etfw">@eunsolc</a>, and <a href="https://twitter.com/liviobs?ref_src=twsrc%5Etfw">@liviobs</a>. <a href="https://t.co/wVDJzBwxSz">https://t.co/wVDJzBwxSz</a></p>&mdash; Matthew Lamm (@MatthewRLamm) <a href="https://twitter.com/MatthewRLamm/status/1305899122512351232?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Compressed Deep Networks: Goodbye SVD, Hello Robust Low-Rank  Approximation

Murad Tukan, Alaa Maalouf, Matan Weksler, Dan Feldman

- retweets: 14, favorites: 41 (09/16/2020 09:18:45)

- links: [abs](https://arxiv.org/abs/2009.05647) | [pdf](https://arxiv.org/pdf/2009.05647)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A common technique for compressing a neural network is to compute the $k$-rank $\ell_2$ approximation $A_{k,2}$ of the matrix $A\in\mathbb{R}^{n\times d}$ that corresponds to a fully connected layer (or embedding layer). Here, $d$ is the number of the neurons in the layer, $n$ is the number in the next one, and $A_{k,2}$ can be stored in $O((n+d)k)$ memory instead of $O(nd)$.   This $\ell_2$-approximation minimizes the sum over every entry to the power of $p=2$ in the matrix $A - A_{k,2}$, among every matrix $A_{k,2}\in\mathbb{R}^{n\times d}$ whose rank is $k$. While it can be computed efficiently via SVD, the $\ell_2$-approximation is known to be very sensitive to outliers ("far-away" rows). Hence, machine learning uses e.g. Lasso Regression, $\ell_1$-regularization, and $\ell_1$-SVM that use the $\ell_1$-norm.   This paper suggests to replace the $k$-rank $\ell_2$ approximation by $\ell_p$, for $p\in [1,2]$. We then provide practical and provable approximation algorithms to compute it for any $p\geq1$, based on modern techniques in computational geometry.   Extensive experimental results on the GLUE benchmark for compressing BERT, DistilBERT, XLNet, and RoBERTa confirm this theoretical advantage. For example, our approach achieves $28\%$ compression of RoBERTa's embedding layer with only $0.63\%$ additive drop in the accuracy (without fine-tuning) in average over all tasks in GLUE, compared to $11\%$ drop using the existing $\ell_2$-approximation. Open code is provided for reproducing and extending our results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Compressed Deep Networks: Goodbye SVD, Hello Robust Low-Rank Approximation. <a href="https://t.co/I6hkSlG7Z0">https://t.co/I6hkSlG7Z0</a> <a href="https://t.co/wXbX1BIqlX">pic.twitter.com/wXbX1BIqlX</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1305824481001091080?ref_src=twsrc%5Etfw">September 15, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



