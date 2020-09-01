---
title: Hot Papers 2020-09-01
date: 2020-09-02T08:03:45.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-01"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-01"
socialImage: "/media/flying-marine.jpg"

---

# 1. Knowledge Efficient Deep Learning for Natural Language Processing

Hai Wang

- retweets: 65, favorites: 148 (09/02/2020 08:03:45)

- links: [abs](https://arxiv.org/abs/2008.12878) | [pdf](https://arxiv.org/pdf/2008.12878)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Deep learning has become the workhorse for a wide range of natural language processing applications. But much of the success of deep learning relies on annotated examples. Annotation is time-consuming and expensive to produce at scale. Here we are interested in methods for reducing the required quantity of annotated data -- by making the learning methods more knowledge efficient so as to make them more applicable in low annotation (low resource) settings. There are various classical approaches to making the models more knowledge efficient such as multi-task learning, transfer learning, weakly supervised and unsupervised learning etc. This thesis focuses on adapting such classical methods to modern deep learning models and algorithms.   This thesis describes four works aimed at making machine learning models more knowledge efficient. First, we propose a knowledge rich deep learning model (KRDL) as a unifying learning framework for incorporating prior knowledge into deep models. In particular, we apply KRDL built on Markov logic networks to denoise weak supervision. Second, we apply a KRDL model to assist the machine reading models to find the correct evidence sentences that can support their decision. Third, we investigate the knowledge transfer techniques in multilingual setting, where we proposed a method that can improve pre-trained multilingual BERT based on the bilingual dictionary. Fourth, we present an episodic memory network for language modelling, in which we encode the large external knowledge for the pre-trained GPT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ph.D. thesis discussing ways on how to develop knowledge efficient deep learning-based methods for NLP and making them more applicable in low resource settings.<br><br>by Hai Wang<a href="https://t.co/AYBw4V9DPm">https://t.co/AYBw4V9DPm</a> <a href="https://t.co/g06pPvADxV">pic.twitter.com/g06pPvADxV</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1300738111647944708?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Knowledge Efficient Deep Learning for Natural Language Processing <a href="https://t.co/eV1nNNOA2D">https://t.co/eV1nNNOA2D</a></p>&mdash; arXiv CS-CL (@arxiv_cscl) <a href="https://twitter.com/arxiv_cscl/status/1300609524957622274?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Off-Path TCP Exploits of the Mixed IPID Assignment

Xuewei Feng, Chuanpu Fu, Qi Li, Kun Sun, Ke Xu

- retweets: 37, favorites: 85 (09/02/2020 08:03:45)

- links: [abs](https://arxiv.org/abs/2008.12981) | [pdf](https://arxiv.org/pdf/2008.12981)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

In this paper, we uncover a new off-path TCP hijacking attack that can be used to terminate victim TCP connections or inject forged data into victim TCP connections by manipulating the new mixed IPID assignment method, which is widely used in Linux kernel version 4.18 and beyond to help defend against TCP hijacking attacks. The attack has three steps. First, an off-path attacker can downgrade the IPID assignment for TCP packets from the more secure per-socket-based policy to the less secure hash-based policy, building a shared IPID counter that forms a side channel on the victim. Second, the attacker detects the presence of TCP connections by observing the shared IPID counter on the victim. Third, the attacker infers the sequence number and the acknowledgment number of the detected connection by observing the side channel of the shared IPID counter. Consequently, the attacker can completely hijack the connection, i.e., resetting the connection or poisoning the data stream.   We evaluate the impacts of this off-path TCP attack in the real world. Our case studies of SSH DoS, manipulating web traffic, and poisoning BGP routing tables show its threat on a wide range of applications. Our experimental results show that our off-path TCP attack can be constructed within 215 seconds and the success rate is over 88%. Finally, we analyze the root cause of the exploit and develop a new IPID assignment method to defeat this attack. We prototype our defense in Linux 4.18 and confirm its effectiveness through extensive evaluation over real applications on the Internet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;a new off-path TCP hijacking attack that can be used to terminate victim TCP connections or inject forged data into victim TCP connections &quot; The paper is really interesting and they implemented a new IPID assignment proposal for Linux. <a href="https://t.co/W5aYYbE1hp">https://t.co/W5aYYbE1hp</a> <a href="https://t.co/Go36x2He5H">pic.twitter.com/Go36x2He5H</a></p>&mdash; Alexandre Dulaunoy (@adulau) <a href="https://twitter.com/adulau/status/1300729724516421632?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Beyond variance reduction: Understanding the true impact of baselines on  policy optimization

Wesley Chung, Valentin Thomas, Marlos C. Machado, Nicolas Le Roux

- retweets: 22, favorites: 75 (09/02/2020 08:03:45)

- links: [abs](https://arxiv.org/abs/2008.13773) | [pdf](https://arxiv.org/pdf/2008.13773)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Policy gradients methods are a popular and effective choice to train reinforcement learning agents in complex environments. The variance of the stochastic policy gradient is often seen as a key quantity to determine the effectiveness of the algorithm. Baselines are a common addition to reduce the variance of the gradient, but previous works have hardly ever considered other effects baselines may have on the optimization process. Using simple examples, we find that baselines modify the optimization dynamics even when the variance is the same. In certain cases, a baseline with lower variance may even be worse than another with higher variance. Furthermore, we find that the choice of baseline can affect the convergence of natural policy gradient, where certain baselines may lead to convergence to a suboptimal policy for any stepsize. Such behaviour emerges when sampling is constrained to be done using the current policy and we show how decoupling the sampling policy from the current policy guarantees convergence for a much wider range of baselines. More broadly, this work suggests that a more careful treatment of stochasticity in the updates---beyond the immediate variance---is necessary to understand the optimization process of policy gradient algorithms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[1/6] Our new preprint is now available on arXiv. We revisit baselines in policy gradient methods and show that they have a much bigger role than simply variance reduction! With <br>Wesley Chung, Valentin Thomas, and <a href="https://twitter.com/le_roux_nicolas?ref_src=twsrc%5Etfw">@le_roux_nicolas</a>.<a href="https://t.co/4lvyyHXSyB">https://t.co/4lvyyHXSyB</a> <a href="https://t.co/pFPrFySgAy">pic.twitter.com/pFPrFySgAy</a></p>&mdash; Marlos C. Machado (@MarlosCMachado) <a href="https://twitter.com/MarlosCMachado/status/1300841940783632385?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Efficient Computation of Expectations under Spanning Tree Distributions

Ran Zmigrod, Tim Vieira, Ryan Cotterell

- retweets: 11, favorites: 77 (09/02/2020 08:03:45)

- links: [abs](https://arxiv.org/abs/2008.12988) | [pdf](https://arxiv.org/pdf/2008.12988)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We give a general framework for inference in spanning tree models. We propose unified algorithms for the important cases of first-order expectations and second-order expectations in edge-factored, non-projective spanning-tree models. Our algorithms exploit a fundamental connection between gradients and expectations, which allows us to derive efficient algorithms. These algorithms are easy to implement, given the prevalence of automatic differentiation software. We motivate the development of our framework with several cautionary tales of previous re-search, which has developed numerous less-than-optimal algorithms for computing expectations and their gradients. We demonstrate how our framework efficiently computes several quantities with known algorithms, including the expected attachment score, entropy, and generalized expectation criteria. As a bonus, we give algorithms for quantities that are missing in the literature, including the KL divergence. In all cases, our approach matches the efficiency of existing algorithms and, in several cases, reducesthe runtime complexity by a factor (or two)of the sentence length. We validate the implementation of our framework through runtime experiments. We find our algorithms are upto $12$ and $26$ times faster than previous algorithms for computing the Shannon entropy and the gradient of the generalized expectation objective, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new TACL paper: Efficient Computation of Expectations under Spanning Tree Distributions [<a href="https://t.co/QWizOyQxpZ">https://t.co/QWizOyQxpZ</a>]:<br>🌲We save a factor of O(n) over existing algorithms<br>✨Explain the backprop ⇔ expectation connection<br>🔥Open source PyTorch implementation coming soon</p>&mdash; Ran Zmigrod (@RanZmigrod) <a href="https://twitter.com/RanZmigrod/status/1300832956701970434?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Dual Attention GANs for Semantic Image Synthesis

Hao Tang, Song Bai, Nicu Sebe

- retweets: 15, favorites: 57 (09/02/2020 08:03:46)

- links: [abs](https://arxiv.org/abs/2008.13024) | [pdf](https://arxiv.org/pdf/2008.13024)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

In this paper, we focus on the semantic image synthesis task that aims at transferring semantic label maps to photo-realistic images. Existing methods lack effective semantic constraints to preserve the semantic information and ignore the structural correlations in both spatial and channel dimensions, leading to unsatisfactory blurry and artifact-prone results. To address these limitations, we propose a novel Dual Attention GAN (DAGAN) to synthesize photo-realistic and semantically-consistent images with fine details from the input layouts without imposing extra training overhead or modifying the network architectures of existing methods. We also propose two novel modules, i.e., position-wise Spatial Attention Module (SAM) and scale-wise Channel Attention Module (CAM), to capture semantic structure attention in spatial and channel dimensions, respectively. Specifically, SAM selectively correlates the pixels at each position by a spatial attention map, leading to pixels with the same semantic label being related to each other regardless of their spatial distances. Meanwhile, CAM selectively emphasizes the scale-wise features at each channel by a channel attention map, which integrates associated features among all channel maps regardless of their scales. We finally sum the outputs of SAM and CAM to further improve feature representation. Extensive experiments on four challenging datasets show that DAGAN achieves remarkably better results than state-of-the-art methods, while using fewer model parameters. The source code and trained models are available at https://github.com/Ha0Tang/DAGAN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dual Attention GANs for Semantic Image Synthesis<br>pdf: <a href="https://t.co/bB7Z9I5iEc">https://t.co/bB7Z9I5iEc</a><br>abs: <a href="https://t.co/u7VtCXV9wI">https://t.co/u7VtCXV9wI</a><br>github: <a href="https://t.co/Wao9W0tc1b">https://t.co/Wao9W0tc1b</a> <a href="https://t.co/pwpsIO7yQQ">pic.twitter.com/pwpsIO7yQQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1300598112357101568?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. DeepFacePencil: Creating Face Images from Freehand Sketches

Yuhang Li, Xuejin Chen, Binxin Yang, Zihan Chen, Zhihua Cheng, Zheng-Jun Zha

- retweets: 13, favorites: 39 (09/02/2020 08:03:46)

- links: [abs](https://arxiv.org/abs/2008.13343) | [pdf](https://arxiv.org/pdf/2008.13343)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we explore the task of generating photo-realistic face images from hand-drawn sketches. Existing image-to-image translation methods require a large-scale dataset of paired sketches and images for supervision. They typically utilize synthesized edge maps of face images as training data. However, these synthesized edge maps strictly align with the edges of the corresponding face images, which limit their generalization ability to real hand-drawn sketches with vast stroke diversity. To address this problem, we propose DeepFacePencil, an effective tool that is able to generate photo-realistic face images from hand-drawn sketches, based on a novel dual generator image translation network during training. A novel spatial attention pooling (SAP) is designed to adaptively handle stroke distortions which are spatially varying to support various stroke styles and different levels of details. We conduct extensive experiments and the results demonstrate the superiority of our model over existing methods on both image quality and model generalization to hand-drawn sketches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeepFacePencil: Creating Face Images from Freehand Sketches<br>pdf: <a href="https://t.co/yk1NTCFoHT">https://t.co/yk1NTCFoHT</a><br>abs: <a href="https://t.co/9esHnFMZ0l">https://t.co/9esHnFMZ0l</a> <a href="https://t.co/pcChZIEC99">pic.twitter.com/pcChZIEC99</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1300620245992865794?ref_src=twsrc%5Etfw">September 1, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



