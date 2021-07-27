---
title: Hot Papers 2021-07-27
date: 2021-07-28T07:10:32.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-27"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-27"
socialImage: "/media/flying-marine.jpg"

---

# 1. Go Wider Instead of Deeper

Fuzhao Xue, Ziji Shi, Yuxuan Lou, Yong Liu, Yang You

- retweets: 548, favorites: 188 (07/28/2021 07:10:32)

- links: [abs](https://arxiv.org/abs/2107.11817) | [pdf](https://arxiv.org/pdf/2107.11817)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

The transformer has recently achieved impressive results on various tasks. To further improve the effectiveness and efficiency of the transformer, there are two trains of thought among existing works: (1) going wider by scaling to more trainable parameters; (2) going shallower by parameter sharing or model compressing along with the depth. However, larger models usually do not scale well when fewer tokens are available to train, and advanced parallelisms are required when the model is extremely large. Smaller models usually achieve inferior performance compared to the original transformer model due to the loss of representation power. In this paper, to achieve better performance with fewer trainable parameters, we propose a framework to deploy trainable parameters efficiently, by going wider instead of deeper. Specially, we scale along model width by replacing feed-forward network (FFN) with mixture-of-experts (MoE). We then share the MoE layers across transformer blocks using individual layer normalization. Such deployment plays the role to transform various semantic representations, which makes the model more parameter-efficient and effective. To evaluate our framework, we design WideNet and evaluate it on ImageNet-1K. Our best model outperforms Vision Transformer (ViT) by $1.46\%$ with $0.72 \times$ trainable parameters. Using $0.46 \times$ and $0.13 \times$ parameters, our WideNet can still surpass ViT and ViT-MoE by $0.83\%$ and $2.08\%$, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Go Wider Instead of Deeper<br>pdf: <a href="https://t.co/2OywbopfJU">https://t.co/2OywbopfJU</a><br>abs: <a href="https://t.co/cYt5hfoRaR">https://t.co/cYt5hfoRaR</a><br><br>best model outperforms Vision Transformer (ViT) by 1.46% with 0.72× trainable parameters <a href="https://t.co/yqro9GtDB3">pic.twitter.com/yqro9GtDB3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419824931181846528?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Eventually, go wider, or go home!<a href="https://t.co/NWHMLxp3sY">https://t.co/NWHMLxp3sY</a><a href="https://t.co/Q2iOHMJG5J">https://t.co/Q2iOHMJG5J</a><a href="https://t.co/ZYYvlgPaaD">https://t.co/ZYYvlgPaaD</a> <a href="https://t.co/fykS3342A8">https://t.co/fykS3342A8</a></p>&mdash; Hamid (@heghbalz) <a href="https://twitter.com/heghbalz/status/1419860653305733120?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Towards Generative Video Compression

Fabian Mentzer, Eirikur Agustsson, Johannes Ballé, David Minnen, Nick Johnston, George Toderici

- retweets: 536, favorites: 101 (07/28/2021 07:10:32)

- links: [abs](https://arxiv.org/abs/2107.12038) | [pdf](https://arxiv.org/pdf/2107.12038)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a neural video compression method based on generative adversarial networks (GANs) that outperforms previous neural video compression methods and is comparable to HEVC in a user study. We propose a technique to mitigate temporal error accumulation caused by recursive frame compression that uses randomized shifting and un-shifting, motivated by a spectral analysis. We present in detail the network design choices, their relative importance, and elaborate on the challenges of evaluating video compression methods in user studies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢📢📢 New paper: &quot;Towards Generative Video Compression&quot;. We present a GAN-based neural video compression system that is comparable to HEVC visually, and outperforms previous work that does not use GANs. Check it out on arxiv: <a href="https://t.co/0yBWEmQ3J9">https://t.co/0yBWEmQ3J9</a> <a href="https://t.co/CDXq2cLICI">pic.twitter.com/CDXq2cLICI</a></p>&mdash; Fabian Mentzer (@mentzer_f) <a href="https://twitter.com/mentzer_f/status/1419979080120901663?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Generative Video Compression<br>pdf: <a href="https://t.co/ZNguNuZ2QW">https://t.co/ZNguNuZ2QW</a><br>abs: <a href="https://t.co/uUqoOt5Vqf">https://t.co/uUqoOt5Vqf</a><br><br>a neural video compression method based on GANs that outperforms previous neural video compression methods<br>and is comparable to HEVC in a user study <a href="https://t.co/rYFKPbUn8J">pic.twitter.com/rYFKPbUn8J</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419837044948410373?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Contextual Transformer Networks for Visual Recognition

Yehao Li, Ting Yao, Yingwei Pan, Tao Mei

- retweets: 424, favorites: 91 (07/28/2021 07:10:32)

- links: [abs](https://arxiv.org/abs/2107.12292) | [pdf](https://arxiv.org/pdf/2107.12292)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Transformer with self-attention has led to the revolutionizing of natural language processing field, and recently inspires the emergence of Transformer-style architecture design with competitive results in numerous computer vision tasks. Nevertheless, most of existing designs directly employ self-attention over a 2D feature map to obtain the attention matrix based on pairs of isolated queries and keys at each spatial location, but leave the rich contexts among neighbor keys under-exploited. In this work, we design a novel Transformer-style module, i.e., Contextual Transformer (CoT) block, for visual recognition. Such design fully capitalizes on the contextual information among input keys to guide the learning of dynamic attention matrix and thus strengthens the capacity of visual representation. Technically, CoT block first contextually encodes input keys via a $3\times3$ convolution, leading to a static contextual representation of inputs. We further concatenate the encoded keys with input queries to learn the dynamic multi-head attention matrix through two consecutive $1\times1$ convolutions. The learnt attention matrix is multiplied by input values to achieve the dynamic contextual representation of inputs. The fusion of the static and dynamic contextual representations are finally taken as outputs. Our CoT block is appealing in the view that it can readily replace each $3\times3$ convolution in ResNet architectures, yielding a Transformer-style backbone named as Contextual Transformer Networks (CoTNet). Through extensive experiments over a wide range of applications (e.g., image recognition, object detection and instance segmentation), we validate the superiority of CoTNet as a stronger backbone. Source code is available at \url{https://github.com/JDAI-CV/CoTNet}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contextual Transformer Networks for Visual Recognition<br>pdf: <a href="https://t.co/tH0VpuPWWw">https://t.co/tH0VpuPWWw</a><br>abs: <a href="https://t.co/2x6X0vumBy">https://t.co/2x6X0vumBy</a><br>github: <a href="https://t.co/T9KGgm66J9">https://t.co/T9KGgm66J9</a><br><br>exploits the contextual information among input keys<br>to guide self-attention learning <a href="https://t.co/0UKgfx3syY">pic.twitter.com/0UKgfx3syY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419827453044961301?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for  Sequences

Zhenhai Zhu, Radu Soricut

- retweets: 301, favorites: 143 (07/28/2021 07:10:32)

- links: [abs](https://arxiv.org/abs/2107.11906) | [pdf](https://arxiv.org/pdf/2107.11906)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We describe an efficient hierarchical method to compute attention in the Transformer architecture. The proposed attention mechanism exploits a matrix structure similar to the Hierarchical Matrix (H-Matrix) developed by the numerical analysis community, and has linear run time and memory complexity. We perform extensive experiments to show that the inductive bias embodied by our hierarchical attention is effective in capturing the hierarchical structure in the sequences typical for natural language and vision tasks. Our method is superior to alternative sub-quadratic proposals by over +6 points on average on the Long Range Arena benchmark. It also sets a new SOTA test perplexity on One-Billion Word dataset with 5x fewer model parameters than that of the previous-best Transformer-based models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences<br><br>Gains +6 points on average on the Long<br>Range Arena benchmark over the subquadratic alternatives. <br><br>Sets a new SOTA ppl on One-Billion Word<br>dataset with 5x fewer model parameters.<a href="https://t.co/bP66VDZ8de">https://t.co/bP66VDZ8de</a> <a href="https://t.co/sxNxDmZ9bE">pic.twitter.com/sxNxDmZ9bE</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1419826024573927427?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">H-Transformer-1D: Fast One-Dimensional Hierarchical Attention for Sequences<br>pdf: <a href="https://t.co/zAoEjISOph">https://t.co/zAoEjISOph</a><br>abs: <a href="https://t.co/AwzWbH2Te4">https://t.co/AwzWbH2Te4</a><br><br>SOTA test perplexity on One-Billion Word dataset with 5x fewer model parameters than that of the previous-best Transformer-based models <a href="https://t.co/zgKYDp4qA4">pic.twitter.com/zgKYDp4qA4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419826120673923072?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. A brief note on understanding neural networks as Gaussian processes

Mengwu Guo

- retweets: 80, favorites: 51 (07/28/2021 07:10:33)

- links: [abs](https://arxiv.org/abs/2107.11892) | [pdf](https://arxiv.org/pdf/2107.11892)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

As a generalization of the work in [Lee et al., 2017], this note briefly discusses when the prior of a neural network output follows a Gaussian process, and how a neural-network-induced Gaussian process is formulated. The posterior mean functions of such a Gaussian process regression lie in the reproducing kernel Hilbert space defined by the neural-network-induced kernel. In the case of two-layer neural networks, the induced Gaussian processes provide an interpretation of the reproducing kernel Hilbert spaces whose union forms a Barron space.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A brief note on understanding neural networks as Gaussian processes. (arXiv:2107.11892v1 [cs.LG]) <a href="https://t.co/8s8SzRezmW">https://t.co/8s8SzRezmW</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1419926372651356161?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Realistic Simulation Framework for Learning with Label Noise

Keren Gu, Xander Masotto, Vandana Bachani, Balaji Lakshminarayanan, Jack Nikodem, Dong Yin

- retweets: 72, favorites: 29 (07/28/2021 07:10:33)

- links: [abs](https://arxiv.org/abs/2107.11413) | [pdf](https://arxiv.org/pdf/2107.11413)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

We propose a simulation framework for generating realistic instance-dependent noisy labels via a pseudo-labeling paradigm. We show that this framework generates synthetic noisy labels that exhibit important characteristics of the label noise in practical settings via comparison with the CIFAR10-H dataset. Equipped with controllable label noise, we study the negative impact of noisy labels across a few realistic settings to understand when label noise is more problematic. We also benchmark several existing algorithms for learning with noisy labels and compare their behavior on our synthetic datasets and on the datasets with independent random label noise. Additionally, with the availability of annotator information from our simulation framework, we propose a new technique, Label Quality Model (LQM), that leverages annotator features to predict and correct against noisy labels. We show that by adding LQM as a label correction step before applying existing noisy label techniques, we can further improve the models' performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Realistic Simulation Framework for Learning with Label Noise<br>pdf: <a href="https://t.co/fn4Pj58uSp">https://t.co/fn4Pj58uSp</a><br>abs: <a href="https://t.co/0Q4TxBXrt2">https://t.co/0Q4TxBXrt2</a><br>github: <a href="https://t.co/mAxDa4BmjB">https://t.co/mAxDa4BmjB</a><br><br>a simulation framework for generating realistic instance-dependent noisy labels via a pseudolabeling paradigm <a href="https://t.co/0TVA2fCbJm">pic.twitter.com/0TVA2fCbJm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419840328178208769?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. The Impact of Negative Sampling on Contrastive Structured World Models

Ondrej Biza, Elise van der Pol, Thomas Kipf

- retweets: 64, favorites: 22 (07/28/2021 07:10:33)

- links: [abs](https://arxiv.org/abs/2107.11676) | [pdf](https://arxiv.org/pdf/2107.11676)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

World models trained by contrastive learning are a compelling alternative to autoencoder-based world models, which learn by reconstructing pixel states. In this paper, we describe three cases where small changes in how we sample negative states in the contrastive loss lead to drastic changes in model performance. In previously studied Atari datasets, we show that leveraging time step correlations can double the performance of the Contrastive Structured World Model. We also collect a full version of the datasets to study contrastive learning under a more diverse set of experiences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Impact of Negative Sampling on Contrastive Structured World Models<br>pdf: <a href="https://t.co/C5UnwBKYvQ">https://t.co/C5UnwBKYvQ</a><br>abs: <a href="https://t.co/SsJYD5zGy7">https://t.co/SsJYD5zGy7</a><br>github: <a href="https://t.co/6rgbTT4PJ1">https://t.co/6rgbTT4PJ1</a> <a href="https://t.co/KWKoNKG7AT">pic.twitter.com/KWKoNKG7AT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419828428447027200?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Transcript to Video: Efficient Clip Sequencing from Texts

Yu Xiong, Fabian Caba Heilbron, Dahua Lin

- retweets: 42, favorites: 37 (07/28/2021 07:10:33)

- links: [abs](https://arxiv.org/abs/2107.11851) | [pdf](https://arxiv.org/pdf/2107.11851)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Among numerous videos shared on the web, well-edited ones always attract more attention. However, it is difficult for inexperienced users to make well-edited videos because it requires professional expertise and immense manual labor. To meet the demands for non-experts, we present Transcript-to-Video -- a weakly-supervised framework that uses texts as input to automatically create video sequences from an extensive collection of shots. Specifically, we propose a Content Retrieval Module and a Temporal Coherent Module to learn visual-language representations and model shot sequencing styles, respectively. For fast inference, we introduce an efficient search strategy for real-time video clip sequencing. Quantitative results and user studies demonstrate empirically that the proposed learning framework can retrieve content-relevant shots while creating plausible video sequences in terms of style. Besides, the run-time performance analysis shows that our framework can support real-world applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transcript to Video: Efficient Clip Sequencing from Texts<br>pdf: <a href="https://t.co/sLEnh8zIZl">https://t.co/sLEnh8zIZl</a><br>abs: <a href="https://t.co/pI1bWVelEg">https://t.co/pI1bWVelEg</a><br>project page: <a href="https://t.co/lhtepIOlnV">https://t.co/lhtepIOlnV</a> <a href="https://t.co/RNJ352eRPg">pic.twitter.com/RNJ352eRPg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419859250147381449?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. NeLF: Neural Light-transport Field for Portrait View Synthesis and  Relighting

Tiancheng Sun, Kai-En Lin, Sai Bi, Zexiang Xu, Ravi Ramamoorthi

- retweets: 42, favorites: 35 (07/28/2021 07:10:33)

- links: [abs](https://arxiv.org/abs/2107.12351) | [pdf](https://arxiv.org/pdf/2107.12351)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Human portraits exhibit various appearances when observed from different views under different lighting conditions. We can easily imagine how the face will look like in another setup, but computer algorithms still fail on this problem given limited observations. To this end, we present a system for portrait view synthesis and relighting: given multiple portraits, we use a neural network to predict the light-transport field in 3D space, and from the predicted Neural Light-transport Field (NeLF) produce a portrait from a new camera view under a new environmental lighting. Our system is trained on a large number of synthetic models, and can generalize to different synthetic and real portraits under various lighting conditions. Our method achieves simultaneous view synthesis and relighting given multi-view portraits as the input, and achieves state-of-the-art results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeLF: Neural Light-transport Field for Portrait View Synthesis and Relighting<br>pdf: <a href="https://t.co/hqHCdhOAM1">https://t.co/hqHCdhOAM1</a><br>abs: <a href="https://t.co/1ety67LGu8">https://t.co/1ety67LGu8</a><br>project page: <a href="https://t.co/5pF2u0tGDX">https://t.co/5pF2u0tGDX</a> <a href="https://t.co/3QgEg7KUix">pic.twitter.com/3QgEg7KUix</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1419833494545608724?ref_src=twsrc%5Etfw">July 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



