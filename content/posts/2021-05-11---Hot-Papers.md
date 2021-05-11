---
title: Hot Papers 2021-05-11
date: 2021-05-12T08:56:20.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-11"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-11"
socialImage: "/media/flying-marine.jpg"

---

# 1. The Modern Mathematics of Deep Learning

Julius Berner, Philipp Grohs, Gitta Kutyniok, Philipp Petersen

- retweets: 5214, favorites: 439 (05/12/2021 08:56:20)

- links: [abs](https://arxiv.org/abs/2105.04026) | [pdf](https://arxiv.org/pdf/2105.04026)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We describe the new field of mathematical analysis of deep learning. This field emerged around a list of research questions that were not answered within the classical framework of learning theory. These questions concern: the outstanding generalization power of overparametrized neural networks, the role of depth in deep architectures, the apparent absence of the curse of dimensionality, the surprisingly successful optimization performance despite the non-convexity of the problem, understanding what features are learned, why deep architectures perform exceptionally well in physical problems, and which fine aspects of an architecture affect the behavior of a learning task in which way. We present an overview of modern approaches that yield partial answers to these questions. For selected approaches, we describe the main ideas in more detail.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Modern Mathematics of Deep Learning. (arXiv:2105.04026v1 [cs.LG]) <a href="https://t.co/FLzJGgcO7X">https://t.co/FLzJGgcO7X</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1392023521585475589?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Self-Supervised Learning with Swin Transformers

Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, Han Hu

- retweets: 1680, favorites: 196 (05/12/2021 08:56:20)

- links: [abs](https://arxiv.org/abs/2105.04553) | [pdf](https://arxiv.org/pdf/2105.04553)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We are witnessing a modeling shift from CNN to Transformers in computer vision. In this paper, we present a self-supervised learning approach called MoBY, with Vision Transformers as its backbone architecture. The approach is basically a combination of MoCo v2 and BYOL, tuned to achieve reasonably high accuracy on ImageNet-1K linear evaluation: 72.8% and 75.0% top-1 accuracy using DeiT-S and Swin-T, respectively, by 300-epoch training. The performance is slightly better than recent works of MoCo v3 and DINO which adopt DeiT as the backbone, but with much lighter tricks.   More importantly, the general-purpose Swin Transformer backbone enables us to also evaluate the learnt representations on downstream tasks such as object detection and semantic segmentation, in contrast to a few recent approaches built on ViT/DeiT which only report linear evaluation results on ImageNet-1K due to ViT/DeiT not tamed for these dense prediction tasks. We hope our results can facilitate more comprehensive evaluation of self-supervised learning methods designed for Transformer architectures. Our code and models are available at https://github.com/SwinTransformer/Transformer-SSL, which will be continually enriched.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Learning with Swin Transformers<br>pdf: <a href="https://t.co/uNAcu1JViD">https://t.co/uNAcu1JViD</a><br>abs: <a href="https://t.co/J70bvOq6nc">https://t.co/J70bvOq6nc</a><br>github: <a href="https://t.co/5dktODVxa9">https://t.co/5dktODVxa9</a><br><br>a self-supervised learning approach called MoBY, with Vision Transformers as its backbone architecture <a href="https://t.co/4yJm1AWHkh">pic.twitter.com/4yJm1AWHkh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391930163588440065?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Stochastic Image-to-Video Synthesis using cINNs

Michael Dorkenwald, Timo Milbich, Andreas Blattmann, Robin Rombach, Konstantinos G. Derpanis, Björn Ommer

- retweets: 466, favorites: 126 (05/12/2021 08:56:20)

- links: [abs](https://arxiv.org/abs/2105.04551) | [pdf](https://arxiv.org/pdf/2105.04551)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Video understanding calls for a model to learn the characteristic interplay between static scene content and its dynamics: Given an image, the model must be able to predict a future progression of the portrayed scene and, conversely, a video should be explained in terms of its static image content and all the remaining characteristics not present in the initial frame. This naturally suggests a bijective mapping between the video domain and the static content as well as residual information. In contrast to common stochastic image-to-video synthesis, such a model does not merely generate arbitrary videos progressing the initial image. Given this image, it rather provides a one-to-one mapping between the residual vectors and the video with stochastic outcomes when sampling. The approach is naturally implemented using a conditional invertible neural network (cINN) that can explain videos by independently modelling static and other video characteristics, thus laying the basis for controlled video synthesis. Experiments on four diverse video datasets demonstrate the effectiveness of our approach in terms of both the quality and diversity of the synthesized results. Our project page is available at https://bit.ly/3t66bnU.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work &quot;Stochastic Image-to-Video Synthesis using cINNs&quot; accepted to <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a>.<br><br>Joint work with <a href="https://twitter.com/timoMil?ref_src=twsrc%5Etfw">@timoMil</a>  <a href="https://twitter.com/andi_blatt?ref_src=twsrc%5Etfw">@andi_blatt</a>  <a href="https://twitter.com/robrombach?ref_src=twsrc%5Etfw">@robrombach</a> <a href="https://twitter.com/CSProfKGD?ref_src=twsrc%5Etfw">@csprofkgd</a> and B. Ommer.<br><br>arxiv: <a href="https://t.co/S6SEOneJjZ">https://t.co/S6SEOneJjZ</a><br>project page: <a href="https://t.co/NRzLBVQXsy">https://t.co/NRzLBVQXsy</a><br>code: <a href="https://t.co/K7sQqywY0h">https://t.co/K7sQqywY0h</a> <a href="https://t.co/9BUmCnMkIF">pic.twitter.com/9BUmCnMkIF</a></p>&mdash; Michael Dorkenwald (@mdorkenw) <a href="https://twitter.com/mdorkenw/status/1392053637908664321?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stochastic Image-to-Video Synthesis using cINNs<br><br>Performs diverse, high-quality controlled video synthesis with a conditional invertible neural network (cINN).<br><br>project: <a href="https://t.co/3teif6jAs5">https://t.co/3teif6jAs5</a><br>abs: <a href="https://t.co/88e0EpmaMq">https://t.co/88e0EpmaMq</a> <a href="https://t.co/zvD2jRGPk9">pic.twitter.com/zvD2jRGPk9</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1391922881852952578?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stochastic Image-to-Video Synthesis using cINNs<br>pdf: <a href="https://t.co/at7Y1jQe2K">https://t.co/at7Y1jQe2K</a><br>abs: <a href="https://t.co/IPAfQ1pKaa">https://t.co/IPAfQ1pKaa</a><br>project page: <a href="https://t.co/SivYxbqiM0">https://t.co/SivYxbqiM0</a> <a href="https://t.co/TQ3Ab7pf2E">pic.twitter.com/TQ3Ab7pf2E</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391936920054272004?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. How could Neural Networks understand Programs?

Dinglan Peng, Shuxin Zheng, Yatao Li, Guolin Ke, Di He, Tie-Yan Liu

- retweets: 325, favorites: 154 (05/12/2021 08:56:21)

- links: [abs](https://arxiv.org/abs/2105.04297) | [pdf](https://arxiv.org/pdf/2105.04297)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

Semantic understanding of programs is a fundamental problem for programming language processing (PLP). Recent works that learn representations of code based on pre-training techniques in NLP have pushed the frontiers in this direction. However, the semantics of PL and NL have essential differences. These being ignored, we believe it is difficult to build a model to better understand programs, by either directly applying off-the-shelf NLP pre-training techniques to the source code, or adding features to the model by the heuristic. In fact, the semantics of a program can be rigorously defined by formal semantics in PL theory. For example, the operational semantics, describes the meaning of a valid program as updating the environment (i.e., the memory address-value function) through fundamental operations, such as memory I/O and conditional branching. Inspired by this, we propose a novel program semantics learning paradigm, that the model should learn from information composed of (1) the representations which align well with the fundamental operations in operational semantics, and (2) the information of environment transition, which is indispensable for program understanding. To validate our proposal, we present a hierarchical Transformer-based pre-training model called OSCAR to better facilitate the understanding of programs. OSCAR learns from intermediate representation (IR) and an encoded representation derived from static analysis, which are used for representing the fundamental operations and approximating the environment transitions respectively. OSCAR empirically shows the outstanding capability of program semantics understanding on many practical software engineering tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How could Neural Networks understand Programs?<br>pdf: <a href="https://t.co/FgocOx4OEY">https://t.co/FgocOx4OEY</a><br>abs: <a href="https://t.co/E6eIbwQ9u2">https://t.co/E6eIbwQ9u2</a><br><br>a hierarchical Transformer-based pre-training model<br>called OSCAR to better facilitate the understanding of programs <a href="https://t.co/RuPPPz6rVc">pic.twitter.com/RuPPPz6rVc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391928740175925251?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with Just  One Transformer VAE

Shih-Lun Wu, Yi-Hsuan Yang

- retweets: 331, favorites: 133 (05/12/2021 08:56:21)

- links: [abs](https://arxiv.org/abs/2105.04090) | [pdf](https://arxiv.org/pdf/2105.04090)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Transformers and variational autoencoders (VAE) have been extensively employed for symbolic (e.g., MIDI) domain music generation. While the former boast an impressive capability in modeling long sequences, the latter allow users to willingly exert control over different parts (e.g., bars) of the music to be generated. In this paper, we are interested in bringing the two together to construct a single model that exhibits both strengths. The task is split into two steps. First, we equip Transformer decoders with the ability to accept segment-level, time-varying conditions during sequence generation. Subsequently, we combine the developed and tested in-attention decoder with a Transformer encoder, and train the resulting MuseMorphose model with the VAE objective to achieve style transfer of long musical pieces, in which users can specify musical attributes including rhythmic intensity and polyphony (i.e., harmonic fullness) they desire, down to the bar level. Experiments show that MuseMorphose outperforms recurrent neural network (RNN) based prior art on numerous widely-used metrics for style transfer tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint from our lab! by <a href="https://twitter.com/slseanwu?ref_src=twsrc%5Etfw">@slseanwu</a> <br>--<br>🐋MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with Just One Transformer VAE<a href="https://t.co/0zV0CbNeHS">https://t.co/0zV0CbNeHS</a><a href="https://t.co/5nPw2R62L4">https://t.co/5nPw2R62L4</a><br>(style transfer on long musical pieces + user control of musical attributes down to bar level) <a href="https://t.co/3g3WjOZliW">pic.twitter.com/3g3WjOZliW</a></p>&mdash; Yi-Hsuan Yang (@affige_yang) <a href="https://twitter.com/affige_yang/status/1391923941099278341?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MuseMorphose: Full-Song and Fine-Grained Music Style Transfer with Just One Transformer VAE🎶<br>pdf: <a href="https://t.co/9L5bZELLje">https://t.co/9L5bZELLje</a><br>abs: <a href="https://t.co/m18poSImxf">https://t.co/m18poSImxf</a><br>project page: <a href="https://t.co/rvQ5psMsY7">https://t.co/rvQ5psMsY7</a><br><br>outperforms RNN based prior art on numerous widely-used metrics for style transfer tasks <a href="https://t.co/O9eOjVn9hJ">pic.twitter.com/O9eOjVn9hJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391926443983228928?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Sampling-Frequency-Independent Audio Source Separation Using Convolution  Layer Based on Impulse Invariant Method

Koichi Saito, Tomohiko Nakamura, Kohei Yatabe, Yuma Koizumi, Hiroshi Saruwatari

- retweets: 341, favorites: 116 (05/12/2021 08:56:21)

- links: [abs](https://arxiv.org/abs/2105.04079) | [pdf](https://arxiv.org/pdf/2105.04079)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Audio source separation is often used as preprocessing of various applications, and one of its ultimate goals is to construct a single versatile model capable of dealing with the varieties of audio signals. Since sampling frequency, one of the audio signal varieties, is usually application specific, the preceding audio source separation model should be able to deal with audio signals of all sampling frequencies specified in the target applications. However, conventional models based on deep neural networks (DNNs) are trained only at the sampling frequency specified by the training data, and there are no guarantees that they work with unseen sampling frequencies. In this paper, we propose a convolution layer capable of handling arbitrary sampling frequencies by a single DNN. Through music source separation experiments, we show that the introduction of the proposed layer enables a conventional audio source separation model to consistently work with even unseen sampling frequencies.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">サンプリング周波数に依存しない畳み込み層を提案して音源分離しました！アナログフィルタのパラメータを学習してます！<a href="https://t.co/uLi2CVzQQA">https://t.co/uLi2CVzQQA</a> <a href="https://t.co/COL84Vo839">pic.twitter.com/COL84Vo839</a></p>&mdash; 矢田部浩平 (@yatabe_) <a href="https://twitter.com/yatabe_/status/1391942550743785472?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. ReadTwice: Reading Very Large Documents with Memories

Yury Zemlyanskiy, Joshua Ainslie, Michiel de Jong, Philip Pham, Ilya Eckstein, Fei Sha

- retweets: 301, favorites: 147 (05/12/2021 08:56:21)

- links: [abs](https://arxiv.org/abs/2105.04241) | [pdf](https://arxiv.org/pdf/2105.04241)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Knowledge-intensive tasks such as question answering often require assimilating information from different sections of large inputs such as books or article collections. We propose ReadTwuce, a simple and effective technique that combines several strengths of prior approaches to model long-range dependencies with Transformers. The main idea is to read text in small segments, in parallel, summarizing each segment into a memory table to be used in a second read of the text. We show that the method outperforms models of comparable size on several question answering (QA) datasets and sets a new state of the art on the challenging NarrativeQA task, with questions about entire books. Source code and pre-trained checkpoints for ReadTwice can be found at https://goo.gle/research-readtwice.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">READTWICE: Reading Very Large Documents with Memories<br><br>READTWICE reads text in small segments, in parallel, summarizing each segment into a memory table to<br>be used in a second read of the text. <br><br>The method sets a new SotA on NarrativeQA.<a href="https://t.co/3hpLnUiTiL">https://t.co/3hpLnUiTiL</a> <a href="https://t.co/9sJsVDjsHR">pic.twitter.com/9sJsVDjsHR</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1391929412656926724?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Forsage: Anatomy of a Smart-Contract Pyramid Scheme

Tyler Kell, Haaroon Yousaf, Sarah Allen, Sarah Meiklejohn, Ari Juels

- retweets: 306, favorites: 50 (05/12/2021 08:56:21)

- links: [abs](https://arxiv.org/abs/2105.04380) | [pdf](https://arxiv.org/pdf/2105.04380)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Pyramid schemes are investment scams in which top-level participants in a hierarchical network recruit and profit from an expanding base of defrauded newer participants. Pyramid schemes have existed for over a century, but there have been no in-depth studies of their dynamics and communities because of the opacity of participants' transactions.   In this paper, we present an empirical study of Forsage, a pyramid scheme implemented as a smart contract and at its peak one of the largest consumers of resources in Ethereum. As a smart contract, Forsage makes its (byte)code and all of its transactions visible on the blockchain. We take advantage of this unprecedented transparency to gain insight into the mechanics, impact on participants, and evolution of Forsage.   We quantify the (multi-million-dollar) gains of top-level participants as well as the losses of the vast majority (around 88%) of users. We analyze Forsage code both manually and using a purpose-built transaction simulator to uncover the complex mechanics of the scheme. Through complementary study of promotional videos and social media, we show how Forsage promoters have leveraged the unique features of smart contracts to lure users with false claims of trustworthiness and profitability, and how Forsage activity is concentrated within a small number of national communities.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New IC3 research: &quot;Forsage: Anatomy of a Smart-Contract Pyramid Scheme&quot; <br><br>by Tyler Kell (<a href="https://twitter.com/relyt29?ref_src=twsrc%5Etfw">@relyt29</a>), Haaroon Yousaf (<a href="https://twitter.com/Haaroony?ref_src=twsrc%5Etfw">@Haaroony</a>), Sarah Allen (<a href="https://twitter.com/4SarahAllen?ref_src=twsrc%5Etfw">@4SarahAllen</a>), Sarah Meiklejohn, Ari Juels (<a href="https://twitter.com/AriJuels?ref_src=twsrc%5Etfw">@AriJuels</a>) <br><br>Full paper: <a href="https://t.co/6bUh4pcrfX">https://t.co/6bUh4pcrfX</a>   <br><br>1/8 <a href="https://t.co/GmXH78t9i9">pic.twitter.com/GmXH78t9i9</a></p>&mdash; IC3 (@initc3org) <a href="https://twitter.com/initc3org/status/1392188153209950210?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Conformer: Local Features Coupling Global Representations for Visual  Recognition

Zhiliang Peng, Wei Huang, Shanzhi Gu, Lingxi Xie, Yaowei Wang, Jianbin Jiao, Qixiang Ye

- retweets: 240, favorites: 65 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.03889) | [pdf](https://arxiv.org/pdf/2105.03889)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Within Convolutional Neural Network (CNN), the convolution operations are good at extracting local features but experience difficulty to capture global representations. Within visual transformer, the cascaded self-attention modules can capture long-distance feature dependencies but unfortunately deteriorate local feature details. In this paper, we propose a hybrid network structure, termed Conformer, to take advantage of convolutional operations and self-attention mechanisms for enhanced representation learning. Conformer roots in the Feature Coupling Unit (FCU), which fuses local features and global representations under different resolutions in an interactive fashion. Conformer adopts a concurrent structure so that local features and global representations are retained to the maximum extent. Experiments show that Conformer, under the comparable parameter complexity, outperforms the visual transformer (DeiT-B) by 2.3% on ImageNet. On MSCOCO, it outperforms ResNet-101 by 3.7% and 3.6% mAPs for object detection and instance segmentation, respectively, demonstrating the great potential to be a general backbone network. Code is available at https://github.com/pengzhiliang/Conformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Conformer: Local Features Coupling Global Representations for Visual Recognition<br>pdf: <a href="https://t.co/eVn6PqF6ru">https://t.co/eVn6PqF6ru</a><br>abs: <a href="https://t.co/jdmDtu1B2N">https://t.co/jdmDtu1B2N</a><br>github: <a href="https://t.co/TZYZFMZACF">https://t.co/TZYZFMZACF</a><br><br>dual backbone to combining CNN with visual transformer <a href="https://t.co/l04ujH73oL">pic.twitter.com/l04ujH73oL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391933629274673155?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. A unifying tutorial on Approximate Message Passing

Oliver Y. Feng, Ramji Venkataramanan, Cynthia Rush, Richard J. Samworth

- retweets: 210, favorites: 92 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.02180) | [pdf](https://arxiv.org/pdf/2105.02180)
- [math.ST](https://arxiv.org/list/math.ST/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Over the last decade or so, Approximate Message Passing (AMP) algorithms have become extremely popular in various structured high-dimensional statistical problems. The fact that the origins of these techniques can be traced back to notions of belief propagation in the statistical physics literature lends a certain mystique to the area for many statisticians. Our goal in this work is to present the main ideas of AMP from a statistical perspective, to illustrate the power and flexibility of the AMP framework. Along the way, we strengthen and unify many of the results in the existing literature.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Please check out this approximate message passing tutorial just posted: <a href="https://t.co/UdNo532yNX">https://t.co/UdNo532yNX</a>. I had so much fun writing this with my truly amazing <a href="https://twitter.com/Cambridge_Uni?ref_src=twsrc%5Etfw">@Cambridge_Uni</a> colleagues Oliver Feng, Ramji Venkataramanan, &amp; Richard Samworth. Feedback, especially from AMP folks, is welcome.</p>&mdash; Cindy Rush (@CindyRush) <a href="https://twitter.com/CindyRush/status/1390322808752902148?ref_src=twsrc%5Etfw">May 6, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Which transformer architecture fits my data? A vocabulary bottleneck in  self-attention

Noam Wies, Yoav Levine, Daniel Jannai, Amnon Shashua

- retweets: 213, favorites: 82 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.03928) | [pdf](https://arxiv.org/pdf/2105.03928)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

After their successful debut in natural language processing, Transformer architectures are now becoming the de-facto standard in many domains. An obstacle for their deployment over new modalities is the architectural configuration: the optimal depth-to-width ratio has been shown to dramatically vary across data types (e.g., $10$x larger over images than over language). We theoretically predict the existence of an embedding rank bottleneck that limits the contribution of self-attention width to the Transformer expressivity. We thus directly tie the input vocabulary size and rank to the optimal depth-to-width ratio, since a small vocabulary size or rank dictates an added advantage of depth over width. We empirically demonstrate the existence of this bottleneck and its implications on the depth-to-width interplay of Transformer architectures, linking the architecture variability across domains to the often glossed-over usage of different vocabulary sizes or embedding ranks in different domains. As an additional benefit, our rank bottlenecking framework allows us to identify size redundancies of $25\%-50\%$ in leading NLP models such as ALBERT and T5.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Which Transformer architecture fits my data? A vocabulary bottleneck in self-attention<br><br>Investigates the depth-to-width interplay of Transformer<br>architectures across domains and when either dimension becomes a bottleneck to the optimal performance. <a href="https://t.co/DN0PryXJdt">https://t.co/DN0PryXJdt</a> <a href="https://t.co/SCuAaRP8IR">pic.twitter.com/SCuAaRP8IR</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1391927070222000129?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Which Transformer architecture fits my data? A vocabulary bottleneck in self-attention<br>pdf: <a href="https://t.co/HvNH2C6Toc">https://t.co/HvNH2C6Toc</a><br>abs: <a href="https://t.co/sLUrP8pd06">https://t.co/sLUrP8pd06</a><br><br>our rank bottlenecking framework allows us to identify size redundancies of 25% − 50% in leading<br>NLP models such as ALBERT and T5 <a href="https://t.co/ntR3mvJQxN">pic.twitter.com/ntR3mvJQxN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391925063201480707?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Optimization of Graph Neural Networks: Implicit Acceleration by Skip  Connections and More Depth

Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, Kenji Kawaguchi

- retweets: 119, favorites: 112 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.04550) | [pdf](https://arxiv.org/pdf/2105.04550)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Graph Neural Networks (GNNs) have been studied from the lens of expressive power and generalization. However, their optimization properties are less well understood. We take the first step towards analyzing GNN training by studying the gradient dynamics of GNNs. First, we analyze linearized GNNs and prove that despite the non-convexity of training, convergence to a global minimum at a linear rate is guaranteed under mild assumptions that we validate on real-world graphs. Second, we study what may affect the GNNs' training speed. Our results show that the training of GNNs is implicitly accelerated by skip connections, more depth, and/or a good label distribution. Empirical results confirm that our theoretical results for linearized GNNs align with the training behavior of nonlinear GNNs. Our results provide the first theoretical support for the success of GNNs with skip connections in terms of optimization, and suggest that deep GNNs with skip connections would be promising in practice.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our following 3 papers on graph learning will appear at <a href="https://twitter.com/hashtag/ICML2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2021</a>. Details will follow!<br><br>Convergence and implicit acceleration of GNNs<a href="https://t.co/oBayNAWaFF">https://t.co/oBayNAWaFF</a><br><br>GraphNorm for accelerating training <a href="https://t.co/2yPo9vJoD1">https://t.co/2yPo9vJoD1</a><br><br>Graph adversarial networks (GAL)<a href="https://t.co/hMKhdVRx3w">https://t.co/hMKhdVRx3w</a></p>&mdash; Keyulu Xu (@KeyuluXu) <a href="https://twitter.com/KeyuluXu/status/1391937736945319944?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Societal Biases in Language Generation: Progress and Challenges

Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, Nanyun Peng

- retweets: 156, favorites: 45 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.04054) | [pdf](https://arxiv.org/pdf/2105.04054)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Technology for language generation has advanced rapidly, spurred by advancements in pre-training large models on massive amounts of data and the need for intelligent agents to communicate in a natural manner. While techniques can effectively generate fluent text, they can also produce undesirable societal biases that can have a disproportionately negative impact on marginalized populations. Language generation presents unique challenges in terms of direct user interaction and the structure of decoding techniques. To better understand these challenges, we present a survey on societal biases in language generation, focusing on how techniques contribute to biases and on progress towards bias analysis and mitigation. Motivated by a lack of studies on biases from decoding techniques, we also conduct experiments to quantify the effects of these techniques. By further discussing general trends and open challenges, we call to attention promising directions for research and the importance of fairness and inclusivity considerations for language generation applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Having trouble keeping track of progress and challenges for biases in NLG? We’ve written a survey on societal biases in language generation 😉👉 <a href="https://t.co/T3vz1dDGUI">https://t.co/T3vz1dDGUI</a><br><br>w/<a href="https://twitter.com/kaiwei_chang?ref_src=twsrc%5Etfw">@kaiwei_chang</a> <a href="https://twitter.com/natarajan_prem?ref_src=twsrc%5Etfw">@natarajan_prem</a> <a href="https://twitter.com/VioletNPeng?ref_src=twsrc%5Etfw">@VioletNPeng</a> <a href="https://twitter.com/hashtag/ACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021</a></p>&mdash; Emily Sheng (@ewsheng) <a href="https://twitter.com/ewsheng/status/1392175214591496194?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. FNet: Mixing Tokens with Fourier Transforms

James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon

- retweets: 100, favorites: 70 (05/12/2021 08:56:22)

- links: [abs](https://arxiv.org/abs/2105.03824) | [pdf](https://arxiv.org/pdf/2105.03824)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We show that Transformer encoder architectures can be massively sped up, with limited accuracy costs, by replacing the self-attention sublayers with simple linear transformations that "mix" input tokens. These linear transformations, along with simple nonlinearities in feed-forward layers, are sufficient to model semantic relationships in several text classification tasks. Perhaps most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform achieves 92% of the accuracy of BERT on the GLUE benchmark, but pre-trains and runs up to seven times faster on GPUs and twice as fast on TPUs. The resulting model, which we name FNet, scales very efficiently to long inputs, matching the accuracy of the most accurate "efficient" Transformers on the Long Range Arena benchmark, but training and running faster across all sequence lengths on GPUs and relatively shorter sequence lengths on TPUs. Finally, FNet has a light memory footprint and is particularly efficient at smaller model sizes: for a fixed speed and accuracy budget, small FNet models outperform Transformer counterparts.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FNet: Mixing Tokens with Fourier Transforms<br>pdf: <a href="https://t.co/3RrCP8MQ2T">https://t.co/3RrCP8MQ2T</a><br>abs: <a href="https://t.co/NENrCjRwBg">https://t.co/NENrCjRwBg</a><br>Transformer encoder architectures massively sped up, with limited accuracy costs, by replacing self-attention sublayers with simple linear transformations<br>that “mix” input tokens <a href="https://t.co/g2VfmEVZzQ">pic.twitter.com/g2VfmEVZzQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391923242982596608?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Poolingformer: Long Document Modeling with Pooling Attention

Hang Zhang, Yeyun Gong, Yelong Shen, Weisheng Li, Jiancheng Lv, Nan Duan, Weizhu Chen

- retweets: 100, favorites: 38 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.04371) | [pdf](https://arxiv.org/pdf/2105.04371)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

In this paper, we introduce a two-level attention schema, Poolingformer, for long document modeling. Its first level uses a smaller sliding window pattern to aggregate information from neighbors. Its second level employs a larger window to increase receptive fields with pooling attention to reduce both computational cost and memory consumption. We first evaluate Poolingformer on two long sequence QA tasks: the monolingual NQ and the multilingual TyDi QA. Experimental results show that Poolingformer sits atop three official leaderboards measured by F1, outperforming previous state-of-the-art models by 1.9 points (79.8 vs. 77.9) on NQ long answer, 1.9 points (79.5 vs. 77.6) on TyDi QA passage answer, and 1.6 points (67.6 vs. 66.0) on TyDi QA minimal answer. We further evaluate Poolingformer on a long sequence summarization task. Experimental results on the arXiv benchmark continue to demonstrate its superior performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Poolingformer: Long Document Modeling with Pooling Attention🌊<br>pdf: <a href="https://t.co/z6BZ9AGhw4">https://t.co/z6BZ9AGhw4</a><br>abs: <a href="https://t.co/iOZyB9JWca">https://t.co/iOZyB9JWca</a><br><br>two-level attention model for long sequence modeling with linear complexity. SOTA on long-document QA and superior performance on long-document summarization <a href="https://t.co/FnWALLRJBM">pic.twitter.com/FnWALLRJBM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391962895664173056?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Deep Neural Networks as Point Estimates for Deep Gaussian Processes

Vincent Dutordoir, James Hensman, Mark van der Wilk, Carl Henrik Ek, Zoubin Ghahramani, Nicolas Durrande

- retweets: 72, favorites: 47 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.04504) | [pdf](https://arxiv.org/pdf/2105.04504)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep Gaussian processes (DGPs) have struggled for relevance in applications due to the challenges and cost associated with Bayesian inference. In this paper we propose a sparse variational approximation for DGPs for which the approximate posterior mean has the same mathematical structure as a Deep Neural Network (DNN). We make the forward pass through a DGP equivalent to a ReLU DNN by finding an interdomain transformation that represents the GP posterior mean as a sum of ReLU basis functions. This unification enables the initialisation and training of the DGP as a neural network, leveraging the well established practice in the deep learning community, and so greatly aiding the inference task. The experiments demonstrate improved accuracy and faster training compared to current DGP methods, while retaining favourable predictive uncertainties.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper out where we show how to make a forward pass through a Deep GP equivalent to a ReLU DNN. Another step towards unifying DGPs and DNNs.<br> <a href="https://t.co/PfxewnOPAo">https://t.co/PfxewnOPAo</a></p>&mdash; Vincent Dutordoir (@vdutor) <a href="https://twitter.com/vdutor/status/1392187171319205890?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Continual Learning via Bit-Level Information Preserving

Yujun Shi, Li Yuan, Yunpeng Chen, Jiashi Feng

- retweets: 72, favorites: 41 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.04444) | [pdf](https://arxiv.org/pdf/2105.04444)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Continual learning tackles the setting of learning different tasks sequentially. Despite the lots of previous solutions, most of them still suffer significant forgetting or expensive memory cost. In this work, targeted at these problems, we first study the continual learning process through the lens of information theory and observe that forgetting of a model stems from the loss of \emph{information gain} on its parameters from the previous tasks when learning a new task. From this viewpoint, we then propose a novel continual learning approach called Bit-Level Information Preserving (BLIP) that preserves the information gain on model parameters through updating the parameters at the bit level, which can be conveniently implemented with parameter quantization. More specifically, BLIP first trains a neural network with weight quantization on the new incoming task and then estimates information gain on each parameter provided by the task data to determine the bits to be frozen to prevent forgetting. We conduct extensive experiments ranging from classification tasks to reinforcement learning tasks, and the results show that our method produces better or on par results comparing to previous state-of-the-arts. Indeed, BLIP achieves close to zero forgetting while only requiring constant memory overheads throughout continual learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Continual Learning via Bit-Level Information Preserving<br>pdf: <a href="https://t.co/BQaqCJOH7P">https://t.co/BQaqCJOH7P</a><br>abs: <a href="https://t.co/KBncVCCTN6">https://t.co/KBncVCCTN6</a><br>github: <a href="https://t.co/MOtsQ70Ofb">https://t.co/MOtsQ70Ofb</a> <a href="https://t.co/Xj9Ljy2elJ">pic.twitter.com/Xj9Ljy2elJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391942861734744071?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Pareto-Optimal Quantized ResNet Is Mostly 4-bit

AmirAli Abdolrashidi, Lisa Wang, Shivani Agrawal, Jonathan Malmaud, Oleg Rybakov, Chas Leichner, Lukasz Lew

- retweets: 80, favorites: 31 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.03536) | [pdf](https://arxiv.org/pdf/2105.03536)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Quantization has become a popular technique to compress neural networks and reduce compute cost, but most prior work focuses on studying quantization without changing the network size. Many real-world applications of neural networks have compute cost and memory budgets, which can be traded off with model quality by changing the number of parameters. In this work, we use ResNet as a case study to systematically investigate the effects of quantization on inference compute cost-quality tradeoff curves. Our results suggest that for each bfloat16 ResNet model, there are quantized models with lower cost and higher accuracy; in other words, the bfloat16 compute cost-quality tradeoff curve is Pareto-dominated by the 4-bit and 8-bit curves, with models primarily quantized to 4-bit yielding the best Pareto curve. Furthermore, we achieve state-of-the-art results on ImageNet for 4-bit ResNet-50 with quantization-aware training, obtaining a top-1 eval accuracy of 77.09%. We demonstrate the regularizing effect of quantization by measuring the generalization gap. The quantization method we used is optimized for practicality: It requires little tuning and is designed with hardware capabilities in mind. Our work motivates further research into optimal numeric formats for quantization, as well as the development of machine learning accelerators supporting these formats. As part of this work, we contribute a quantization library written in JAX, which is open-sourced at https://github.com/google-research/google-research/tree/master/aqt.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pareto-Optimal Quantized ResNet Is Mostly 4-bit<br>pdf: <a href="https://t.co/OG7kg8Py3g">https://t.co/OG7kg8Py3g</a><br>abs: <a href="https://t.co/AO4zIDWoM5">https://t.co/AO4zIDWoM5</a><br>github: <a href="https://t.co/mtCTXWaoOq">https://t.co/mtCTXWaoOq</a><br><br>sota results on ImageNet for 4-bit ResNet-50 with quantization-aware training, obtaining a top-1 eval accuracy of 77.09% <a href="https://t.co/hFCgF7rWqL">pic.twitter.com/hFCgF7rWqL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391932338196697089?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. You Only Learn One Representation: Unified Network for Multiple Tasks

Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao

- retweets: 72, favorites: 37 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.04206) | [pdf](https://arxiv.org/pdf/2105.04206)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

People ``understand'' the world via vision, hearing, tactile, and also the past experience. Human experience can be learned through normal learning (we call it explicit knowledge), or subconsciously (we call it implicit knowledge). These experiences learned through normal learning or subconsciously will be encoded and stored in the brain. Using these abundant experience as a huge database, human beings can effectively process data, even they were unseen beforehand. In this paper, we propose a unified network to encode implicit knowledge and explicit knowledge together, just like the human brain can learn knowledge from normal learning as well as subconsciousness learning. The unified network can generate a unified representation to simultaneously serve various tasks. We can perform kernel space alignment, prediction refinement, and multi-task learning in a convolutional neural network. The results demonstrate that when implicit knowledge is introduced into the neural network, it benefits the performance of all tasks. We further analyze the implicit representation learnt from the proposed unified network, and it shows great capability on catching the physical meaning of different tasks. The source code of this work is at : https://github.com/WongKinYiu/yolor.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You Only Learn One Representation: Unified Network for Multiple Tasks<br>pdf: <a href="https://t.co/5SLswTtbWA">https://t.co/5SLswTtbWA</a><br>abs: <a href="https://t.co/yC0SCoIAGt">https://t.co/yC0SCoIAGt</a><br><br>a unified network that integrates implicit knowledge and explicit knowledge, effective for multi-task learning <a href="https://t.co/v1eE8R8oTN">pic.twitter.com/v1eE8R8oTN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1391935112703582220?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Boltzmann machines as two-dimensional tensor networks

Sujie Li, Feng Pan, Pengfei Zhou, Pan Zhang

- retweets: 30, favorites: 48 (05/12/2021 08:56:23)

- links: [abs](https://arxiv.org/abs/2105.04130) | [pdf](https://arxiv.org/pdf/2105.04130)
- [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent) | [quant-ph](https://arxiv.org/list/quant-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Restricted Boltzmann machines (RBM) and deep Boltzmann machines (DBM) are important models in machine learning, and recently found numerous applications in quantum many-body physics. We show that there are fundamental connections between them and tensor networks. In particular, we demonstrate that any RBM and DBM can be exactly represented as a two-dimensional tensor network. This representation gives an understanding of the expressive power of RBM and DBM using entanglement structures of the tensor networks, also provides an efficient tensor network contraction algorithm for the computing partition function of RBM and DBM. Using numerical experiments, we demonstrate that the proposed algorithm is much more accurate than the state-of-the-art machine learning methods in estimating the partition function of restricted Boltzmann machines and deep Boltzmann machines, and have potential applications in training deep Boltzmann machines for general machine learning tasks.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr"><a href="https://twitter.com/hashtag/%E3%82%AD%E3%83%A3%E3%83%AB%E3%81%A1%E3%82%83%E3%82%93%E3%81%AEquantph%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF?src=hash&amp;ref_src=twsrc%5Etfw">#キャルちゃんのquantphチェック</a><br>制限ボルツマンマシン(RBM)と深層ボルツマンマシン(DBM)が2次元のtensor networkで正確に表現できると判明。この表現によりRBMとDBMの表現力の理解と、RBMとDBMの分配関数計算のための効率的なtensor network収縮アルゴリズムを提供できる。<a href="https://t.co/7DJ62nTqd6">https://t.co/7DJ62nTqd6</a> <a href="https://t.co/JbrQMMm7KE">pic.twitter.com/JbrQMMm7KE</a></p>&mdash; キャルちゃん、🇺🇸移住10ヶ月目。 (@tweet_nakasho) <a href="https://twitter.com/tweet_nakasho/status/1392133985199788033?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. FlingBot: The Unreasonable Effectiveness of Dynamic Manipulation for  Cloth Unfolding

Huy Ha, Shuran Song

- retweets: 42, favorites: 35 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.03655) | [pdf](https://arxiv.org/pdf/2105.03655)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

High-velocity dynamic actions (e.g., fling or throw) play a crucial role in our every-day interaction with deformable objects by improving our efficiency and effectively expanding our physical reach range. Yet, most prior works have tackled cloth manipulation using exclusively single-arm quasi-static actions, which requires a large number of interactions for challenging initial cloth configurations and strictly limits the maximum cloth size by the robot's reach range. In this work, we demonstrate the effectiveness of dynamic flinging actions for cloth unfolding. We propose a self-supervised learning framework, FlingBot, that learns how to unfold a piece of fabric from arbitrary initial configurations using a pick, stretch, and fling primitive for a dual-arm setup from visual observations. The final system achieves over 80\% coverage within 3 actions on novel cloths, can unfold cloths larger than the system's reach range, and generalizes to T-shirts despite being trained on only rectangular cloths. We also finetuned FlingBot on a real-world dual-arm robot platform, where it increased the cloth coverage 3.6 times more than the quasi-static baseline did. The simplicity of FlingBot combined with its superior performance over quasi-static baselines demonstrates the effectiveness of dynamic actions for deformable object manipulation. The project video is available at $\href{https://youtu.be/T4tDy5y_6ZM}{here}$.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FlingBot: The Unreasonable Effectiveness of Dynamic Manipulation for Cloth Unfolding<br>pdf: <a href="https://t.co/CSXMOdaphG">https://t.co/CSXMOdaphG</a><br>abs: <a href="https://t.co/60yCoqal5A">https://t.co/60yCoqal5A</a><br>project page: <a href="https://t.co/6t8khKBsYt">https://t.co/6t8khKBsYt</a> <a href="https://t.co/X4vCwA0Mqe">pic.twitter.com/X4vCwA0Mqe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1392001389103140865?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. Stronger Privacy for Federated Collaborative Filtering with Implicit  Feedback

Lorenzo Minto, Moritz Haller, Hammed Haddadi, Benjamin Livshits

- retweets: 42, favorites: 32 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.03941) | [pdf](https://arxiv.org/pdf/2105.03941)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

Recommender systems are commonly trained on centrally collected user interaction data like views or clicks. This practice however raises serious privacy concerns regarding the recommender's collection and handling of potentially sensitive data. Several privacy-aware recommender systems have been proposed in recent literature, but comparatively little attention has been given to systems at the intersection of implicit feedback and privacy. To address this shortcoming, we propose a practical federated recommender system for implicit data under user-level local differential privacy (LDP). The privacy-utility trade-off is controlled by parameters $\epsilon$ and $k$, regulating the per-update privacy budget and the number of $\epsilon$-LDP gradient updates sent by each user respectively. To further protect the user's privacy, we introduce a proxy network to reduce the fingerprinting surface by anonymizing and shuffling the reports before forwarding them to the recommender. We empirically demonstrate the effectiveness of our framework on the MovieLens dataset, achieving up to Hit Ratio with K=10 (HR@10) 0.68 on 50k users with 5k items. Even on the full dataset, we show that it is possible to achieve reasonable utility with HR@10>0.5 without compromising user privacy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Some of our of work on privacy-preserving ML at Brave is being unveiled today: a practical federated recommender system for implicit data under local differential privacy <a href="https://t.co/xvZbgNS2M1">https://t.co/xvZbgNS2M1</a></p>&mdash; Ben Livshits (@convoluted_code) <a href="https://twitter.com/convoluted_code/status/1392041983154262017?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 23. Reinforcement learning of rare diffusive dynamics

Avishek Das, Dominic C. Rose, Juan P. Garrahan, David T. Limmer

- retweets: 16, favorites: 48 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.04321) | [pdf](https://arxiv.org/pdf/2105.04321)
- [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

We present a method to probe rare molecular dynamics trajectories directly using reinforcement learning. We consider trajectories that are conditioned to transition between regions of configuration space in finite time, like those relevant in the study of reactive events, as well as trajectories exhibiting rare fluctuations of time-integrated quantities in the long time limit, like those relevant in the calculation of large deviation functions. In both cases, reinforcement learning techniques are used to optimize an added force that minimizes the Kullback-Leibler divergence between the conditioned trajectory ensemble and a driven one. Under the optimized added force, the system evolves the rare fluctuation as a typical one, affording a variational estimate of its likelihood in the original trajectory ensemble. Low variance gradients employing value functions are proposed to increase the convergence of the optimal force. The method we develop employing these gradients leads to efficient and accurate estimates of both the optimal force and the likelihood of the rare event for a variety of model systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Super excited to see Avishek&#39;s latest with Dom Rose and JPG, where we take a foray into RL to generate rare trajectories with the correct dynamical weights- solving  difficult problems with metastability and multiple paths  <a href="https://t.co/C6uzACOihf">https://t.co/C6uzACOihf</a> <a href="https://twitter.com/pleplostelous?ref_src=twsrc%5Etfw">@pleplostelous</a> <a href="https://twitter.com/UCB_Chemistry?ref_src=twsrc%5Etfw">@UCB_Chemistry</a> <a href="https://t.co/AG8d52GGMO">pic.twitter.com/AG8d52GGMO</a></p>&mdash; limmerlab (@limmerlab) <a href="https://twitter.com/limmerlab/status/1391924443404926979?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. Opening the Blackbox: Accelerating Neural Differential Equations by  Regularizing Internal Solver Heuristics

Avik Pal, Yingbo Ma, Viral Shah, Christopher Rackauckas

- retweets: 30, favorites: 30 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.03918) | [pdf](https://arxiv.org/pdf/2105.03918)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.NA](https://arxiv.org/list/math.NA/recent)

Democratization of machine learning requires architectures that automatically adapt to new problems. Neural Differential Equations (NDEs) have emerged as a popular modeling framework by removing the need for ML practitioners to choose the number of layers in a recurrent model. While we can control the computational cost by choosing the number of layers in standard architectures, in NDEs the number of neural network evaluations for a forward pass can depend on the number of steps of the adaptive ODE solver. But, can we force the NDE to learn the version with the least steps while not increasing the training cost? Current strategies to overcome slow prediction require high order automatic differentiation, leading to significantly higher training time. We describe a novel regularization method that uses the internal cost heuristics of adaptive differential equation solvers combined with discrete adjoint sensitivities to guide the training process towards learning NDEs that are easier to solve. This approach opens up the blackbox numerical analysis behind the differential equation solver's algorithm and directly uses its local error estimates and stiffness heuristics as cheap and accurate cost estimates. We incorporate our method without any change in the underlying NDE framework and show that our method extends beyond Ordinary Differential Equations to accommodate Neural Stochastic Differential Equations. We demonstrate how our approach can halve the prediction time and, unlike other methods which can increase the training time by an order of magnitude, we demonstrate similar reduction in training times. Together this showcases how the knowledge embedded within state-of-the-art equation solvers can be used to enhance machine learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New ICML Paper: Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics w/ <a href="https://twitter.com/YingboMa1?ref_src=twsrc%5Etfw">@YingboMa1</a> <a href="https://twitter.com/Viral_B_Shah?ref_src=twsrc%5Etfw">@Viral_B_Shah</a> <a href="https://twitter.com/ChrisRackauckas?ref_src=twsrc%5Etfw">@ChrisRackauckas</a> <br><br>Arxiv: <a href="https://t.co/ACoDNOadiH">https://t.co/ACoDNOadiH</a><a href="https://twitter.com/JuliaLanguage?ref_src=twsrc%5Etfw">@JuliaLanguage</a> / <a href="https://twitter.com/SciML_Org?ref_src=twsrc%5Etfw">@SciML_Org</a> code: <a href="https://t.co/GXZgbuE1N1">https://t.co/GXZgbuE1N1</a><br><br>[1/4] <a href="https://t.co/fywkulvMrZ">pic.twitter.com/fywkulvMrZ</a></p>&mdash; Avik Pal (@avikpal1410) <a href="https://twitter.com/avikpal1410/status/1392000179394420736?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 25. Neuroscience-inspired perception-action in robotics: applying active  inference for state estimation, control and self-perception

Pablo Lanillos, Marcel van Gerven

- retweets: 42, favorites: 13 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.04261) | [pdf](https://arxiv.org/pdf/2105.04261)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent)

Unlike robots, humans learn, adapt and perceive their bodies by interacting with the world. Discovering how the brain represents the body and generates actions is of major importance for robotics and artificial intelligence. Here we discuss how neuroscience findings open up opportunities to improve current estimation and control algorithms in robotics. In particular, how active inference, a mathematical formulation of how the brain resists a natural tendency to disorder, provides a unified recipe to potentially solve some of the major challenges in robotics, such as adaptation, robustness, flexibility, generalization and safe interaction. This paper summarizes some experiments and lessons learned from developing such a computational model on real embodied platforms, i.e., humanoid and industrial robots. Finally, we showcase the limitations and challenges that we are still facing to give robots human-like perception




# 26. Simplicial contagion in temporal higher-order networks

Sandeep Chowdhary, Aanjaneya Kumar, Giulia Cencetti, Iacopo Iacopini, Federico Battiston

- retweets: 25, favorites: 27 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.04455) | [pdf](https://arxiv.org/pdf/2105.04455)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Complex networks represent the natural backbone to study epidemic processes in populations of interacting individuals. Such a modeling framework, however, is naturally limited to pairwise interactions, making it less suitable to properly describe social contagion, where individuals acquire new norms or ideas after simultaneous exposure to multiple sources of infections. Simplicial contagion has been proposed as an alternative framework where simplices are used to encode group interactions of any order. The presence of higher-order interactions leads to explosive epidemic transitions and bistability which cannot be obtained when only dyadic ties are considered. In particular, critical mass effects can emerge even for infectivity values below the standard pairwise epidemic threshold, where the size of the initial seed of infectious nodes determines whether the system would eventually fall in the endemic or the healthy state. Here we extend simplicial contagion to time-varying networks, where pairwise and higher-order simplices can be created or destroyed over time. By following a microscopic Markov chain approach, we find that the same seed of infectious nodes might or might not lead to an endemic stationary state, depending on the temporal properties of the underlying network structure, and show that persistent temporal interactions anticipate the onset of the endemic state in finite-size systems. We characterize this behavior on higher-order networks with a prescribed temporal correlation between consecutive interactions and on heterogeneous simplicial complexes, showing that temporality again limits the effect of higher-order spreading, but in a less pronounced way than for homogeneous structures. Our work suggests the importance of incorporating temporality, a realistic feature of many real-world systems, into the investigation of dynamical processes beyond pairwise interactions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Simplicial contagion in temporal higher-order networks. (arXiv:2105.04455v1 [physics.soc-ph]) <a href="https://t.co/Ay2YXHHM6f">https://t.co/Ay2YXHHM6f</a></p>&mdash; NetScience (@net_science) <a href="https://twitter.com/net_science/status/1391955445447675907?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 27. Scalable Projection-Free Optimization

Mingrui Zhang

- retweets: 1, favorites: 49 (05/12/2021 08:56:24)

- links: [abs](https://arxiv.org/abs/2105.03527) | [pdf](https://arxiv.org/pdf/2105.03527)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

As a projection-free algorithm, Frank-Wolfe (FW) method, also known as conditional gradient, has recently received considerable attention in the machine learning community. In this dissertation, we study several topics on the FW variants for scalable projection-free optimization.   We first propose 1-SFW, the first projection-free method that requires only one sample per iteration to update the optimization variable and yet achieves the best known complexity bounds for convex, non-convex, and monotone DR-submodular settings. Then we move forward to the distributed setting, and develop Quantized Frank-Wolfe (QFW), a general communication-efficient distributed FW framework for both convex and non-convex objective functions. We study the performance of QFW in two widely recognized settings: 1) stochastic optimization and 2) finite-sum optimization. Finally, we propose Black-Box Continuous Greedy, a derivative-free and projection-free algorithm, that maximizes a monotone continuous DR-submodular function over a bounded convex body in Euclidean space.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mingrui Zhang, another amazing PhD student from our group, just graduated. His research pushed the boundaries of projection-free optimization. His thesis &quot;Scalable Projection-Free Optimization&quot; can be found on arXiv: <a href="https://t.co/RyUxjX5VmA">https://t.co/RyUxjX5VmA</a> <a href="https://t.co/ta0WJKokOI">pic.twitter.com/ta0WJKokOI</a></p>&mdash; Amin Karbasi (@aminkarbasi) <a href="https://twitter.com/aminkarbasi/status/1392161491999203329?ref_src=twsrc%5Etfw">May 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



