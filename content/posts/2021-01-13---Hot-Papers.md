---
title: Hot Papers 2021-01-13
date: 2021-01-14T10:53:51.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-13"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-13"
socialImage: "/media/flying-marine.jpg"

---

# 1. A Bayesian neural network predicts the dissolution of compact planetary  systems

Miles Cranmer, Daniel Tamayo, Hanno Rein, Peter Battaglia, Samuel Hadden, Philip J. Armitage, Shirley Ho, David N. Spergel

- retweets: 1406, favorites: 212 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04117) | [pdf](https://arxiv.org/pdf/2101.04117)
- [astro-ph.EP](https://arxiv.org/list/astro-ph.EP/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Despite over three hundred years of effort, no solutions exist for predicting when a general planetary configuration will become unstable. We introduce a deep learning architecture to push forward this problem for compact systems. While current machine learning algorithms in this area rely on scientist-derived instability metrics, our new technique learns its own metrics from scratch, enabled by a novel internal structure inspired from dynamics theory. Our Bayesian neural network model can accurately predict not only if, but also when a compact planetary system with three or more planets will go unstable. Our model, trained directly from short N-body time series of raw orbital elements, is more than two orders of magnitude more accurate at predicting instability times than analytical estimators, while also reducing the bias of existing machine learning algorithms by nearly a factor of three. Despite being trained on compact resonant and near-resonant three-planet configurations, the model demonstrates robust generalization to both non-resonant and higher multiplicity configurations, in the latter case outperforming models fit to that specific set of integrations. The model computes instability estimates up to five orders of magnitude faster than a numerical integrator, and unlike previous efforts provides confidence intervals on its predictions. Our inference model is publicly available in the SPOCK package, with training code open-sourced.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very excited to present our new work: we adapt Bayesian neural networks to predict the dissolution of compact planetary systems, a variant of the three-body problem!<br><br>Blogpost/code: <a href="https://t.co/sNKv1Xduff">https://t.co/sNKv1Xduff</a><br>Paper: <a href="https://t.co/bNsN8VqULq">https://t.co/bNsN8VqULq</a><br>API: <a href="https://t.co/wPNMmTOOiq">https://t.co/wPNMmTOOiq</a><br><br>Thread: 👇 <a href="https://t.co/2HG75x3vcU">pic.twitter.com/2HG75x3vcU</a></p>&mdash; Miles Cranmer (@MilesCranmer) <a href="https://twitter.com/MilesCranmer/status/1349417287804194821?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Benchmarking Simulation-Based Inference

Jan-Matthis Lueckmann, Jan Boelts, David S. Greenberg, Pedro J. Gonçalves, Jakob H. Macke

- retweets: 176, favorites: 41 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04653) | [pdf](https://arxiv.org/pdf/2101.04653)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent advances in probabilistic modelling have led to a large number of simulation-based inference algorithms which do not require numerical evaluation of likelihoods. However, a public benchmark with appropriate performance metrics for such 'likelihood-free' algorithms has been lacking. This has made it difficult to compare algorithms and identify their strengths and weaknesses. We set out to fill this gap: We provide a benchmark with inference tasks and suitable performance metrics, with an initial selection of algorithms including recent approaches employing neural networks and classical Approximate Bayesian Computation methods. We found that the choice of performance metric is critical, that even state-of-the-art algorithms have substantial room for improvement, and that sequential estimation improves sample efficiency. Neural network-based approaches generally exhibit better performance, but there is no uniformly best algorithm. We provide practical advice and highlight the potential of the benchmark to diagnose problems and improve algorithms. The results can be explored interactively on a companion website. All code is open source, making it possible to contribute further benchmark tasks and inference algorithms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our new paper on benchmarking simulation-based inference!<br><br>Check out our interactive website w/short summary<br><br>With <a href="https://twitter.com/janfiete?ref_src=twsrc%5Etfw">@janfiete</a>, <a href="https://twitter.com/dvdgbg?ref_src=twsrc%5Etfw">@dvdgbg</a>, <a href="https://twitter.com/ppjgoncalves?ref_src=twsrc%5Etfw">@ppjgoncalves</a> and <a href="https://twitter.com/jakhmack?ref_src=twsrc%5Etfw">@jakhmack</a><br><br>Paper: <a href="https://t.co/bClvCjPhOG">https://t.co/bClvCjPhOG</a><br>Code: <a href="https://t.co/61bKxN9u78">https://t.co/61bKxN9u78</a><a href="https://t.co/mEjQOUD7WA">https://t.co/mEjQOUD7WA</a></p>&mdash; jan-matthis (@janmatthis) <a href="https://twitter.com/janmatthis/status/1349265140735238144?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. From Tinkering to Engineering: Measurements in Tensorflow Playground

Henrik Hoeiness, Axel Harstad, Gerald Friedland

- retweets: 68, favorites: 40 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04141) | [pdf](https://arxiv.org/pdf/2101.04141)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this article, we present an extension of the Tensorflow Playground, called Tensorflow Meter (short TFMeter). TFMeter is an interactive neural network architecting tool that allows the visual creation of different architectures of neural networks. In addition to its ancestor, the playground, our tool shows information-theoretic measurements while constructing, training, and testing the network. As a result, each change results in a change in at least one of the measurements, providing for a better engineering intuition of what different architectures are able to learn. The measurements are derived from various places in the literature. In this demo, we describe our web application that is available online at http://tfmeter.icsi.berkeley.edu/ and argue that in the same way that the original Playground is meant to build an intuition about neural networks, our extension educates users on available measurements, which we hope will ultimately improve experimental design and reproducibility in the field.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">From Tinkering to Engineering: Measurements in Tensorflow Playground<br>pdf: <a href="https://t.co/Ine0ZOcpKX">https://t.co/Ine0ZOcpKX</a><br>abs: <a href="https://t.co/uC6CTcJRqW">https://t.co/uC6CTcJRqW</a><br>web demo: <a href="https://t.co/2sGZsi0nVe">https://t.co/2sGZsi0nVe</a> <a href="https://t.co/Hfqphn6rwD">pic.twitter.com/Hfqphn6rwD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1349179549985214469?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Mixup Without Hesitation

Hao Yu, Huanyu Wang, Jianxin Wu

- retweets: 42, favorites: 41 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04342) | [pdf](https://arxiv.org/pdf/2101.04342)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Mixup linearly interpolates pairs of examples to form new samples, which is easy to implement and has been shown to be effective in image classification tasks. However, there are two drawbacks in mixup: one is that more training epochs are needed to obtain a well-trained model; the other is that mixup requires tuning a hyper-parameter to gain appropriate capacity but that is a difficult task. In this paper, we find that mixup constantly explores the representation space, and inspired by the exploration-exploitation dilemma in reinforcement learning, we propose mixup Without hesitation (mWh), a concise, effective, and easy-to-use training algorithm. We show that mWh strikes a good balance between exploration and exploitation by gradually replacing mixup with basic data augmentation. It can achieve a strong baseline with less training time than original mixup and without searching for optimal hyper-parameter, i.e., mWh acts as mixup without hesitation. mWh can also transfer to CutMix, and gain consistent improvement on other machine learning and computer vision tasks such as object detection. Our code is open-source and available at https://github.com/yuhao318/mwh

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mixup Without Hesitation<a href="https://t.co/QXSgMvHeDR">https://t.co/QXSgMvHeDR</a><a href="https://t.co/pvYqVSYJTi">https://t.co/pvYqVSYJTi</a> <a href="https://t.co/6k637FD7Kg">pic.twitter.com/6k637FD7Kg</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1349322699391176705?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Interpretable discovery of new semiconductors with machine learning

Hitarth Choubisa, Petar Todorović, Joao M. Pina, Darshan H. Parmar, Ziliang Li, Oleksandr Voznyy, Isaac Tamblyn, Edward Sargent

- retweets: 30, favorites: 52 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04383) | [pdf](https://arxiv.org/pdf/2101.04383)
- [cond-mat.mtrl-sci](https://arxiv.org/list/cond-mat.mtrl-sci/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine learning models of materials$^{1-5}$ accelerate discovery compared to ab initio methods: deep learning models now reproduce density functional theory (DFT)-calculated results at one hundred thousandths of the cost of DFT$^{6}$. To provide guidance in experimental materials synthesis, these need to be coupled with an accurate yet effective search algorithm and training data consistent with experimental observations. Here we report an evolutionary algorithm powered search which uses machine-learned surrogate models trained on high-throughput hybrid functional DFT data benchmarked against experimental bandgaps: Deep Adaptive Regressive Weighted Intelligent Network (DARWIN). The strategy enables efficient search over the materials space of ~10$^8$ ternaries and 10$^{11}$ quaternaries$^{7}$ for candidates with target properties. It provides interpretable design rules, such as our finding that the difference in the electronegativity between the halide and B-site cation being a strong predictor of ternary structural stability. As an example, when we seek UV emission, DARWIN predicts K$_2$CuX$_3$ (X = Cl, Br) as a promising materials family, based on its electronegativity difference. We synthesized and found these materials to be stable, direct bandgap UV emitters. The approach also allows knowledge distillation for use by humans.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to introduce DARWIN<a href="https://t.co/H1jKtkZksM">https://t.co/H1jKtkZksM</a><br><br>Using evolutionary search, we provide an interpretable and accurate approach to designing new materials<br><br>Hitarth Choubisa &amp; Petar Todorović did great work building this Deep Adaptive Regressive Weighted Intelligent Network</p>&mdash; Isaac Tamblyn (@itamblyn) <a href="https://twitter.com/itamblyn/status/1349180846935306245?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Categories of Nets

John C. Baez, Fabrizio Genovese, Jade Master, Michael Shulman

- retweets: 30, favorites: 46 (01/14/2021 10:53:51)

- links: [abs](https://arxiv.org/abs/2101.04238) | [pdf](https://arxiv.org/pdf/2101.04238)
- [math.CT](https://arxiv.org/list/math.CT/recent) | [cs.FL](https://arxiv.org/list/cs.FL/recent)

We present a unified framework for Petri nets and various variants, such as pre-nets and Kock's whole-grain Petri nets. Our framework is based on a less well-studied notion that we call $\Sigma$-nets, which allow finer control over whether tokens are treated using the collective or individual token philosophy. We describe three forms of execution semantics in which pre-nets generate strict monoidal categories, $\Sigma$-nets (including whole-grain Petri nets) generate symmetric strict monoidal categories, and Petri nets generate commutative monoidal categories, all by left adjoint functors. We also construct adjunctions relating these categories of nets to each other, in particular showing that all kinds of net can be embedded in the unifying category of $\Sigma$-nets, in a way that commutes coherently with their execution semantics.

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">John C. Baez, Fabrizio Genovese, Jade Master, Michael Shulman: Categories of Nets <a href="https://t.co/zidrOjDdIC">https://t.co/zidrOjDdIC</a> <a href="https://t.co/Gu8ei4d68O">https://t.co/Gu8ei4d68O</a></p>&mdash; arXiv math.CT Category Theory (@mathCTbot) <a href="https://twitter.com/mathCTbot/status/1349177147722657794?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Superpixel-based Refinement for Object Proposal Generation

Christian Wilms, Simone Frintrop

- retweets: 30, favorites: 22 (01/14/2021 10:53:52)

- links: [abs](https://arxiv.org/abs/2101.04574) | [pdf](https://arxiv.org/pdf/2101.04574)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Precise segmentation of objects is an important problem in tasks like class-agnostic object proposal generation or instance segmentation. Deep learning-based systems usually generate segmentations of objects based on coarse feature maps, due to the inherent downsampling in CNNs. This leads to segmentation boundaries not adhering well to the object boundaries in the image. To tackle this problem, we introduce a new superpixel-based refinement approach on top of the state-of-the-art object proposal system AttentionMask. The refinement utilizes superpixel pooling for feature extraction and a novel superpixel classifier to determine if a high precision superpixel belongs to an object or not. Our experiments show an improvement of up to 26.0% in terms of average recall compared to original AttentionMask. Furthermore, qualitative and quantitative analyses of the segmentations reveal significant improvements in terms of boundary adherence for the proposed refinement compared to various deep learning-based state-of-the-art object proposal generation systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Superpixel-based Refinement for Object Proposal Generation<br>pdf: <a href="https://t.co/JOSl6GUStC">https://t.co/JOSl6GUStC</a><br>abs: <a href="https://t.co/gSLWQalmxM">https://t.co/gSLWQalmxM</a><br>github: <a href="https://t.co/BLw6qkwUCC">https://t.co/BLw6qkwUCC</a> <a href="https://t.co/liLog6SUEU">pic.twitter.com/liLog6SUEU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1349187726453723136?ref_src=twsrc%5Etfw">January 13, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



