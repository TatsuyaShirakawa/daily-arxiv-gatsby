---
title: Hot Papers 2020-12-04
date: 2020-12-05T10:55:51.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-04"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-04"
socialImage: "/media/flying-marine.jpg"

---

# 1. pixelNeRF: Neural Radiance Fields from One or Few Images

Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa

- retweets: 894, favorites: 245 (12/05/2020 10:55:51)

- links: [abs](https://arxiv.org/abs/2012.02190) | [pdf](https://arxiv.org/pdf/2012.02190)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose pixelNeRF, a learning framework that predicts a continuous neural scene representation conditioned on one or few input images. The existing approach for constructing neural radiance fields involves optimizing the representation to every scene independently, requiring many calibrated views and significant compute time. We take a step towards resolving these shortcomings by introducing an architecture that conditions a NeRF on image inputs in a fully convolutional manner. This allows the network to be trained across multiple scenes to learn a scene prior, enabling it to perform novel view synthesis in a feed-forward manner from a sparse set of views (as few as one). Leveraging the volume rendering approach of NeRF, our model can be trained directly from images with no explicit 3D supervision. We conduct extensive experiments on ShapeNet benchmarks for single image novel view synthesis tasks with held-out objects as well as entire unseen categories. We further demonstrate the flexibility of pixelNeRF by demonstrating it on multi-object ShapeNet scenes and real scenes from the DTU dataset. In all cases, pixelNeRF outperforms current state-of-the-art baselines for novel view synthesis and single image 3D reconstruction. For the video and code, please visit the project website: https://alexyu.net/pixelnerf

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">pixelNeRF: Neural Radiance Fields from One or Few Images<br>pdf: <a href="https://t.co/TNfwoaEpkY">https://t.co/TNfwoaEpkY</a><br>abs: <a href="https://t.co/7GShHQY6Hc">https://t.co/7GShHQY6Hc</a><br>project page: <a href="https://t.co/Kw9MZIWnWh">https://t.co/Kw9MZIWnWh</a><br>github: <a href="https://t.co/Ic17pGCRBd">https://t.co/Ic17pGCRBd</a> <a href="https://t.co/IIYkyE3hMV">pic.twitter.com/IIYkyE3hMV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334688113701122051?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">pixelNeRF: Neural Radiance Fields from One or Few Images <a href="https://t.co/fyM7RZ8wmF">https://t.co/fyM7RZ8wmF</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <br><br>Project page: <a href="https://t.co/qaYxgANcD1">https://t.co/qaYxgANcD1</a> <a href="https://t.co/TFUK3AOaQs">pic.twitter.com/TFUK3AOaQs</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1334698368807919616?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Neural Deformation Graphs for Globally-consistent Non-rigid  Reconstruction

Aljaž Božič, Pablo Palafox, Michael Zollhöfer, Justus Thies, Angela Dai, Matthias Nießner

- retweets: 805, favorites: 174 (12/05/2020 10:55:52)

- links: [abs](https://arxiv.org/abs/2012.01451) | [pdf](https://arxiv.org/pdf/2012.01451)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce Neural Deformation Graphs for globally-consistent deformation tracking and 3D reconstruction of non-rigid objects. Specifically, we implicitly model a deformation graph via a deep neural network. This neural deformation graph does not rely on any object-specific structure and, thus, can be applied to general non-rigid deformation tracking. Our method globally optimizes this neural graph on a given sequence of depth camera observations of a non-rigidly moving object. Based on explicit viewpoint consistency as well as inter-frame graph and surface consistency constraints, the underlying network is trained in a self-supervised fashion. We additionally optimize for the geometry of the object with an implicit deformable multi-MLP shape representation. Our approach does not assume sequential input data, thus enabling robust tracking of fast motions or even temporally disconnected recordings. Our experiments demonstrate that our Neural Deformation Graphs outperform state-of-the-art non-rigid reconstruction approaches both qualitatively and quantitatively, with 64% improved reconstruction and 62% improved deformation tracking performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check <a href="https://twitter.com/BozicAljaz?ref_src=twsrc%5Etfw">@BozicAljaz</a> work on &quot;Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction&quot;<br><br>Pretty cool idea: an implicit neural network models a deformation graph which is globally optimized with Adam!<br><br>Video: <a href="https://t.co/AgGNXzUME9">https://t.co/AgGNXzUME9</a><br>Paper: <a href="https://t.co/E95Pu2Ug4o">https://t.co/E95Pu2Ug4o</a> <a href="https://t.co/GaDnfi3pEg">pic.twitter.com/GaDnfi3pEg</a></p>&mdash; Matthias Niessner (@MattNiessner) <a href="https://twitter.com/MattNiessner/status/1334783398477500416?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction<br>pdf: <a href="https://t.co/MbzSxrKHvG">https://t.co/MbzSxrKHvG</a><br>abs: <a href="https://t.co/iYW4rNuKGG">https://t.co/iYW4rNuKGG</a> <a href="https://t.co/P1g7tWQdsq">pic.twitter.com/P1g7tWQdsq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334692961469259776?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Entropy and Diversity: The Axiomatic Approach

Tom Leinster

- retweets: 136, favorites: 98 (12/05/2020 10:55:52)

- links: [abs](https://arxiv.org/abs/2012.02113) | [pdf](https://arxiv.org/pdf/2012.02113)
- [q-bio.PE](https://arxiv.org/list/q-bio.PE/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [math.CA](https://arxiv.org/list/math.CA/recent) | [math.CT](https://arxiv.org/list/math.CT/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

This book brings new mathematical rigour to the ongoing vigorous debate on how to quantify biological diversity. The question "what is diversity?" has surprising mathematical depth, and breadth too: this book involves parts of mathematics ranging from information theory, functional equations and probability theory to category theory, geometric measure theory and number theory. It applies the power of the axiomatic method to a biological problem of pressing concern, but the new concepts and theorems are also motivated from a purely mathematical perspective.   The main narrative thread requires no more than an undergraduate course in analysis. No familiarity with entropy or diversity is assumed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Breaking radio silence to post this: a new 450 page book by Tom Leinster: &quot;Entropy and Diversity: The Axiomatic Approach&quot; (!)<a href="https://t.co/zbQS10rbyP">https://t.co/zbQS10rbyP</a> <a href="https://t.co/zBiNCAOlOx">pic.twitter.com/zBiNCAOlOx</a></p>&mdash; theHigherGeometer needs a job next year (@HigherGeometer) <a href="https://twitter.com/HigherGeometer/status/1334786021020499968?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tom Leinster just dropped a 450 page book on the arXiv: “Entropy and diversity: The axiomatic approach”. I saw his talk on this, measures of diversity in ecosystems, at the *first* Applied Category Theory conference in 2018<a href="https://t.co/Ds0jrDp5HU">https://t.co/Ds0jrDp5HU</a> <a href="https://t.co/v0Me5tw09U">pic.twitter.com/v0Me5tw09U</a></p>&mdash; julesh (@_julesh_) <a href="https://twitter.com/_julesh_/status/1334836727802982400?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. AutoInt: Automatic Integration for Fast Neural Volume Rendering

David B. Lindell, Julien N. P. Martel, Gordon Wetzstein

- retweets: 66, favorites: 105 (12/05/2020 10:55:52)

- links: [abs](https://arxiv.org/abs/2012.01714) | [pdf](https://arxiv.org/pdf/2012.01714)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Numerical integration is a foundational technique in scientific computing and is at the core of many computer vision applications. Among these applications, implicit neural volume rendering has recently been proposed as a new paradigm for view synthesis, achieving photorealistic image quality. However, a fundamental obstacle to making these methods practical is the extreme computational and memory requirements caused by the required volume integrations along the rendered rays during training and inference. Millions of rays, each requiring hundreds of forward passes through a neural network are needed to approximate those integrations with Monte Carlo sampling. Here, we propose automatic integration, a new framework for learning efficient, closed-form solutions to integrals using implicit neural representation networks. For training, we instantiate the computational graph corresponding to the derivative of the implicit neural representation. The graph is fitted to the signal to integrate. After optimization, we reassemble the graph to obtain a network that represents the antiderivative. By the fundamental theorem of calculus, this enables the calculation of any definite integral in two evaluations of the network. Using this approach, we demonstrate a greater than 10x improvement in computation requirements, enabling fast neural volume rendering.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Closed form integration for rendering NeRFs, akin to the integral image trick. Very thoughtful, and 10x faster! <a href="https://t.co/3bGgaY6iIu">https://t.co/3bGgaY6iIu</a></p>&mdash; Jon Barron (@jon_barron) <a href="https://twitter.com/jon_barron/status/1334691010866114560?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AutoInt: Automatic Integration for Fast Neural Volume Rendering<br>pdf: <a href="https://t.co/C4TTPOeeYF">https://t.co/C4TTPOeeYF</a><br>abs: <a href="https://t.co/n0S5B04L0s">https://t.co/n0S5B04L0s</a> <a href="https://t.co/L29wYgkace">pic.twitter.com/L29wYgkace</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334683855169740801?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Ethical Testing in the Real World: Evaluating Physical Testing of  Adversarial Machine Learning

Kendra Albert, Maggie Delano, Jonathon Penney, Afsaneh Rigot, Ram Shankar Siva Kumar

- retweets: 100, favorites: 40 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.02048) | [pdf](https://arxiv.org/pdf/2012.02048)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

This paper critically assesses the adequacy and representativeness of physical domain testing for various adversarial machine learning (ML) attacks against computer vision systems involving human subjects. Many papers that deploy such attacks characterize themselves as "real world." Despite this framing, however, we found the physical or real-world testing conducted was minimal, provided few details about testing subjects and was often conducted as an afterthought or demonstration. Adversarial ML research without representative trials or testing is an ethical, scientific, and health/safety issue that can cause real harms. We introduce the problem and our methodology, and then critique the physical domain testing methodologies employed by papers in the field. We then explore various barriers to more inclusive physical testing in adversarial ML and offer recommendations to improve such testing notwithstanding these challenges.




# 6. Full-Resolution Correspondence Learning for Image Translation

Xingran Zhou, Bo Zhang, Ting Zhang, Pan Zhang, Jianmin Bao, Dong Chen, Zhongfei Zhang, Fang Wen

- retweets: 90, favorites: 33 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.02047) | [pdf](https://arxiv.org/pdf/2012.02047)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present the full-resolution correspondence learning for cross-domain images, which aids image translation. We adopt a hierarchical strategy that uses the correspondence from coarse level to guide the finer levels. In each hierarchy, the correspondence can be efficiently computed via PatchMatch that iteratively leverages the matchings from the neighborhood. Within each PatchMatch iteration, the ConvGRU module is employed to refine the current correspondence considering not only the matchings of larger context but also the historic estimates. The proposed GRU-assisted PatchMatch is fully differentiable and highly efficient. When jointly trained with image translation, full-resolution semantic correspondence can be established in an unsupervised manner, which in turn facilitates the exemplar-based image translation. Experiments on diverse translation tasks show our approach performs considerably better than state-of-the-arts on producing high-resolution images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Full-Resolution Correspondence Learning for Image Translation<br>pdf: <a href="https://t.co/wCe4BebKcC">https://t.co/wCe4BebKcC</a><br>abs: <a href="https://t.co/HY1HJMhoIK">https://t.co/HY1HJMhoIK</a> <a href="https://t.co/hbPPqX9HIV">pic.twitter.com/hbPPqX9HIV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1334721200170340352?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Quantum learning algorithms imply circuit lower bounds

Srinivasan Arunachalam, Alex B. Grilo, Tom Gur, Igor C. Oliveira, Aarthi Sundaram

- retweets: 50, favorites: 72 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.01920) | [pdf](https://arxiv.org/pdf/2012.01920)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We establish the first general connection between the design of quantum algorithms and circuit lower bounds. Specifically, let $\mathfrak{C}$ be a class of polynomial-size concepts, and suppose that $\mathfrak{C}$ can be PAC-learned with membership queries under the uniform distribution with error $1/2 - \gamma$ by a time $T$ quantum algorithm. We prove that if $\gamma^2 \cdot T \ll 2^n/n$, then $\mathsf{BQE} \nsubseteq \mathfrak{C}$, where $\mathsf{BQE} = \mathsf{BQTIME}[2^{O(n)}]$ is an exponential-time analogue of $\mathsf{BQP}$. This result is optimal in both $\gamma$ and $T$, since it is not hard to learn any class $\mathfrak{C}$ of functions in (classical) time $T = 2^n$ (with no error), or in quantum time $T = \mathsf{poly}(n)$ with error at most $1/2 - \Omega(2^{-n/2})$ via Fourier sampling. In other words, even a marginal improvement on these generic learning algorithms would lead to major consequences in complexity theory.   Our proof builds on several works in learning theory, pseudorandomness, and computational complexity, and crucially, on a connection between non-trivial classical learning algorithms and circuit lower bounds established by Oliveira and Santhanam (CCC 2017). Extending their approach to quantum learning algorithms turns out to create significant challenges. To achieve that, we show among other results how pseudorandom generators imply learning-to-lower-bound connections in a generic fashion, construct the first conditional pseudorandom generator secure against uniform quantum computations, and extend the local list-decoding algorithm of Impagliazzo, Jaiswal, Kabanets and Wigderson (SICOMP 2010) to quantum circuits via a delicate analysis. We believe that these contributions are of independent interest and might find other applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Quantum learning algorithms imply circuit lower bounds!<br>I’m very excited about this joint work with <a href="https://twitter.com/Qsrinivasan_1?ref_src=twsrc%5Etfw">@Qsrinivasan_1</a><br>,<a href="https://twitter.com/abgrilo?ref_src=twsrc%5Etfw">@abgrilo</a>, Igor Oliveira, and Aarthi Sundaram:⁰<a href="https://t.co/ieWUcHUSyr">https://t.co/ieWUcHUSyr</a><br><br>So what is it all about? 1/3</p>&mdash; Tom Gur (@TomGur) <a href="https://twitter.com/TomGur/status/1334883031308521473?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Patch2Pix: Epipolar-Guided Pixel-Level Correspondences

Qunjie Zhou, Torsten Sattler, Laura Leal-Taixe

- retweets: 37, favorites: 55 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.01909) | [pdf](https://arxiv.org/pdf/2012.01909)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep learning has been applied to a classical matching pipeline which typically involves three steps: (i) local feature detection and description, (ii) feature matching, and (iii) outlier rejection. Recently emerged correspondence networks propose to perform those steps inside a single network but suffer from low matching resolution due to the memory bottleneck. In this work, we propose a new perspective to estimate correspondences in a detect-to-refine manner, where we first predict patch-level match proposals and then refine them. We present a novel refinement network Patch2Pix that refines match proposals by regressing pixel-level matches from the local regions defined by those proposals and jointly rejecting outlier matches with confidence scores, which is weakly supervised to learn correspondences that are consistent with the epipolar geometry of an input image pair. We show that our refinement network significantly improves the performance of correspondence networks on image matching, homography estimation, and localization tasks. In addition, we show that our learned refinement generalizes to fully-supervised methods without re-training, which leads us to state-of-the-art localization performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Patch2Pix: Epipolar-Guided Pixel-Level Correspondences<br>Qunjie Zhou, <a href="https://twitter.com/SattlerTorsten?ref_src=twsrc%5Etfw">@SattlerTorsten</a>, <a href="https://twitter.com/lealtaixe?ref_src=twsrc%5Etfw">@lealtaixe</a> <a href="https://t.co/2PGiMdZmRs">https://t.co/2PGiMdZmRs</a><br>tl;dr: Get imprecise correspondences first, refine them later<br>-&gt; PROFIT!<br>A bit similar to &quot;Multi-View Optimization of Local Feature Geometry&quot; by <a href="https://twitter.com/mihaidusmanu?ref_src=twsrc%5Etfw">@mihaidusmanu</a> <a href="https://t.co/qEg9Aa6B0t">pic.twitter.com/qEg9Aa6B0t</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1334824536571785217?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. COVID-19 Contact Tracing and Privacy: A Longitudinal Study of Public  Opinion

Lucy Simko, Jack Lucas Chang, Maggie Jiang, Ryan Calo, Franziska Roesner, Tadayoshi Kohno

- retweets: 72, favorites: 15 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.01553) | [pdf](https://arxiv.org/pdf/2012.01553)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

There is growing use of technology-enabled contact tracing, the process of identifying potentially infected COVID-19 patients by notifying all recent contacts of an infected person. Governments, technology companies, and research groups alike have been working towards releasing smartphone apps, using IoT devices, and distributing wearable technology to automatically track "close contacts" and identify prior contacts in the event an individual tests positive. However, there has been significant public discussion about the tensions between effective technology-based contact tracing and the privacy of individuals. To inform this discussion, we present the results of seven months of online surveys focused on contact tracing and privacy, each with 100 participants. Our first surveys were on April 1 and 3, before the first peak of the virus in the US, and we continued to conduct the surveys weekly for 10 weeks (through June), and then fortnightly through November, adding topical questions to reflect current discussions about contact tracing and COVID-19. Our results present the diversity of public opinion and can inform policy makers, technologists, researchers, and public health experts on whether and how to leverage technology to reduce the spread of COVID-19, while considering potential privacy concerns. We are continuing to conduct longitudinal measurements and will update this report over time; citations to this version of the report should reference Report Version 2.0, December 2, 2020.




# 10. Towards Part-Based Understanding of RGB-D Scans

Alexey Bokhovkin, Vladislav Ishimtsev, Emil Bogomolov, Denis Zorin, Alexey Artemov, Evgeny Burnaev, Angela Dai

- retweets: 42, favorites: 30 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.02094) | [pdf](https://arxiv.org/pdf/2012.02094)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent advances in 3D semantic scene understanding have shown impressive progress in 3D instance segmentation, enabling object-level reasoning about 3D scenes; however, a finer-grained understanding is required to enable interactions with objects and their functional understanding. Thus, we propose the task of part-based scene understanding of real-world 3D environments: from an RGB-D scan of a scene, we detect objects, and for each object predict its decomposition into geometric part masks, which composed together form the complete geometry of the observed object. We leverage an intermediary part graph representation to enable robust completion as well as building of part priors, which we use to construct the final part mask predictions. Our experiments demonstrate that guiding part understanding through part graph to part prior-based predictions significantly outperforms alternative approaches to the task of semantic part completion.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out <a href="https://twitter.com/ABokhovkin?ref_src=twsrc%5Etfw">@ABokhovkin</a>&#39;s work on part understanding of 3D scans -- predicting geometrically complete part decompositions of the detected objects in a scene!<br><br>Video: <a href="https://t.co/dP1bZW6EyU">https://t.co/dP1bZW6EyU</a><br>Paper: <a href="https://t.co/xh2ncddOTO">https://t.co/xh2ncddOTO</a> <a href="https://t.co/Cr4cmMrEWI">pic.twitter.com/Cr4cmMrEWI</a></p>&mdash; Angela Dai (@angelaqdai) <a href="https://twitter.com/angelaqdai/status/1334898769079492612?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Classification and reconstruction of optical quantum states with deep  neural networks

Shahnawaz Ahmed, Carlos Sánchez Muñoz, Franco Nori, Anton Frisk Kockum

- retweets: 20, favorites: 38 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.02185) | [pdf](https://arxiv.org/pdf/2012.02185)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We apply deep-neural-network-based techniques to quantum state classification and reconstruction. We demonstrate high classification accuracies and reconstruction fidelities, even in the presence of noise and with little data. Using optical quantum states as examples, we first demonstrate how convolutional neural networks (CNNs) can successfully classify several types of states distorted by, e.g., additive Gaussian noise or photon loss. We further show that a CNN trained on noisy inputs can learn to identify the most important regions in the data, which potentially can reduce the cost of tomography by guiding adaptive data collection. Secondly, we demonstrate reconstruction of quantum-state density matrices using neural networks that incorporate quantum-physics knowledge. The knowledge is implemented as custom neural-network layers that convert outputs from standard feedforward neural networks to valid descriptions of quantum states. Any standard feed-forward neural-network architecture can be adapted for quantum state tomography (QST) with our method. We present further demonstrations of our proposed [arXiv:2008.03240] QST technique with conditional generative adversarial networks (QST-CGAN). We motivate our choice of a learnable loss function within an adversarial framework by demonstrating that the QST-CGAN outperforms, across a range of scenarios, generative networks trained with standard loss functions. For pure states with additive or convolutional Gaussian noise, the QST-CGAN is able to adapt to the noise and reconstruct the underlying state. The QST-CGAN reconstructs states using up to two orders of magnitude fewer iterative steps than a standard iterative maximum likelihood (iMLE) method. Further, the QST-CGAN can reconstruct both pure and mixed states from two orders of magnitude fewer randomly chosen data points than iMLE.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/4) New preprint: Classification and reconstruction of optical quantum states with deep neural networks (<a href="https://t.co/DBEqZ8YxoL">https://t.co/DBEqZ8YxoL</a>). Recognize and reconstruct states with various types of noise, faster and with much fewer data points. <a href="https://t.co/ZcV9yNN7XA">pic.twitter.com/ZcV9yNN7XA</a></p>&mdash; Shahnawaz Ahmed (@quantshah) <a href="https://twitter.com/quantshah/status/1334829245936361473?ref_src=twsrc%5Etfw">December 4, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Phonetic Posteriorgrams based Many-to-Many Singing Voice Conversion via  Adversarial Training

Haohan Guo, Heng Lu, Na Hu, Chunlei Zhang, Shan Yang, Lei Xie, Dan Su, Dong Yu

- retweets: 22, favorites: 33 (12/05/2020 10:55:53)

- links: [abs](https://arxiv.org/abs/2012.01837) | [pdf](https://arxiv.org/pdf/2012.01837)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper describes an end-to-end adversarial singing voice conversion (EA-SVC) approach. It can directly generate arbitrary singing waveform by given phonetic posteriorgram (PPG) representing content, F0 representing pitch, and speaker embedding representing timbre, respectively. Proposed system is composed of three modules: generator $G$, the audio generation discriminator $D_{A}$, and the feature disentanglement discriminator $D_F$. The generator $G$ encodes the features in parallel and inversely transforms them into the target waveform. In order to make timbre conversion more stable and controllable, speaker embedding is further decomposed to the weighted sum of a group of trainable vectors representing different timbre clusters. Further, to realize more robust and accurate singing conversion, disentanglement discriminator $D_F$ is proposed to remove pitch and timbre related information that remains in the encoded PPG. Finally, a two-stage training is conducted to keep a stable and effective adversarial training process. Subjective evaluation results demonstrate the effectiveness of our proposed methods. Proposed system outperforms conventional cascade approach and the WaveNet based end-to-end approach in terms of both singing quality and singer similarity. Further objective analysis reveals that the model trained with the proposed two-stage training strategy can produce a smoother and sharper formant which leads to higher audio quality.



