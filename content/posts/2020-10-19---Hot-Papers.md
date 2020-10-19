---
title: Hot Papers 2020-10-19
date: 2020-10-20T08:55:25.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. The network structure of scientific revolutions

Harang Ju, Dale Zhou, Ann S. Blevins, David M. Lydon-Staley, Judith Kaplan, Julio R. Tuma, Danielle S. Bassett

- retweets: 5012, favorites: 338 (10/20/2020 08:55:25)

- links: [abs](https://arxiv.org/abs/2010.08381) | [pdf](https://arxiv.org/pdf/2010.08381)
- [cs.DL](https://arxiv.org/list/cs.DL/recent) | [physics.hist-ph](https://arxiv.org/list/physics.hist-ph/recent)

Philosophers of science have long postulated how collective scientific knowledge grows. Empirical validation has been challenging due to limitations in collecting and systematizing large historical records. Here, we capitalize on the largest online encyclopedia to formulate knowledge as growing networks of articles and their hyperlinked inter-relations. We demonstrate that concept networks grow not by expanding from their core but rather by creating and filling knowledge gaps, a process which produces discoveries that are more frequently awarded Nobel prizes than others. Moreover, we operationalize paradigms as network modules to reveal a temporal signature in structural stability across scientific subjects. In a network formulation of scientific discovery, data-driven conditions underlying breakthroughs depend just as much on identifying uncharted gaps as on advancing solutions within scientific communities.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new work “The network structure of scientific revolutions” out in <a href="https://t.co/VnoMhwulHC">https://t.co/VnoMhwulHC</a>. We use network science to operationalize and test philosophical theories about the development of scientific ideas on growing networks of hyperlinked articles on Wikipedia. 1/</p>&mdash; Harang Ju (@harangju) <a href="https://twitter.com/harangju/status/1318161036302757888?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How does science happen? This question has fascinated philosophers for eons. Could their theories be tested with emerging digital encyclopedias? <a href="https://twitter.com/harangju?ref_src=twsrc%5Etfw">@harangju</a> collaborates with a philosopher of science &amp; a historian of science (&amp; some other scientists) in <a href="https://t.co/38hNhC7dSp">https://t.co/38hNhC7dSp</a> <a href="https://t.co/I7aaUanxjl">https://t.co/I7aaUanxjl</a></p>&mdash; Dani Bassett (@DaniSBassett) <a href="https://twitter.com/DaniSBassett/status/1318164639390326789?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. The Deep Bootstrap: Good Online Learners are Good Offline Generalizers

Preetum Nakkiran, Behnam Neyshabur, Hanie Sedghi

- retweets: 2708, favorites: 262 (10/20/2020 08:55:26)

- links: [abs](https://arxiv.org/abs/2010.08127) | [pdf](https://arxiv.org/pdf/2010.08127)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [math.ST](https://arxiv.org/list/math.ST/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We propose a new framework for reasoning about generalization in deep learning. The core idea is to couple the Real World, where optimizers take stochastic gradient steps on the empirical loss, to an Ideal World, where optimizers take steps on the population loss. This leads to an alternate decomposition of test error into: (1) the Ideal World test error plus (2) the gap between the two worlds. If the gap (2) is universally small, this reduces the problem of generalization in offline learning to the problem of optimization in online learning. We then give empirical evidence that this gap between worlds can be small in realistic deep learning settings, in particular supervised image classification. For example, CNNs generalize better than MLPs on image distributions in the Real World, but this is "because" they optimize faster on the population loss in the Ideal World. This suggests our framework is a useful tool for understanding generalization in deep learning, and lays a foundation for future research in the area.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Deep Bootstrap: Good Online Learners are Good Offline Generalizers<a href="https://t.co/Exdpx4yGcp">https://t.co/Exdpx4yGcp</a><br>with <a href="https://twitter.com/bneyshabur?ref_src=twsrc%5Etfw">@bneyshabur</a> and <a href="https://twitter.com/HanieSedghi?ref_src=twsrc%5Etfw">@HanieSedghi</a> at Google.<br> <br>We give a promising new approach to understand generalization in DL: optimization is all you need. Feat. Vision Transformers and more.. 1/ <a href="https://t.co/8iGbFdxEJh">pic.twitter.com/8iGbFdxEJh</a></p>&mdash; Preetum Nakkiran (@PreetumNakkiran) <a href="https://twitter.com/PreetumNakkiran/status/1318007088321335297?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. For self-supervised learning, Rationality implies generalization,  provably

Yamini Bansal, Gal Kaplun, Boaz Barak

- retweets: 1069, favorites: 193 (10/20/2020 08:55:26)

- links: [abs](https://arxiv.org/abs/2010.08508) | [pdf](https://arxiv.org/pdf/2010.08508)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We prove a new upper bound on the generalization gap of classifiers that are obtained by first using self-supervision to learn a representation $r$ of the training data, and then fitting a simple (e.g., linear) classifier $g$ to the labels. Specifically, we show that (under the assumptions described below) the generalization gap of such classifiers tends to zero if $\mathsf{C}(g) \ll n$, where $\mathsf{C}(g)$ is an appropriately-defined measure of the simple classifier $g$'s complexity, and $n$ is the number of training samples. We stress that our bound is independent of the complexity of the representation $r$. We do not make any structural or conditional-independence assumptions on the representation-learning task, which can use the same training dataset that is later used for classification. Rather, we assume that the training procedure satisfies certain natural noise-robustness (adding small amount of label noise causes small degradation in performance) and rationality (getting the wrong label is not better than getting no label at all) conditions that widely hold across many standard architectures. We show that our bound is non-vacuous for many popular representation-learning based classifiers on CIFAR-10 and ImageNet, including SimCLR, AMDIM and MoCo.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper <a href="https://t.co/UTCiz1k66L">https://t.co/UTCiz1k66L</a> w Gal Kaplun &amp; <a href="https://twitter.com/boazbaraktcs?ref_src=twsrc%5Etfw">@boazbaraktcs</a>!<br><br>Recent work has focused on the &quot;deep learning generalization puzzle&quot; (highlighted by Zhang  et al <a href="https://t.co/HwlONZV7hC">https://t.co/HwlONZV7hC</a>). Since deep nets interpolate train data, the train err doesn&#39;t tell you much about test err. <a href="https://t.co/B8bu3vW3p4">https://t.co/B8bu3vW3p4</a></p>&mdash; Yamini Bansal (@whybansal) <a href="https://twitter.com/whybansal/status/1317995856256245760?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. What Can You Learn from Your Muscles? Learning Visual Representation  from Human Interactions

Kiana Ehsani, Daniel Gordon, Thomas Nguyen, Roozbeh Mottaghi, Ali Farhadi

- retweets: 146, favorites: 97 (10/20/2020 08:55:26)

- links: [abs](https://arxiv.org/abs/2010.08539) | [pdf](https://arxiv.org/pdf/2010.08539)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Learning effective representations of visual data that generalize to a variety of downstream tasks has been a long quest for computer vision. Most representation learning approaches rely solely on visual data such as images or videos. In this paper, we explore a novel approach, where we use human interaction and attention cues to investigate whether we can learn better representations compared to visual-only representations. For this study, we collect a dataset of human interactions capturing body part movements and gaze in their daily lives. Our experiments show that our self-supervised representation that encodes interaction and attention cues outperforms a visual-only state-of-the-art method MoCo (He et al., 2020), on a variety of target tasks: scene classification (semantic), action recognition (temporal), depth estimation (geometric), dynamics prediction (physics) and walkable surface estimation (affordance).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Should we learn visual representations only from visual data? In this paper, we show that other signals such as body movements and gaze improve our representation for a variety of visual tasks.<br><br>paper: <a href="https://t.co/0PEpPZZPzi">https://t.co/0PEpPZZPzi</a><br>code and data: <a href="https://t.co/OtTDWuYgRZ">https://t.co/OtTDWuYgRZ</a> <a href="https://t.co/3Z1YE8MuE1">pic.twitter.com/3Z1YE8MuE1</a></p>&mdash; Roozbeh Mottaghi (@RoozbehMottaghi) <a href="https://twitter.com/RoozbehMottaghi/status/1318271338696515584?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What Can You Learn from Your Muscles? Learning Visual Representation from Human Interactions<br>pdf: <a href="https://t.co/PnoUm41We4">https://t.co/PnoUm41We4</a><br>abs: <a href="https://t.co/OCdopatyy7">https://t.co/OCdopatyy7</a><br>github: <a href="https://t.co/C3xS0b3RjB">https://t.co/C3xS0b3RjB</a> <a href="https://t.co/tigIBPxtKL">pic.twitter.com/tigIBPxtKL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1318072852248252417?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense  Spatiotemporal Grounding

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, Jason Baldridge

- retweets: 76, favorites: 37 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.07954) | [pdf](https://arxiv.org/pdf/2010.07954)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We introduce Room-Across-Room (RxR), a new Vision-and-Language Navigation (VLN) dataset. RxR is multilingual (English, Hindi, and Telugu) and larger (more paths and instructions) than other VLN datasets. It emphasizes the role of language in VLN by addressing known biases in paths and eliciting more references to visible entities. Furthermore, each word in an instruction is time-aligned to the virtual poses of instruction creators and validators. We establish baseline scores for monolingual and multilingual settings and multitask learning when including Room-to-Room annotations. We also provide results for a model that learns from synchronized pose traces by focusing only on portions of the panorama attended to in human demonstrations. The size, scope and detail of RxR dramatically expands the frontier for research on embodied language agents in simulated, photo-realistic environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We&#39;ll be providing a test evaluation server and leaderboard. We&#39;re excited to see what the community can do with RxR!<br><br>Work with <a href="https://twitter.com/alex_y_ku?ref_src=twsrc%5Etfw">@alex_y_ku</a>, <a href="https://twitter.com/996roma?ref_src=twsrc%5Etfw">@996roma</a>, <a href="https://twitter.com/ieeugene?ref_src=twsrc%5Etfw">@ieeugene</a>, and <a href="https://twitter.com/jasonbaldridge?ref_src=twsrc%5Etfw">@jasonbaldridge</a>. To appear at <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a>.<br><br>Paper: <a href="https://t.co/TfQqsz8wE9">https://t.co/TfQqsz8wE9</a><br>Dataset: <a href="https://t.co/9FIO1wPrsT">https://t.co/9FIO1wPrsT</a></p>&mdash; Peter Anderson (@panderson_me) <a href="https://twitter.com/panderson_me/status/1318218992134377478?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Non-saturating GAN training as divergence minimization

Matt Shannon, Ben Poole, Soroosh Mariooryad, Tom Bagby, Eric Battenberg, David Kao, Daisy Stanton, RJ Skerry-Ryan

- retweets: 58, favorites: 22 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08029) | [pdf](https://arxiv.org/pdf/2010.08029)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Non-saturating generative adversarial network (GAN) training is widely used and has continued to obtain groundbreaking results. However so far this approach has lacked strong theoretical justification, in contrast to alternatives such as f-GANs and Wasserstein GANs which are motivated in terms of approximate divergence minimization. In this paper we show that non-saturating GAN training does in fact approximately minimize a particular f-divergence. We develop general theoretical tools to compare and classify f-divergences and use these to show that the new f-divergence is qualitatively similar to reverse KL. These results help to explain the high sample quality but poor diversity often observed empirically when using this scheme.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Non-saturating GAN training as divergence minimization<br>pdf: <a href="https://t.co/sCooIaWgVa">https://t.co/sCooIaWgVa</a><br>abs: <a href="https://t.co/D7aoDEMPQS">https://t.co/D7aoDEMPQS</a> <a href="https://t.co/NE2gLmU3CZ">pic.twitter.com/NE2gLmU3CZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1318037047949578240?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Why Are Convolutional Nets More Sample-Efficient than Fully-Connected  Nets?

Zhiyuan Li, Yi Zhang, Sanjeev Arora

- retweets: 30, favorites: 34 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08515) | [pdf](https://arxiv.org/pdf/2010.08515)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Convolutional neural networks often dominate fully-connected counterparts in generalization performance, especially on image classification tasks. This is often explained in terms of 'better inductive bias'. However, this has not been made mathematically rigorous, and the hurdle is that the fully connected net can always simulate the convolutional net (for a fixed task). Thus the training algorithm plays a role. The current work describes a natural task on which a provable sample complexity gap can be shown, for standard training algorithms. We construct a single natural distribution on $\mathbb{R}^d\times\{\pm 1\}$ on which any orthogonal-invariant algorithm (i.e. fully-connected networks trained with most gradient-based methods from gaussian initialization) requires $\Omega(d^2)$ samples to generalize while $O(1)$ samples suffice for convolutional architectures. Furthermore, we demonstrate a single target function, learning which on all possible distributions leads to an $O(1)$ vs $\Omega(d^2/\varepsilon)$ gap. The proof relies on the fact that SGD on fully-connected network is orthogonal equivariant. Similar results are achieved for $\ell_2$ regression and adaptive training algorithms, e.g. Adam and AdaGrad, which are only permutation equivariant.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why Are Convolutional Nets More Sample-Efficient than Fully-Connected Nets?. (arXiv:2010.08515v1 [cs.LG]) <a href="https://t.co/q47Psp44VE">https://t.co/q47Psp44VE</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1318005789265625090?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Probabilistic Programming with CuPPL

Alexander Collins, Vinod Grover

- retweets: 18, favorites: 44 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08454) | [pdf](https://arxiv.org/pdf/2010.08454)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

Probabilistic Programming Languages (PPLs) are a powerful tool in machine learning, allowing highly expressive generative models to be expressed succinctly. They couple complex inference algorithms, implemented by the language, with an expressive modelling language that allows a user to implement any computable function as the generative model. Such languages are usually implemented on top of existing high level programming languages and do not make use of hardware accelerators. PPLs that do make use of accelerators exist, but restrict the expressivity of the language in order to do so. In this paper, we present a language and toolchain that generates highly efficient code for both CPUs and GPUs. The language is functional in style, and the tool chain is built on top of LLVM. Our implementation uses de-limited continuations on CPU to perform inference, and custom CUDA codes on GPU. We obtain significant speed ups across a suite of PPL workloads, compared to other state of the art approaches on CPU. Furthermore, our compiler can also generate efficient code that runs on CUDA GPUs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper, Nvidia researchers have presented CuPPL, a probabilistic programming language that generates highly efficient code for both CPUs and CUDA GPUs.<a href="https://t.co/PmHrWYsI4z">https://t.co/PmHrWYsI4z</a> <a href="https://t.co/9lPB9n5leu">pic.twitter.com/9lPB9n5leu</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1318138247013371904?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Robust Keypoint Detection and Pose Estimation of Robot Manipulators with  Self-Occlusions via Sim-to-Real Transfer

Jingpei Lu, Florian Richter, Michael Yip

- retweets: 28, favorites: 33 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08054) | [pdf](https://arxiv.org/pdf/2010.08054)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Keypoint detection is an essential building block for many robotic applications like motion capture and pose estimation. Historically, keypoints are detected using uniquely engineered markers such as checkerboards, fiducials, or markers. More recently, deep learning methods have been explored as they have the ability to detect user-defined keypoints in a marker-less manner. However, deep neural network (DNN) detectors can have an uneven performance for different manually selected keypoints along the kinematic chain. An example of this can be found on symmetric robotic tools where DNN detectors cannot solve the correspondence problem correctly. In this work, we propose a new and autonomous way to define the keypoint locations that overcomes these challenges. The approach involves finding the optimal set of keypoints on robotic manipulators for robust visual detection. Using a robotic simulator as a medium, our algorithm utilizes synthetic data for DNN training, and the proposed algorithm is used to optimize the selection of keypoints through an iterative approach. The results show that when using the optimized keypoints, the detection performance of the DNNs improved so significantly that they can even be detected in cases of self-occlusion. We further use the optimized keypoints for real robotic applications by using domain randomization to bridge the reality gap between the simulator and the physical world. The physical world experiments show how the proposed method can be applied to the wide-breadth of robotic applications that require visual feedback, such as camera-to-robot calibration, robotic tool tracking, and whole-arm pose estimation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robust Keypoint Detection and Pose Estimation of Robot Manipulators with Self-Occlusions via Sim-to-Real Transfer<a href="https://t.co/ugyapTYA9s">https://t.co/ugyapTYA9s</a> <a href="https://t.co/WqgCsqwmFL">pic.twitter.com/WqgCsqwmFL</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1317993697103945729?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Adaptive Feature Selection for End-to-End Speech Translation

Biao Zhang, Ivan Titov, Barry Haddow, Rico Sennrich

- retweets: 30, favorites: 29 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08518) | [pdf](https://arxiv.org/pdf/2010.08518)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Information in speech signals is not evenly distributed, making it an additional challenge for end-to-end (E2E) speech translation (ST) to learn to focus on informative features. In this paper, we propose adaptive feature selection (AFS) for encoder-decoder based E2E ST. We first pre-train an ASR encoder and apply AFS to dynamically estimate the importance of each encoded speech feature to SR. A ST encoder, stacked on top of the ASR encoder, then receives the filtered features from the (frozen) ASR encoder. We take L0DROP (Zhang et al., 2020) as the backbone for AFS, and adapt it to sparsify speech features with respect to both temporal and feature dimensions. Results on LibriSpeech En-Fr and MuST-C benchmarks show that AFS facilitates learning of ST by pruning out ~84% temporal features, yielding an average translation gain of ~1.3-1.6 BLEU and a decoding speedup of ~1.4x. In particular, AFS reduces the performance gap compared to the cascade baseline, and outperforms it on LibriSpeech En-Fr with a BLEU score of 18.56 (without data augmentation)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[1/4] Excited to share our new work on improving end-to-end speech translation with adaptive feature selection, to appear at Findings of <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a>.<br><br>Joint work with <a href="https://twitter.com/iatitov?ref_src=twsrc%5Etfw">@iatitov</a>, <a href="https://twitter.com/bazril?ref_src=twsrc%5Etfw">@bazril</a> and <a href="https://twitter.com/RicoSennrich?ref_src=twsrc%5Etfw">@RicoSennrich</a>.<br><br>Paper: <a href="https://t.co/811qlbqgvg">https://t.co/811qlbqgvg</a><br>Code: <a href="https://t.co/Z29A2PZ02O">https://t.co/Z29A2PZ02O</a> <a href="https://t.co/e70AZNL6SM">pic.twitter.com/e70AZNL6SM</a></p>&mdash; Biao Zhang (@BZhangGo) <a href="https://twitter.com/BZhangGo/status/1318096399754559488?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. AAMDRL: Augmented Asset Management with Deep Reinforcement Learning

Eric Benhamou, David Saltiel, Sandrine Ungari, Abhishek Mukhopadhyay, Jamal Atif

- retweets: 52, favorites: 1 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08497) | [pdf](https://arxiv.org/pdf/2010.08497)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [q-fin.MF](https://arxiv.org/list/q-fin.MF/recent)

Can an agent learn efficiently in a noisy and self adapting environment with sequential, non-stationary and non-homogeneous observations? Through trading bots, we illustrate how Deep Reinforcement Learning (DRL) can tackle this challenge. Our contributions are threefold: (i) the use of contextual information also referred to as augmented state in DRL, (ii) the impact of a one period lag between observations and actions that is more realistic for an asset management environment, (iii) the implementation of a new repetitive train test method called walk forward analysis, similar in spirit to cross validation for time series. Although our experiment is on trading bots, it can easily be translated to other bot environments that operate in sequential environment with regime changes and noisy data. Our experiment for an augmented asset manager interested in finding the best portfolio for hedging strategies shows that AAMDRL achieves superior returns and lower risk.




# 12. How to Sell Hard Information

S. Nageeb Ali, Nima Haghpanah, Xiao Lin, Ron Siegel

- retweets: 25, favorites: 27 (10/20/2020 08:55:27)

- links: [abs](https://arxiv.org/abs/2010.08037) | [pdf](https://arxiv.org/pdf/2010.08037)
- [econ.TH](https://arxiv.org/list/econ.TH/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent)

The seller of an asset has the option to buy hard information about the value of the asset from an intermediary. The seller can then disclose the acquired information before selling the asset in a competitive market. We study how the intermediary designs and sells hard information to robustly maximize her revenue across all equilibria. Even though the intermediary could use an accurate test that reveals the asset's value, we show that robust revenue maximization leads to a noisy test with a continuum of possible scores that are distributed exponentially. In addition, the intermediary always charges the seller for disclosing the test score to the market, but not necessarily for running the test. This enables the intermediary to robustly appropriate a significant share of the surplus resulting from the asset sale even though the information generated by the test provides no social value.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited about new paper “How to Sell Hard Information” with <a href="https://twitter.com/SNageebAli?ref_src=twsrc%5Etfw">@SNageebAli</a>, Xiao Lin, Ron Siegel!<br><br>Why do information intermediaries exist?<br><br>An answer: They increase efficiency. Or at least benefit whoever is willing to pay for them.<br><br>Not necessarily!<br><br>Link: <a href="https://t.co/g0HJw41o0M">https://t.co/g0HJw41o0M</a></p>&mdash; Nima Haghpanah (@NimaHaghpanah) <a href="https://twitter.com/NimaHaghpanah/status/1318202505684258816?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for  Pairwise Sentence Scoring Tasks

Nandan Thakur, Nils Reimers, Johannes Daxenberger, Iryna Gurevych

- retweets: 14, favorites: 36 (10/20/2020 08:55:28)

- links: [abs](https://arxiv.org/abs/2010.08240) | [pdf](https://arxiv.org/pdf/2010.08240)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

There are two approaches for pairwise sentence scoring: Cross-encoders, which perform full-attention over the input pair, and Bi-encoders, which map each input independently to a dense vector space. While cross-encoders often achieve higher performance, they are too slow for many practical use cases. Bi-encoders, on the other hand, require substantial training data and fine-tuning over the target task to achieve competitive performance. We present a simple yet efficient data augmentation strategy called Augmented SBERT, where we use the cross-encoder to label a larger set of input pairs to augment the training data for the bi-encoder. We show that, in this process, selecting the sentence pairs is non-trivial and crucial for the success of the method. We evaluate our approach on multiple tasks (in-domain) as well as on a domain adaptation task. Augmented SBERT achieves an improvement of up to 6 points for in-domain and of up to 37 points for domain adaptation tasks compared to the original bi-encoder performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share my first paper <a href="https://twitter.com/UKPLab?ref_src=twsrc%5Etfw">@UKPLab</a> w/ Nils, Johannes, and Iryna Gurevych - Augmented SBERT<br><br>Wondered how to create sentence-embeddings when little or zero in-domain training data is available?<br><br>Click 👇!<br><br>Paper - <a href="https://t.co/0sF1m3E0Cr">https://t.co/0sF1m3E0Cr</a><br>Code - <a href="https://t.co/SJ87RPvIvG">https://t.co/SJ87RPvIvG</a> <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://t.co/YrP5Rz14Pc">pic.twitter.com/YrP5Rz14Pc</a></p>&mdash; Nandan Thakur (@Nthakur20) <a href="https://twitter.com/Nthakur20/status/1318231686174355457?ref_src=twsrc%5Etfw">October 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



