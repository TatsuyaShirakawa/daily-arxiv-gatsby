---
title: Hot Papers 2021-07-20
date: 2021-07-21T10:09:31.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-20"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-20"
socialImage: "/media/flying-marine.jpg"

---

# 1. Just Train Twice: Improving Group Robustness without Training Group  Information

Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn

- retweets: 5544, favorites: 350 (07/21/2021 10:09:31)

- links: [abs](https://arxiv.org/abs/2107.09044) | [pdf](https://arxiv.org/pdf/2107.09044)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Standard training via empirical risk minimization (ERM) can produce models that achieve high accuracy on average but low accuracy on certain groups, especially in the presence of spurious correlations between the input and label. Prior approaches that achieve high worst-group accuracy, like group distributionally robust optimization (group DRO) require expensive group annotations for each training point, whereas approaches that do not use such group annotations typically achieve unsatisfactory worst-group accuracy. In this paper, we propose a simple two-stage approach, JTT, that first trains a standard ERM model for several epochs, and then trains a second model that upweights the training examples that the first model misclassified. Intuitively, this upweights examples from groups on which standard ERM models perform poorly, leading to improved worst-group performance. Averaged over four image classification and natural language processing tasks with spurious correlations, JTT closes 75% of the gap in worst-group accuracy between standard ERM and group DRO, while only requiring group annotations on a small validation set in order to tune hyperparameters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Does your neural network struggle with spurious correlations?<br><br>Check out Evan’s long talk at <a href="https://twitter.com/hashtag/ICML2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2021</a> on why they should just train twice (JTT).<br><br>Paper: <a href="https://t.co/MBrgmvyqLB">https://t.co/MBrgmvyqLB</a><br>Talk: <a href="https://t.co/Xr3q0oZlR2">https://t.co/Xr3q0oZlR2</a><br>Code: <a href="https://t.co/HhPqbhXKMh">https://t.co/HhPqbhXKMh</a> <a href="https://t.co/sYrh7SwFNG">pic.twitter.com/sYrh7SwFNG</a></p>&mdash; Chelsea Finn (@chelseabfinn) <a href="https://twitter.com/chelseabfinn/status/1417322595276320769?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. YOLOX: Exceeding YOLO Series in 2021

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun

- retweets: 2129, favorites: 240 (07/21/2021 10:09:31)

- links: [abs](https://arxiv.org/abs/2107.08430) | [pdf](https://arxiv.org/pdf/2107.08430)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported. Source code is at https://github.com/Megvii-BaseDetection/YOLOX.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">YOLOX: Exceeding YOLO Series in 2021<br>pdf: <a href="https://t.co/xC1ZEPOLRW">https://t.co/xC1ZEPOLRW</a><br>abs: <a href="https://t.co/BNkflEgqaC">https://t.co/BNkflEgqaC</a><br>github: <a href="https://t.co/rym6pRl10e">https://t.co/rym6pRl10e</a> <a href="https://t.co/7Gg3ov9SUN">pic.twitter.com/7Gg3ov9SUN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1417298145663496196?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Epistemic Neural Networks

Ian Osband, Zheng Wen, Mohammad Asghari, Morteza Ibrahimi, Xiyuan Lu, Benjamin Van Roy

- retweets: 932, favorites: 127 (07/21/2021 10:09:31)

- links: [abs](https://arxiv.org/abs/2107.08924) | [pdf](https://arxiv.org/pdf/2107.08924)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce the \textit{epistemic neural network} (ENN) as an interface for uncertainty modeling in deep learning. All existing approaches to uncertainty modeling can be expressed as ENNs, and any ENN can be identified with a Bayesian neural network. However, this new perspective provides several promising directions for future research. Where prior work has developed probabilistic inference tools for neural networks; we ask instead, `which neural networks are suitable as tools for probabilistic inference?'. We propose a clear and simple metric for progress in ENNs: the KL-divergence with respect to a target distribution. We develop a computational testbed based on inference in a neural network Gaussian process and release our code as a benchmark at \url{https://github.com/deepmind/enn}. We evaluate several canonical approaches to uncertainty modeling in deep learning, and find they vary greatly in their performance. We provide insight to the sensitivity of these results and show that our metric is highly correlated with performance in sequential decision problems. Finally, we provide indications that new ENN architectures can improve performance in both the statistical quality and computational cost.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Epistemic Neural Networks<br>pdf: <a href="https://t.co/rVU20iQMbb">https://t.co/rVU20iQMbb</a><br>abs: <a href="https://t.co/t9vAPk83FZ">https://t.co/t9vAPk83FZ</a><br><br>introduce the epistemic neural network (ENN) as an interface for uncertainty modeling in deep learning <a href="https://t.co/A4VP0VvfFW">pic.twitter.com/A4VP0VvfFW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1417299613502033924?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Reasoning-Modulated Representations

Petar Veličković, Matko Bošnjak, Thomas Kipf, Alexander Lerchner, Raia Hadsell, Razvan Pascanu, Charles Blundell

- retweets: 842, favorites: 164 (07/21/2021 10:09:31)

- links: [abs](https://arxiv.org/abs/2107.08881) | [pdf](https://arxiv.org/pdf/2107.08881)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Neural networks leverage robust internal representations in order to generalise. Learning them is difficult, and often requires a large training set that covers the data distribution densely. We study a common setting where our task is not purely opaque. Indeed, very often we may have access to information about the underlying system (e.g. that observations must obey certain laws of physics) that any "tabula rasa" neural network would need to re-learn from scratch, penalising data efficiency. We incorporate this information into a pre-trained reasoning module, and investigate its role in shaping the discovered representations in diverse self-supervised learning settings from pixels. Our approach paves the way for a new class of data-efficient representation learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Delighted to share our work on reasoning-modulated representations! Contributed talk at <a href="https://twitter.com/icmlconf?ref_src=twsrc%5Etfw">@icmlconf</a> SSL Workshop 🎉<a href="https://t.co/5iTLOx0KpC">https://t.co/5iTLOx0KpC</a><br><br>Algo reasoning can help representation learning! See thread👇🧵<br><br>w/ Matko <a href="https://twitter.com/thomaskipf?ref_src=twsrc%5Etfw">@thomaskipf</a> <a href="https://twitter.com/AlexLerchner?ref_src=twsrc%5Etfw">@AlexLerchner</a> <a href="https://twitter.com/RaiaHadsell?ref_src=twsrc%5Etfw">@RaiaHadsell</a> <a href="https://twitter.com/rpascanu?ref_src=twsrc%5Etfw">@rpascanu</a> <a href="https://twitter.com/BlundellCharles?ref_src=twsrc%5Etfw">@BlundellCharles</a> <a href="https://t.co/W7qtNuAFyt">pic.twitter.com/W7qtNuAFyt</a></p>&mdash; Petar Veličković (@PetarV_93) <a href="https://twitter.com/PetarV_93/status/1417388054713622535?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. EvilModel: Hiding Malware Inside of Neural Network Models

Zhi Wang, Chaoge Liu, Xiang Cui

- retweets: 380, favorites: 102 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08590) | [pdf](https://arxiv.org/pdf/2107.08590)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Delivering malware covertly and detection-evadingly is critical to advanced malware campaigns. In this paper, we present a method that delivers malware covertly and detection-evadingly through neural network models. Neural network models are poorly explainable and have a good generalization ability. By embedding malware into the neurons, malware can be delivered covertly with minor or even no impact on the performance of neural networks. Meanwhile, since the structure of the neural network models remains unchanged, they can pass the security scan of antivirus engines. Experiments show that 36.9MB of malware can be embedded into a 178MB-AlexNet model within 1% accuracy loss, and no suspicious are raised by antivirus engines in VirusTotal, which verifies the feasibility of this method. With the widespread application of artificial intelligence, utilizing neural networks becomes a forwarding trend of malware. We hope this work could provide a referenceable scenario for the defense on neural network-assisted attacks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GROAN! EvilModel: Hiding Malware Inside of Neural Network Models: <a href="https://t.co/vIBtOu5Sq1">https://t.co/vIBtOu5Sq1</a><br><br>(Caveat: pre-print, unreviewed. Not obviously implausible, though, and utterly horrible security implications if substantiated.) <a href="https://t.co/QOEcrBKBpN">pic.twitter.com/QOEcrBKBpN</a></p>&mdash; Charlie Stross (@cstross) <a href="https://twitter.com/cstross/status/1417481277712838660?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">OMG LOL.<br> <br>EvilModel: Hiding Malware Inside of Neural Network Models<a href="https://t.co/bFYLQfuMr5">https://t.co/bFYLQfuMr5</a><a href="https://twitter.com/kcarruthers?ref_src=twsrc%5Etfw">@kcarruthers</a> <a href="https://twitter.com/bruces?ref_src=twsrc%5Etfw">@bruces</a></p>&mdash; Incredible Good Fun Frances Dances (@datakid23) <a href="https://twitter.com/datakid23/status/1417451027092041729?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Equivariant Manifold Flows

Isay Katsman, Aaron Lou, Derek Lim, Qingxuan Jiang, Ser-Nam Lim, Christopher De Sa

- retweets: 380, favorites: 89 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08596) | [pdf](https://arxiv.org/pdf/2107.08596)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.DG](https://arxiv.org/list/math.DG/recent)

Tractably modelling distributions over manifolds has long been an important goal in the natural sciences. Recent work has focused on developing general machine learning models to learn such distributions. However, for many applications these distributions must respect manifold symmetries -- a trait which most previous models disregard. In this paper, we lay the theoretical foundations for learning symmetry-invariant distributions on arbitrary manifolds via equivariant manifold flows. We demonstrate the utility of our approach by using it to learn gauge invariant densities over $SU(n)$ in the context of quantum field theory.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am happy to present our new work, “Equivariant Manifold Flows”, together with <a href="https://twitter.com/aaron_lou?ref_src=twsrc%5Etfw">@aaron_lou</a>, <a href="https://twitter.com/dereklim_lzh?ref_src=twsrc%5Etfw">@dereklim_lzh</a>, Qingxuan Jiang, <a href="https://twitter.com/sernamlim?ref_src=twsrc%5Etfw">@sernamlim</a>, <a href="https://twitter.com/chrismdesa?ref_src=twsrc%5Etfw">@chrismdesa</a>!<br><br>Arxiv: <a href="https://t.co/S1GkKikgcz">https://t.co/S1GkKikgcz</a> <a href="https://t.co/GsQRAgYzyb">pic.twitter.com/GsQRAgYzyb</a></p>&mdash; Isay Katsman (@isaykatsman) <a href="https://twitter.com/isaykatsman/status/1417353457875443712?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Translatotron 2: Robust direct speech-to-speech translation

Ye Jia, Michelle Tadmor Ramanovich, Tal Remez, Roi Pomerantz

- retweets: 323, favorites: 68 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08661) | [pdf](https://arxiv.org/pdf/2107.08661)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We present Translatotron 2, a neural direct speech-to-speech translation model that can be trained end-to-end. Translatotron 2 consists of a speech encoder, a phoneme decoder, a mel-spectrogram synthesizer, and an attention module that connects all the previous three components. Experimental results suggest that Translatotron 2 outperforms the original Translatotron by a large margin in terms of translation quality and predicted speech naturalness, and drastically improves the robustness of the predicted speech by mitigating over-generation, such as babbling or long pause. We also propose a new method for retaining the source speaker's voice in the translated speech. The trained model is restricted to retain the source speaker's voice, and unlike the original Translatotron, it is not able to generate speech in a different speaker's voice, making the model more robust for production deployment, by mitigating potential misuse for creating spoofing audio artifacts. When the new method is used together with a simple concatenation-based data augmentation, the trained Translatotron 2 model is able to retain each speaker's voice for input with speaker turns.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Translatotron 2: Robust direct speech-to-speech translation<br><br>pdf: <a href="https://t.co/9IPIWOwWac">https://t.co/9IPIWOwWac</a><br>samples: <a href="https://t.co/TEXw3z59O2">https://t.co/TEXw3z59O2</a><br><br>outperforms Translatotron by a large margin in terms of translation quality and predicted speech naturalness <a href="https://t.co/dQ97yE9iow">pic.twitter.com/dQ97yE9iow</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1417294697584812037?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Autonomy 2.0: Why is self-driving always 5 years away?

Ashesh Jain, Luca Del Pero, Hugo Grimmett, Peter Ondruska

- retweets: 228, favorites: 90 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08142) | [pdf](https://arxiv.org/pdf/2107.08142)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite the numerous successes of machine learning over the past decade (image recognition, decision-making, NLP, image synthesis), self-driving technology has not yet followed the same trend. In this paper, we study the history, composition, and development bottlenecks of the modern self-driving stack. We argue that the slow progress is caused by approaches that require too much hand-engineering, an over-reliance on road testing, and high fleet deployment costs. We observe that the classical stack has several bottlenecks that preclude the necessary scale needed to capture the long tail of rare events. To resolve these problems, we outline the principles of Autonomy 2.0, an ML-first approach to self-driving, as a viable alternative to the currently adopted state-of-the-art. This approach is based on (i) a fully differentiable AV stack trainable from human demonstrations, (ii) closed-loop data-driven reactive simulation, and (iii) large-scale, low-cost data collections as critical solutions towards scalability issues. We outline the general architecture, survey promising works in this direction and propose key challenges to be addressed by the community in the future.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Autonomy 2.0: Why is self-driving always 5 years away?<br>pdf: <a href="https://t.co/z3QOYvPAC3">https://t.co/z3QOYvPAC3</a><br>abs: <a href="https://t.co/pHWTgqjQU1">https://t.co/pHWTgqjQU1</a><br><br>outlines the Autonomy 2.0 paradigm, which is designed to solve self-driving using an ML-first approach <a href="https://t.co/KR7BwT3TRh">pic.twitter.com/KR7BwT3TRh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1417320762223079448?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. CodeMapping: Real-Time Dense Mapping for Sparse SLAM using Compact Scene  Representations

Hidenobu Matsuki, Raluca Scona, Jan Czarnowski, Andrew J. Davison

- retweets: 132, favorites: 53 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08994) | [pdf](https://arxiv.org/pdf/2107.08994)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We propose a novel dense mapping framework for sparse visual SLAM systems which leverages a compact scene representation. State-of-the-art sparse visual SLAM systems provide accurate and reliable estimates of the camera trajectory and locations of landmarks. While these sparse maps are useful for localization, they cannot be used for other tasks such as obstacle avoidance or scene understanding. In this paper we propose a dense mapping framework to complement sparse visual SLAM systems which takes as input the camera poses, keyframes and sparse points produced by the SLAM system and predicts a dense depth image for every keyframe. We build on CodeSLAM and use a variational autoencoder (VAE) which is conditioned on intensity, sparse depth and reprojection error images from sparse SLAM to predict an uncertainty-aware dense depth map. The use of a VAE then enables us to refine the dense depth images through multi-view optimization which improves the consistency of overlapping frames. Our mapper runs in a separate thread in parallel to the SLAM system in a loosely coupled manner. This flexible design allows for integration with arbitrary metric sparse SLAM systems without delaying the main SLAM process. Our dense mapper can be used not only for local mapping but also globally consistent dense 3D reconstruction through TSDF fusion. We demonstrate our system running with ORB-SLAM3 and show accurate dense depth estimation which could enable applications such as robotics and augmented reality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first work at Imperial is accepted to IEEE Robotics and Automation Letters!  <br>We propose CodeMapping, a real-time and code-based dense mapper for sparse vSLAM.<br>Huge thanks to co-authors <a href="https://twitter.com/RalucaScona?ref_src=twsrc%5Etfw">@RalucaScona</a> <a href="https://twitter.com/czarnowskij?ref_src=twsrc%5Etfw">@czarnowskij</a> <a href="https://twitter.com/AjdDavison?ref_src=twsrc%5Etfw">@AjdDavison</a>!<a href="https://t.co/bFLir4nKmL">https://t.co/bFLir4nKmL</a><a href="https://t.co/Q6DML69zxS">https://t.co/Q6DML69zxS</a> <a href="https://t.co/NHNZKxR0Xv">pic.twitter.com/NHNZKxR0Xv</a></p>&mdash; Hide (@HideMatsu82) <a href="https://twitter.com/HideMatsu82/status/1417491121383616533?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Megaverse: Simulating Embodied Agents at One Million Experiences per  Second

Aleksei Petrenko, Erik Wijmans, Brennan Shacklett, Vladlen Koltun

- retweets: 90, favorites: 66 (07/21/2021 10:09:32)

- links: [abs](https://arxiv.org/abs/2107.08170) | [pdf](https://arxiv.org/pdf/2107.08170)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present Megaverse, a new 3D simulation platform for reinforcement learning and embodied AI research. The efficient design of our engine enables physics-based simulation with high-dimensional egocentric observations at more than 1,000,000 actions per second on a single 8-GPU node. Megaverse is up to 70x faster than DeepMind Lab in fully-shaded 3D scenes with interactive objects. We achieve this high simulation performance by leveraging batched simulation, thereby taking full advantage of the massive parallelism of modern GPUs. We use Megaverse to build a new benchmark that consists of several single-agent and multi-agent tasks covering a variety of cognitive challenges. We evaluate model-free RL on this benchmark to provide baselines and facilitate future research. The source code is available at https://www.megaverse.info

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Megaverse: Simulating Embodied Agents at One Million Experiences per Second<br>pdf: <a href="https://t.co/IMTBxLsXKZ">https://t.co/IMTBxLsXKZ</a><br>abs: <a href="https://t.co/LZws2Eg7gl">https://t.co/LZws2Eg7gl</a><br>project page: <a href="https://t.co/S6NmtU2poc">https://t.co/S6NmtU2poc</a><br>github: <a href="https://t.co/OqNTANBSfI">https://t.co/OqNTANBSfI</a> <a href="https://t.co/wWz3s0uprs">pic.twitter.com/wWz3s0uprs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1417296255416406016?ref_src=twsrc%5Etfw">July 20, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



