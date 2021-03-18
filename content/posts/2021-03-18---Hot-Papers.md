---
title: Hot Papers 2021-03-18
date: 2021-03-19T06:21:49.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Training GANs with Stronger Augmentations via Contrastive Discriminator

Jongheon Jeong, Jinwoo Shin

- retweets: 1802, favorites: 277 (03/19/2021 06:21:49)

- links: [abs](https://arxiv.org/abs/2103.09742) | [pdf](https://arxiv.org/pdf/2103.09742)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent works in Generative Adversarial Networks (GANs) are actively revisiting various data augmentation techniques as an effective way to prevent discriminator overfitting. It is still unclear, however, that which augmentations could actually improve GANs, and in particular, how to apply a wider range of augmentations in training. In this paper, we propose a novel way to address these questions by incorporating a recent contrastive representation learning scheme into the GAN discriminator, coined ContraD. This "fusion" enables the discriminators to work with much stronger augmentations without increasing their training instability, thereby preventing the discriminator overfitting issue in GANs more effectively. Even better, we observe that the contrastive learning itself also benefits from our GAN training, i.e., by maintaining discriminative features between real and fake samples, suggesting a strong coherence between the two worlds: good contrastive representations are also good for GAN discriminators, and vice versa. Our experimental results show that GANs with ContraD consistently improve FID and IS compared to other recent techniques incorporating data augmentations, still maintaining highly discriminative features in the discriminator in terms of the linear evaluation. Finally, as a byproduct, we also show that our GANs trained in an unsupervised manner (without labels) can induce many conditional generative models via a simple latent sampling, leveraging the learned features of ContraD. Code is available at https://github.com/jh-jeong/ContraD.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Training GANs with Stronger Augmentations via Contrastive Discriminator<br>pdf: <a href="https://t.co/iItQPJw1BP">https://t.co/iItQPJw1BP</a><br>abs: <a href="https://t.co/bB65Smkg4T">https://t.co/bB65Smkg4T</a><br>github: <a href="https://t.co/jl91uhlceX">https://t.co/jl91uhlceX</a> <a href="https://t.co/0qSaqur46d">pic.twitter.com/0qSaqur46d</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372351248398225408?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Training GANs with Stronger Augmentations via Contrastive Discriminator<br><br>The use of contrastive discriminator consistently improves FID of various architectures, including StyleGAN2.<br><br>abs: <a href="https://t.co/lCx8kw03Bx">https://t.co/lCx8kw03Bx</a><br>code: <a href="https://t.co/cIomKrz6w5">https://t.co/cIomKrz6w5</a> <a href="https://t.co/4qn5ucadMD">pic.twitter.com/4qn5ucadMD</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1372353079115358208?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. You Only Look One-level Feature

Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, Jian Sun

- retweets: 789, favorites: 132 (03/19/2021 06:21:50)

- links: [abs](https://arxiv.org/abs/2103.09460) | [pdf](https://arxiv.org/pdf/2103.09460)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being $2.5\times$ faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with $7\times$ less training epochs. With an image size of $608\times608$, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is $13\%$ faster than YOLOv4. Code is available at \url{https://github.com/megvii-model/YOLOF}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You Only Look One-level Feature<br><br>Proposes YOLOF, which achieves matches RetinaNet while being 2.5x faster and 44.3 mAP on COCO with image size of 608 × 608 running at 60 fps on 2080Ti, which is 13% faster than YOLOv4.<br><br>abs: <a href="https://t.co/JrN1tNSFIs">https://t.co/JrN1tNSFIs</a><br>code: <a href="https://t.co/CqlYmh2qAa">https://t.co/CqlYmh2qAa</a> <a href="https://t.co/Zik4NSmUTe">pic.twitter.com/Zik4NSmUTe</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1372349396164182016?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You Only Look One-level Feature<br>pdf: <a href="https://t.co/upQxeRO4Em">https://t.co/upQxeRO4Em</a><br>abs: <a href="https://t.co/jZEu5569Yo">https://t.co/jZEu5569Yo</a><br>github: <a href="https://t.co/iuwGX1p2q5">https://t.co/iuwGX1p2q5</a> <a href="https://t.co/noaF2zZzMA">pic.twitter.com/noaF2zZzMA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372347313734189061?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. A Practical Guide to Multi-Objective Reinforcement Learning and Planning

Conor F. Hayes, Roxana Rădulescu, Eugenio Bargiacchi, Johan Källström, Matthew Macfarlane, Mathieu Reymond, Timothy Verstraeten, Luisa M. Zintgraf, Richard Dazeley, Fredrik Heintz, Enda Howley, Athirai A. Irissappane, Patrick Mannion, Ann Nowé, Gabriel Ramos, Marcello Restelli, Peter Vamplew, Diederik M. Roijers

- retweets: 552, favorites: 165 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09568) | [pdf](https://arxiv.org/pdf/2103.09568)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Real-world decision-making tasks are generally complex, requiring trade-offs between multiple, often conflicting, objectives. Despite this, the majority of research in reinforcement learning and decision-theoretic planning either assumes only a single objective, or that multiple objectives can be adequately handled via a simple linear combination. Such approaches may oversimplify the underlying problem and hence produce suboptimal results. This paper serves as a guide to the application of multi-objective methods to difficult problems, and is aimed at researchers who are already familiar with single-objective reinforcement learning and planning methods who wish to adopt a multi-objective perspective on their research, as well as practitioners who encounter multi-objective decision problems in practice. It identifies the factors that may influence the nature of the desired solution, and illustrates by example how these influence the design of multi-objective decision-making systems for complex problems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Curious about Multi-Objective RL? 🤓 Think your problem has multiple objectives - but not sure where to start? 🤔 We got you! 🙈😍<br><br>&quot;A Practical Guide to Multi-Objective Reinforcement Learning and Planning&quot; - Hayes, Radulescu, et (16!) al.<br><br>Preprint: <a href="https://t.co/mkFRpqvIf5">https://t.co/mkFRpqvIf5</a> <a href="https://t.co/8cctywU8Wy">pic.twitter.com/8cctywU8Wy</a></p>&mdash; Luisa Zintgraf (@luisa_zintgraf) <a href="https://twitter.com/luisa_zintgraf/status/1372544756853702658?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multiple objectives, multiple agents, multiple authors. Now available on arxiv: A Practical Guide to Multi-Objective Reinforcement Learning and Planning. A collaboration of 18 MORL researchers from 9 different countries and 4 different continents! <a href="https://t.co/S3kb6MHumY">https://t.co/S3kb6MHumY</a> 1/n</p>&mdash; Peter Vamplew (@amp1874) <a href="https://twitter.com/amp1874/status/1372399626297376771?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. PythonFOAM: In-situ data analyses with OpenFOAM and Python

Romit Maulik, Dimitrios Fytanidis, Bethany Lusch, Venkatram Vishwanath, Saumil Patel

- retweets: 169, favorites: 25 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09389) | [pdf](https://arxiv.org/pdf/2103.09389)
- [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [physics.flu-dyn](https://arxiv.org/list/physics.flu-dyn/recent)

In this article, we outline the development of a general-purpose Python-based data analysis tool for OpenFOAM 8. Our implementation relies on the construction of OpenFOAM applications that have bindings to data analysis libraries in Python. Double precision data in OpenFOAM is cast to a NumPy array using the NumPy C-API and Python modules may then be used for arbitrary data analysis and manipulation on flow-field information. This document highlights how the proposed framework may be used for an in-situ online singular value decomposition (SVD) implemented in Python and accessed from the OpenFOAM solver PimpleFOAM. Here, `in-situ' refers to a programming paradigm that allows for a concurrent computation of the data analysis on the same computational resources utilized for the partial differential equation solver. In addition, to demonstrate data-parallel analyses, we deploy a distributed SVD, which collects snapshot data across the ranks of a distributed simulation to compute the global left singular vectors. Crucially, both OpenFOAM and Python share the same message passing interface (MPI) communicator for this deployment which allows Python objects and functions to exchange NumPy arrays across ranks. Our experiments also demonstrate how customized data science libraries such as TensorFlow may be leveraged through these bindings. Subsequently, we provide scaling assessments of our framework and the selected algorithms on multiple nodes of Intel Broadwell and KNL architectures for canonical test cases such as the large eddy simulations of a backward facing step and a channel flow at friction Reynolds number of 395.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">PythonFOAMですってよ<a href="https://t.co/ALen9iZVEb">https://t.co/ALen9iZVEb</a></p>&mdash; ダムブレークP🌸 (@dmbrkp_) <a href="https://twitter.com/dmbrkp_/status/1372398906403737602?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Softermax: Hardware/Software Co-Design of an Efficient Softmax for  Transformers

Jacob R. Stevens, Rangharajan Venkatesan, Steve Dai, Brucek Khailany, Anand Raghunathan

- retweets: 120, favorites: 60 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09301) | [pdf](https://arxiv.org/pdf/2103.09301)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

Transformers have transformed the field of natural language processing. This performance is largely attributed to the use of stacked self-attention layers, each of which consists of matrix multiplies as well as softmax operations. As a result, unlike other neural networks, the softmax operation accounts for a significant fraction of the total run-time of Transformers. To address this, we propose Softermax, a hardware-friendly softmax design. Softermax consists of base replacement, low-precision softmax computations, and an online normalization calculation. We show Softermax results in 2.35x the energy efficiency at 0.90x the size of a comparable baseline, with negligible impact on network accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers<br>pdf: <a href="https://t.co/L4R0LFeHcL">https://t.co/L4R0LFeHcL</a><br>abs: <a href="https://t.co/7otCt4aYXe">https://t.co/7otCt4aYXe</a> <a href="https://t.co/GN1N50NfwY">pic.twitter.com/GN1N50NfwY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372346745045258240?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Evaluation of soccer team defense based on prediction models of ball  recovery and being attacked

Kosuke Toda, Masakiyo Teranishi, Keisuke Kushiro, Keisuke Fujii

- retweets: 110, favorites: 38 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09627) | [pdf](https://arxiv.org/pdf/2103.09627)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

With the development of measurement technology, data on the movements of actual games in various sports are available and are expected to be used for planning and evaluating the tactics and strategy. In particular, defense in team sports is generally difficult to be evaluated because of the lack of statistical data. Conventional evaluation methods based on predictions of scores are considered unreliable and predict rare events throughout the entire game, and it is difficult to evaluate various plays leading up to a score. On the other hand, evaluation methods based on certain plays that lead to scoring and dominant regions are sometimes unsuitable to evaluate the performance (e.g., goals scored) of players and teams. In this study, we propose a method to evaluate team defense from a comprehensive perspective related to team performance based on the prediction of ball recovery and being attacked, which occur more frequently than goals, using player behavior and positional data of all players and the ball. Using data from 45 soccer matches, we examined the relationship between the proposed index and team performance in actual matches and throughout a season. Results show that the proposed classifiers more accurately predicted the true events than the existing classifiers which were based on rare events (i.e., goals). Also, the proposed index had a moderate correlation with the long-term outcomes of the season. These results suggest that the proposed index might be a more reliable indicator rather than winning or losing with the inclusion of accidental factors.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">この「サッカーにおけるボール奪取・被有効攻撃予測に基づくチームの守備評価」に関する論文を、arXivにて公開しました：<a href="https://t.co/VYN7ur1uDn">https://t.co/VYN7ur1uDn</a> <a href="https://t.co/XlP23nOx7W">https://t.co/XlP23nOx7W</a></p>&mdash; Keisuke Fujii (@keisuke_fj) <a href="https://twitter.com/keisuke_fj/status/1372370595459989505?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Large-Scale Zero-Shot Image Classification from Rich and Diverse Textual  Descriptions

Sebastian Bujwid, Josephine Sullivan

- retweets: 76, favorites: 72 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09669) | [pdf](https://arxiv.org/pdf/2103.09669)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We study the impact of using rich and diverse textual descriptions of classes for zero-shot learning (ZSL) on ImageNet. We create a new dataset ImageNet-Wiki that matches each ImageNet class to its corresponding Wikipedia article. We show that merely employing these Wikipedia articles as class descriptions yields much higher ZSL performance than prior works. Even a simple model using this type of auxiliary data outperforms state-of-the-art models that rely on standard features of word embedding encodings of class names. These results highlight the usefulness and importance of textual descriptions for ZSL, as well as the relative importance of auxiliary data type compared to algorithmic progress. Our experimental results also show that standard zero-shot learning approaches generalize poorly across categories of classes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large-Scale Zero-Shot Image Classification from Rich and Diverse Textual Descriptions<br><br>Matching each ImageNet class to its corresponding Wiki page and using the articles as label yield much higher ZSL performance on Imagenet than prior works.<a href="https://t.co/TfgJ9luMnO">https://t.co/TfgJ9luMnO</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1372351602678321155?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Toward Neural-Network-Guided Program Synthesis and Verification

Naoki Kobayashi, Taro Sekiyama, Issei Sato, Hiroshi Unno

- retweets: 90, favorites: 43 (03/19/2021 06:21:51)

- links: [abs](https://arxiv.org/abs/2103.09414) | [pdf](https://arxiv.org/pdf/2103.09414)
- [cs.PL](https://arxiv.org/list/cs.PL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a novel framework of program and invariant synthesis called neural network-guided synthesis. We first show that, by suitably designing and training neural networks, we can extract logical formulas over integers from the weights and biases of the trained neural networks. Based on the idea, we have implemented a tool to synthesize formulas from positive/negative examples and implication constraints, and obtained promising experimental results. We also discuss two applications of our synthesis method. One is the use of our tool for qualifier discovery in the framework of ICE-learning-based CHC solving, which can in turn be applied to program verification and inductive invariant synthesis. Another application is to a new program development framework called oracle-based programming, which is a neural-network-guided variation of Solar-Lezama's program synthesis by sketching.

<blockquote class="twitter-tweet"><p lang="et" dir="ltr">Toward Neural-Network-Guided Program Synthesis and Verification. Naoki Kobayashi, Taro Sekiyama, Issei Sato, and Hiroshi Unno <a href="https://t.co/sfHQhcPFD5">https://t.co/sfHQhcPFD5</a></p>&mdash; cs.LG Papers (@arxiv_cs_LG) <a href="https://twitter.com/arxiv_cs_LG/status/1372405028544073729?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. The Invertible U-Net for Optical-Flow-free Video Interframe Generation

Saem Park, Donghun Han, Nojun Kwak

- retweets: 63, favorites: 49 (03/19/2021 06:21:52)

- links: [abs](https://arxiv.org/abs/2103.09576) | [pdf](https://arxiv.org/pdf/2103.09576)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Video frame interpolation is the task of creating an interface between two adjacent frames along the time axis. So, instead of simply averaging two adjacent frames to create an intermediate image, this operation should maintain semantic continuity with the adjacent frames. Most conventional methods use optical flow, and various tools such as occlusion handling and object smoothing are indispensable. Since the use of these various tools leads to complex problems, we tried to tackle the video interframe generation problem without using problematic optical flow. To enable this, we have tried to use a deep neural network with an invertible structure and developed an invertible U-Net which is a modified normalizing flow. In addition, we propose a learning method with a new consistency loss in the latent space to maintain semantic temporal consistency between frames. The resolution of the generated image is guaranteed to be identical to that of the original images by using an invertible network. Furthermore, as it is not a random image like the ones by generative models, our network guarantees stable outputs without flicker. Through experiments, we confirmed the feasibility of the proposed algorithm and would like to suggest invertible U-Net as a new possibility for baseline in video frame interpolation. This paper is meaningful in that it is the worlds first attempt to use invertible networks instead of optical flows for video interpolation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Invertible U-Net for Optical-Flow-free Video Interframe Generation<br>pdf: <a href="https://t.co/CsriWt65Co">https://t.co/CsriWt65Co</a><br>abs: <a href="https://t.co/GNJazrIFwF">https://t.co/GNJazrIFwF</a> <a href="https://t.co/oD6c99Leoc">pic.twitter.com/oD6c99Leoc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372369529477685249?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs

Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, Jure Leskovec

- retweets: 62, favorites: 48 (03/19/2021 06:21:52)

- links: [abs](https://arxiv.org/abs/2103.09430) | [pdf](https://arxiv.org/pdf/2103.09430)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Enabling effective and efficient machine learning (ML) over large-scale graph data (e.g., graphs with billions of edges) can have a huge impact on both industrial and scientific applications. However, community efforts to advance large-scale graph ML have been severely limited by the lack of a suitable public benchmark. For KDD Cup 2021, we present OGB Large-Scale Challenge (OGB-LSC), a collection of three real-world datasets for advancing the state-of-the-art in large-scale graph ML. OGB-LSC provides graph datasets that are orders of magnitude larger than existing ones and covers three core graph learning tasks -- link prediction, graph regression, and node classification. Furthermore, OGB-LSC provides dedicated baseline experiments, scaling up expressive graph ML models to the massive datasets. We show that the expressive models significantly outperform simple scalable baselines, indicating an opportunity for dedicated efforts to further improve graph ML at scale. Our datasets and baseline code are released and maintained as part of our OGB initiative (Hu et al., 2020). We hope OGB-LSC at KDD Cup 2021 can empower the community to discover innovative solutions for large-scale graph ML.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to release our paper on the OGB Large-Scale Challenge (OGB-LSC) @ KDD Cup 2021 (on-going)!<br><br>Learn about our large datasets + extensive baseline analyses, showing that big expressive models are promising for large-scale graph data!<a href="https://t.co/Ozrtyxr2mc">https://t.co/Ozrtyxr2mc</a></p>&mdash; Weihua Hu (@weihua916) <a href="https://twitter.com/weihua916/status/1372432770044026884?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. ALADIN: All Layer Adaptive Instance Normalization for Fine-grained Style  Similarity

Dan Ruta, Saeid Motiian, Baldo Faieta, Zhe Lin, Hailin Jin, Alex Filipkowski, Andrew Gilbert, John Collomosse

- retweets: 38, favorites: 43 (03/19/2021 06:21:52)

- links: [abs](https://arxiv.org/abs/2103.09776) | [pdf](https://arxiv.org/pdf/2103.09776)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present ALADIN (All Layer AdaIN); a novel architecture for searching images based on the similarity of their artistic style. Representation learning is critical to visual search, where distance in the learned search embedding reflects image similarity. Learning an embedding that discriminates fine-grained variations in style is hard, due to the difficulty of defining and labelling style. ALADIN takes a weakly supervised approach to learning a representation for fine-grained style similarity of digital artworks, leveraging BAM-FG, a novel large-scale dataset of user generated content groupings gathered from the web. ALADIN sets a new state of the art accuracy for style-based visual search over both coarse labelled style data (BAM) and BAM-FG; a new 2.62 million image dataset of 310,000 fine-grained style groupings also contributed by this work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ALADIN: All Layer Adaptive Instance Normalization for Fine-grained Style Similarity<br>pdf: <a href="https://t.co/K7homQW2GM">https://t.co/K7homQW2GM</a><br>abs: <a href="https://t.co/jTp3DS4u7o">https://t.co/jTp3DS4u7o</a> <a href="https://t.co/NUBfHbZZWE">pic.twitter.com/NUBfHbZZWE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372350531956568064?ref_src=twsrc%5Etfw">March 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Physics-Informed Deep-Learning for Scientific Computing

Stefano Markidis

- retweets: 44, favorites: 14 (03/19/2021 06:21:52)

- links: [abs](https://arxiv.org/abs/2103.09655) | [pdf](https://arxiv.org/pdf/2103.09655)
- [math.NA](https://arxiv.org/list/math.NA/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent)

Physics-Informed Neural Networks (PINN) are neural networks that encode the problem governing equations, such as Partial Differential Equations (PDE), as a part of the neural network training. PINNs have emerged as an essential tool to solve various challenging problems, such as computing linear and non-linear PDEs, completing data assimilation and uncertainty quantification tasks. In this work, we focus on evaluating the PINN potential to replace or accelerate traditional approaches for solving linear systems. We solve the Poisson equation, one of the most critical and computational-intensive tasks in scientific computing, with different source terms. We test and evaluate PINN performance under different configurations (depth, activation functions, input data set distribution, and transfer learning impact). We show how to integrate PINN with traditional scientific computing approaches, such as multigrid and Gauss-Seidel methods. While the accuracy and computational performance is still a limiting factor for the direct use of PINN for solving, hybrid strategies are a viable option for the development of a new class of linear solvers combining emerging deep-learning and traditional scientific computing approaches.




# 13. DoubleML -- An Object-Oriented Implementation of Double Machine Learning  in R

Philipp Bach, Victor Chernozhukov, Malte S. Kurz, Martin Spindler

- retweets: 43, favorites: 14 (03/19/2021 06:21:52)

- links: [abs](https://arxiv.org/abs/2103.09603) | [pdf](https://arxiv.org/pdf/2103.09603)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [econ.EM](https://arxiv.org/list/econ.EM/recent)

The R package DoubleML implements the double/debiased machine learning framework of Chernozhukov et al. (2018). It provides functionalities to estimate parameters in causal models based on machine learning methods. The double machine learning framework consist of three key ingredients: Neyman orthogonality, high-quality machine learning estimation and sample splitting. Estimation of nuisance components can be performed by various state-of-the-art machine learning methods that are available in the mlr3 ecosystem. DoubleML makes it possible to perform inference in a variety of causal models, including partially linear and interactive regression models and their extensions to instrumental variable estimation. The object-oriented implementation of DoubleML enables a high flexibility for the model specification and makes it easily extendable. This paper serves as an introduction to the double machine learning framework and the R package DoubleML. In reproducible code examples with simulated and real data sets, we demonstrate how DoubleML users can perform valid inference based on machine learning methods.



