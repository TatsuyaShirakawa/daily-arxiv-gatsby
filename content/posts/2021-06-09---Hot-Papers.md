---
title: Hot Papers 2021-06-09
date: 2021-06-10T18:07:48.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-09"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-09"
socialImage: "/media/flying-marine.jpg"

---

# 1. A Survey of Transformers

Tianyang Lin, Yuxin Wang, Xiangyang Liu, Xipeng Qiu

- retweets: 17382, favorites: 49 (06/10/2021 18:07:48)

- links: [abs](https://arxiv.org/abs/2106.04554) | [pdf](https://arxiv.org/pdf/2106.04554)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformers have achieved great success in many artificial intelligence fields, such as natural language processing, computer vision, and audio processing. Therefore, it is natural to attract lots of interest from academic and industry researchers. Up to the present, a great variety of Transformer variants (a.k.a. X-formers) have been proposed, however, a systematic and comprehensive literature review on these Transformer variants is still missing. In this survey, we provide a comprehensive review of various X-formers. We first briefly introduce the vanilla Transformer and then propose a new taxonomy of X-formers. Next, we introduce the various X-formers from three perspectives: architectural modification, pre-training, and applications. Finally, we outline some potential directions for future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A comprehensive overview of Transformer variants.<br><br>A must-read for students getting into the world of machine learning and NLP.<a href="https://t.co/10DMA4ttdi">https://t.co/10DMA4ttdi</a> <a href="https://t.co/bPZQcxksFQ">pic.twitter.com/bPZQcxksFQ</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1402589585109098500?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Scaling Vision Transformers

Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, Lucas Beyer

- retweets: 4103, favorites: 98 (06/10/2021 18:07:48)

- links: [abs](https://arxiv.org/abs/2106.04560) | [pdf](https://arxiv.org/pdf/2106.04560)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Attention-based neural networks such as the Vision Transformer (ViT) have recently attained state-of-the-art results on many computer vision benchmarks. Scale is a primary ingredient in attaining excellent results, therefore, understanding a model's scaling properties is a key to designing future generations effectively. While the laws for scaling Transformer language models have been studied, it is unknown how Vision Transformers scale. To address this, we scale ViT models and data, both up and down, and characterize the relationships between error rate, data, and compute. Along the way, we refine the architecture and training of ViT, reducing memory consumption and increasing accuracy the resulting models. As a result, we successfully train a ViT model with two billion parameters, which attains a new state-of-the-art on ImageNet of 90.45% top-1 accuracy. The model also performs well on few-shot learning, for example, attaining 84.86% top-1 accuracy on ImageNet with only 10 examples per class.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">VisionTransformerを使った画像認識でも、データ、モデルサイズ、計算資源と精度間にべき乗則が成り立つ。30億枚の教師ありデータを使って事前学習。大きなモデルほど学習サンプル効率が高く、良いFewShow学習器になる。20億パラメータのViT-G/14はImageNetのSOTAを更新 <a href="https://t.co/mafcQQtldS">https://t.co/mafcQQtldS</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1402772065250287618?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Vision Transformers<br><br>Train ViT with 2B parameters, which attains a new SotA on ImageNet of 90.45% top-1 accuracy.<br><br>Also, achieves 84.86% top-1 acc. on ImageNet with only 10 examples per class.<a href="https://t.co/dQsrZiS05X">https://t.co/dQsrZiS05X</a> <a href="https://t.co/r33cvHbZep">pic.twitter.com/r33cvHbZep</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1402427601033916416?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Widening Access to Applied Machine Learning with TinyML

Vijay Janapa Reddi, Brian Plancher, Susan Kennedy, Laurence Moroney, Pete Warden, Anant Agarwal, Colby Banbury, Massimo Banzi, Matthew Bennett, Benjamin Brown, Sharad Chitlangia, Radhika Ghosal, Sarah Grafman, Rupert Jaeger, Srivatsan Krishnan, Maximilian Lam, Daniel Leiker, Cara Mann, Mark Mazumder, Dominic Pajak, Dhilan Ramaprasad, J. Evan Smith, Matthew Stewart, Dustin Tingley

- retweets: 686, favorites: 168 (06/10/2021 18:07:48)

- links: [abs](https://arxiv.org/abs/2106.04008) | [pdf](https://arxiv.org/pdf/2106.04008)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Broadening access to both computational and educational resources is critical to diffusing machine-learning (ML) innovation. However, today, most ML resources and experts are siloed in a few countries and organizations. In this paper, we describe our pedagogical approach to increasing access to applied ML through a massive open online course (MOOC) on Tiny Machine Learning (TinyML). We suggest that TinyML, ML on resource-constrained embedded devices, is an attractive means to widen access because TinyML both leverages low-cost and globally accessible hardware, and encourages the development of complete, self-contained applications, from data collection to deployment. To this end, a collaboration between academia (Harvard University) and industry (Google) produced a four-part MOOC that provides application-oriented instruction on how to develop solutions using TinyML. The series is openly available on the edX MOOC platform, has no prerequisites beyond basic programming, and is designed for learners from a global variety of backgrounds. It introduces pupils to real-world applications, ML algorithms, data-set engineering, and the ethical considerations of these technologies via hands-on programming and deployment of TinyML applications in both the cloud and their own microcontrollers. To facilitate continued learning, community building, and collaboration beyond the courses, we launched a standalone website, a forum, a chat, and an optional course-project competition. We also released the course materials publicly, hoping they will inspire the next generation of ML practitioners and educators and further broaden access to cutting-edge ML technologies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here’s a great read on <a href="https://twitter.com/Harvard?ref_src=twsrc%5Etfw">@Harvard</a> and <a href="https://twitter.com/Google?ref_src=twsrc%5Etfw">@Google</a>&#39;s efforts to widen the access to applied ML through their <a href="https://twitter.com/edXOnline?ref_src=twsrc%5Etfw">@edXOnline</a> tinyML course: <a href="https://t.co/om6ieiKI8h">https://t.co/om6ieiKI8h</a> <a href="https://t.co/3ktFJA0aZj">pic.twitter.com/3ktFJA0aZj</a></p>&mdash; Arduino (@arduino) <a href="https://twitter.com/arduino/status/1402620018421084160?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Supporting the <a href="https://twitter.com/EDX?ref_src=twsrc%5Etfw">@edx</a> TinyML course, in collaboration with <a href="https://twitter.com/Harvard?ref_src=twsrc%5Etfw">@Harvard</a> and <a href="https://twitter.com/Google?ref_src=twsrc%5Etfw">@Google</a>, was my last project at <a href="https://twitter.com/arduino?ref_src=twsrc%5Etfw">@Arduino</a> who provided <a href="https://twitter.com/Arm?ref_src=twsrc%5Etfw">@Arm</a> based hardware for the course. Proud to be involved and see they achieved 40K student signups - a phenomenal result! <a href="https://t.co/84OoiNveAH">https://t.co/84OoiNveAH</a> <a href="https://t.co/pBQSk3MJhe">pic.twitter.com/pBQSk3MJhe</a></p>&mdash; Dominic Pajak (@DominicPajak) <a href="https://twitter.com/DominicPajak/status/1402449957609644032?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Swords: A Benchmark for Lexical Substitution with Improved Data Coverage  and Quality

Mina Lee, Chris Donahue, Alexander Iyabor, Robin Jia, Percy Liang

- retweets: 529, favorites: 114 (06/10/2021 18:07:49)

- links: [abs](https://arxiv.org/abs/2106.04102) | [pdf](https://arxiv.org/pdf/2106.04102)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We release a new benchmark for lexical substitution, the task of finding appropriate substitutes for a target word in a context. To assist humans with writing, lexical substitution systems can suggest words that humans cannot easily think of. However, existing benchmarks depend on human recall as the only source of data, and therefore lack coverage of the substitutes that would be most helpful to humans. Furthermore, annotators often provide substitutes of low quality, which are not actually appropriate in the given context. We collect higher-coverage and higher-quality data by framing lexical substitution as a classification problem, guided by the intuition that it is easier for humans to judge the appropriateness of candidate substitutes than conjure them from memory. To this end, we use a context-free thesaurus to produce candidates and rely on human judgement to determine contextual appropriateness. Compared to the previous largest benchmark, our Swords benchmark has 4.1x more substitutes per target word for the same level of quality, and its substitutes are 1.5x more appropriate (based on human judgement) for the same number of substitutes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Swords ⚔️: Stanford word substitution benchmark <a href="https://twitter.com/hashtag/StanfordNLP?src=hash&amp;ref_src=twsrc%5Etfw">#StanfordNLP</a><br><br>Can your model &amp; writing assistant system find appropriate synonyms for a word in context? ✍️ <a href="https://t.co/PoJLq1KTaq">https://t.co/PoJLq1KTaq</a><br><br>June 9 (Wed) @ 9-10:20 AM PT <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> w/ <a href="https://twitter.com/chrisdonahuey?ref_src=twsrc%5Etfw">@chrisdonahuey</a> <a href="https://twitter.com/robinomial?ref_src=twsrc%5Etfw">@robinomial</a> <a href="https://twitter.com/alexanderiyabor?ref_src=twsrc%5Etfw">@alexanderiyabor</a> <a href="https://twitter.com/percyliang?ref_src=twsrc%5Etfw">@percyliang</a> <a href="https://t.co/5YLbAPwyyM">pic.twitter.com/5YLbAPwyyM</a></p>&mdash; MinaLee__ (@MinaLee__) <a href="https://twitter.com/MinaLee__/status/1402475556961087497?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Partial Optimal Transport for a Constant-Volume Lagrangian Mesh with  Free Boundaries

Bruno Lévy

- retweets: 420, favorites: 126 (06/10/2021 18:07:49)

- links: [abs](https://arxiv.org/abs/2106.03936) | [pdf](https://arxiv.org/pdf/2106.03936)
- [physics.flu-dyn](https://arxiv.org/list/physics.flu-dyn/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent)

This article introduces a representation of dynamic meshes, adapted to some numerical simulations that require controlling the volume of objects with free boundaries, such as incompressible fluid simulation, some astrophysical simulations at cosmological scale, and shape/topology optimization. The algorithm decomposes the simulated object into a set of convex cells called a Laguerre diagram, parameterized by the position of $N$ points in 3D and $N$ additional parameters that control the volumes of the cells. These parameters are found as the (unique) solution of a convex optimization problem -- semi-discrete Monge-Amp\`ere equation -- stemming from optimal transport theory. In this article, this setting is extended to objects with free boundaries and arbitrary topology, evolving in a domain of arbitrary shape, by solving a partial optimal transport problem. The resulting Lagrangian scheme makes it possible to accurately control the volume of the object, while precisely tracking interfaces, interactions, collisions, and topology changes.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Draft on free-surface fluid simulation is online:<a href="https://t.co/jJ88ylIh8f">https://t.co/jJ88ylIh8f</a><a href="https://twitter.com/HdeMaleprade?ref_src=twsrc%5Etfw">@HdeMaleprade</a> and <a href="https://twitter.com/KMMoerman?ref_src=twsrc%5Etfw">@KMMoerman</a> you are in the acks ! <a href="https://t.co/CyJQx5XAQt">pic.twitter.com/CyJQx5XAQt</a></p>&mdash; Bruno Levy (@BrunoLevy01) <a href="https://twitter.com/BrunoLevy01/status/1402526180801138688?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. DETReg: Unsupervised Pretraining with Region Priors for Object Detection

Amir Bar, Xin Wang, Vadim Kantorov, Colorado J Reed, Roei Herzig, Gal Chechik, Anna Rohrbach, Trevor Darrell, Amir Globerson

- retweets: 304, favorites: 98 (06/10/2021 18:07:49)

- links: [abs](https://arxiv.org/abs/2106.04550) | [pdf](https://arxiv.org/pdf/2106.04550)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Unsupervised pretraining has recently proven beneficial for computer vision tasks, including object detection. However, previous self-supervised approaches are not designed to handle a key aspect of detection: localizing objects. Here, we present DETReg, an unsupervised pretraining approach for object DEtection with TRansformers using Region priors. Motivated by the two tasks underlying object detection: localization and categorization, we combine two complementary signals for self-supervision. For an object localization signal, we use pseudo ground truth object bounding boxes from an off-the-shelf unsupervised region proposal method, Selective Search, which does not require training data and can detect objects at a high recall rate and very low precision. The categorization signal comes from an object embedding loss that encourages invariant object representations, from which the object category can be inferred. We show how to combine these two signals to train the Deformable DETR detection architecture from large amounts of unlabeled data. DETReg improves the performance over competitive baselines and previous self-supervised methods on standard benchmarks like MS COCO and PASCAL VOC. DETReg also outperforms previous supervised and unsupervised baseline approaches on low-data regime when trained with only 1%, 2%, 5%, and 10% of the labeled data on MS COCO. For code and pretrained models, visit the project page at https://amirbar.net/detreg

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DETReg: Unsupervised Pretraining with Region Priors for Object Detection<br>pdf: <a href="https://t.co/rLPSu27w7A">https://t.co/rLPSu27w7A</a><br>abs: <a href="https://t.co/9aaGSJeeqB">https://t.co/9aaGSJeeqB</a><br>project page: <a href="https://t.co/mnCDiWBftJ">https://t.co/mnCDiWBftJ</a><br><br>unsupervised pretraining approach for object detection with transformers using region priors <a href="https://t.co/lTH36cBcvk">pic.twitter.com/lTH36cBcvk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402517179283562500?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Encoding-dependent generalization bounds for parametrized quantum  circuits

Matthias C. Caro, Elies Gil-Fuster, Johannes Jakob Meyer, Jens Eisert, Ryan Sweke

- retweets: 243, favorites: 123 (06/10/2021 18:07:49)

- links: [abs](https://arxiv.org/abs/2106.03880) | [pdf](https://arxiv.org/pdf/2106.03880)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A large body of recent work has begun to explore the potential of parametrized quantum circuits (PQCs) as machine learning models, within the framework of hybrid quantum-classical optimization. In particular, theoretical guarantees on the out-of-sample performance of such models, in terms of generalization bounds, have emerged. However, none of these generalization bounds depend explicitly on how the classical input data is encoded into the PQC. We derive generalization bounds for PQC-based models that depend explicitly on the strategy used for data-encoding. These imply bounds on the performance of trained PQC-based models on unseen data. Moreover, our results facilitate the selection of optimal data-encoding strategies via structural risk minimization, a mathematically rigorous framework for model selection. We obtain our generalization bounds by bounding the complexity of PQC-based models as measured by the Rademacher complexity and the metric entropy, two complexity measures from statistical learning theory. To achieve this, we rely on a representation of PQC-based models via trigonometric functions. Our generalization bounds emphasize the importance of well-considered data-encoding strategies for PQC-based models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is a piece of work I am particularly happy with: It proves generalization bounds for parametrized quantum circuits in <a href="https://twitter.com/hashtag/quantummachinelearning?src=hash&amp;ref_src=twsrc%5Etfw">#quantummachinelearning</a> taking data-encoding seriously. Thanks to <a href="https://twitter.com/rndm_wlks?ref_src=twsrc%5Etfw">@rndm_wlks</a>, <a href="https://twitter.com/jj_xyz?ref_src=twsrc%5Etfw">@jj_xyz</a>, <a href="https://twitter.com/EliesMiquel?ref_src=twsrc%5Etfw">@EliesMiquel</a> and particularly to <a href="https://twitter.com/IMathYou2?ref_src=twsrc%5Etfw">@IMathYou2</a>. <a href="https://t.co/eBUSNCBgog">https://t.co/eBUSNCBgog</a> <a href="https://t.co/qtLoHFBt8q">pic.twitter.com/qtLoHFBt8q</a></p>&mdash; Jens Eisert (@jenseisert) <a href="https://twitter.com/jenseisert/status/1402452366822105089?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Today, I have the pleasure of telling you about my recent work together with <a href="https://twitter.com/EliesMiquel?ref_src=twsrc%5Etfw">@EliesMiquel</a>, <a href="https://twitter.com/jj_xyz?ref_src=twsrc%5Etfw">@jj_xyz</a>, <a href="https://twitter.com/jenseisert?ref_src=twsrc%5Etfw">@jenseisert</a>, and <a href="https://twitter.com/rndm_wlks?ref_src=twsrc%5Etfw">@rndm_wlks</a>: <br><br>Encoding-dependent generalization bounds for parametrized quantum circuits: <a href="https://t.co/1b5iUISazd">https://t.co/1b5iUISazd</a> <a href="https://t.co/CkinOhb1WX">pic.twitter.com/CkinOhb1WX</a></p>&mdash; Matthias C. Caro (@IMathYou2) <a href="https://twitter.com/IMathYou2/status/1402612149856448513?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Differentiable Quality Diversity

Matthew C. Fontaine, Stefanos Nikolaidis

- retweets: 72, favorites: 59 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.03894) | [pdf](https://arxiv.org/pdf/2106.03894)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Quality diversity (QD) is a growing branch of stochastic optimization research that studies the problem of generating an archive of solutions that maximize a given objective function but are also diverse with respect to a set of specified measure functions. However, even when these functions are differentiable, QD algorithms treat them as "black boxes", ignoring gradient information. We present the differentiable quality diversity (DQD) problem, a special case of QD, where both the objective and measure functions are first order differentiable. We then present MAP-Elites via Gradient Arborescence (MEGA), a DQD algorithm that leverages gradient information to efficiently explore the joint range of the objective and measure functions. Results in two QD benchmark domains and in searching the latent space of a StyleGAN show that MEGA significantly outperforms state-of-the-art QD algorithms, highlighting DQD's promise for efficient quality diversity optimization when gradient information is available. Source code is available at https://github.com/icaros-usc/dqd.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share some recent work with <a href="https://twitter.com/snikolaidis19?ref_src=twsrc%5Etfw">@snikolaidis19</a> on differentiable quality diversity (DQD).<br><br>pdf: <a href="https://t.co/7Js2BrZfw9">https://t.co/7Js2BrZfw9</a> <br>abs: <a href="https://t.co/CTv6MkJiS2">https://t.co/CTv6MkJiS2</a> <br>code: <a href="https://t.co/pAykoLxBPN">https://t.co/pAykoLxBPN</a> <a href="https://t.co/njVFWV38yp">pic.twitter.com/njVFWV38yp</a></p>&mdash; Matt Fontaine (@tehqin17) <a href="https://twitter.com/tehqin17/status/1402427333328334848?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Generative Flows with Invertible Attentions

Rhea Sanjay Sukthanker, Zhiwu Huang, Suryansh Kumar, Radu Timofte, Luc Van Gool

- retweets: 81, favorites: 50 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.03959) | [pdf](https://arxiv.org/pdf/2106.03959)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Flow-based generative models have shown excellent ability to explicitly learn the probability density function of data via a sequence of invertible transformations. Yet, modeling long-range dependencies over normalizing flows remains understudied. To fill the gap, in this paper, we introduce two types of invertible attention mechanisms for generative flow models. To be precise, we propose map-based and scaled dot-product attention for unconditional and conditional generative flow models. The key idea is to exploit split-based attention mechanisms to learn the attention weights and input representations on every two splits of flow feature maps. Our method provides invertible attention modules with tractable Jacobian determinants, enabling seamless integration of it at any positions of the flow-based models. The proposed attention mechanism can model the global data dependencies, leading to more comprehensive flow models. Evaluation on multiple generation tasks demonstrates that the introduced attention flow idea results in efficient flow models and compares favorably against the state-of-the-art unconditional and conditional generative flow methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generative Flows with Invertible Attentions<br>pdf: <a href="https://t.co/wO2gNLgRpA">https://t.co/wO2gNLgRpA</a><br>abs: <a href="https://t.co/rVJd8o9xJF">https://t.co/rVJd8o9xJF</a><br><br>two invertible attention modules for both the unconditional and conditional generative flow models <a href="https://t.co/W8NPuswAW4">pic.twitter.com/W8NPuswAW4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402474330274021379?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Hash Layers For Large Sparse Models

Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston

- retweets: 64, favorites: 66 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.04426) | [pdf](https://arxiv.org/pdf/2106.04426)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We investigate the training of sparse layers that use different parameters for different inputs based on hashing in large Transformer models. Specifically, we modify the feedforward layer to hash to different sets of weights depending on the current token, over all tokens in the sequence. We show that this procedure either outperforms or is competitive with learning-to-route mixture-of-expert methods such as Switch Transformers and BASE Layers, while requiring no routing parameters or extra terms in the objective function such as a load balancing loss, and no sophisticated assignment algorithm. We study the performance of different hashing techniques, hash sizes and input features, and show that balanced and random hashes focused on the most local features work best, compared to either learning clusters or using longer-range context. We show our approach works well both on large language modeling and dialogue tasks, and on downstream fine-tuning tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hash Layers For Large Sparse Models<br>pdf: <a href="https://t.co/2kPK7iY5ew">https://t.co/2kPK7iY5ew</a><br>abs: <a href="https://t.co/YGrLATHrB5">https://t.co/YGrLATHrB5</a><br><br>simple and efficient approach to sparse models in the Transformers-for-NLP setting based on hash layers <a href="https://t.co/oRzuYgagiK">pic.twitter.com/oRzuYgagiK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402428251390320640?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hash Layers For Large Sparse Models<br><br>Modifies FFN to hash to different sets of weights.<br><br>Either outperforms or is competitive with MoE methods such as Switch Transformers, while requiring no routing parameters or extra terms in the objective function.<a href="https://t.co/O2oirI0iK7">https://t.co/O2oirI0iK7</a> <a href="https://t.co/aCbOA4uDBH">pic.twitter.com/aCbOA4uDBH</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1402429227585048577?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces from  Video using Pose and Lighting Normalization

Avisek Lahiri, Vivek Kwatra, Christian Frueh, John Lewis, Chris Bregler

- retweets: 78, favorites: 47 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.04185) | [pdf](https://arxiv.org/pdf/2106.04185)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present a video-based learning framework for animating personalized 3D talking faces from audio. We introduce two training-time data normalizations that significantly improve data sample efficiency. First, we isolate and represent faces in a normalized space that decouples 3D geometry, head pose, and texture. This decomposes the prediction problem into regressions over the 3D face shape and the corresponding 2D texture atlas. Second, we leverage facial symmetry and approximate albedo constancy of skin to isolate and remove spatio-temporal lighting variations. Together, these normalizations allow simple networks to generate high fidelity lip-sync videos under novel ambient illumination while training with just a single speaker-specific video. Further, to stabilize temporal dynamics, we introduce an auto-regressive approach that conditions the model on its previous visual state. Human ratings and objective metrics demonstrate that our method outperforms contemporary state-of-the-art audio-driven video reenactment benchmarks in terms of realism, lip-sync and visual quality scores. We illustrate several applications enabled by our framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LipSync3D: Data-Efficient Learning of Personalized 3D Talking Faces from Video using Pose and Lighting Normalization<br>pdf: <a href="https://t.co/HgDAt5sevf">https://t.co/HgDAt5sevf</a><br>abs: <a href="https://t.co/U4oYihSUV9">https://t.co/U4oYihSUV9</a> <a href="https://t.co/f0CeSO4cZv">pic.twitter.com/f0CeSO4cZv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402435127058313217?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. TIMEDIAL: Temporal Commonsense Reasoning in Dialog

Lianhui Qin, Aditya Gupta, Shyam Upadhyay, Luheng He, Yejin Choi, Manaal Faruqui

- retweets: 82, favorites: 35 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.04571) | [pdf](https://arxiv.org/pdf/2106.04571)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Everyday conversations require understanding everyday events, which in turn, requires understanding temporal commonsense concepts interwoven with those events. Despite recent progress with massive pre-trained language models (LMs) such as T5 and GPT-3, their capability of temporal reasoning in dialogs remains largely under-explored. In this paper, we present the first study to investigate pre-trained LMs for their temporal reasoning capabilities in dialogs by introducing a new task and a crowd-sourced English challenge set, TIMEDIAL. We formulate TIME-DIAL as a multiple-choice cloze task with over 1.1K carefully curated dialogs. Empirical results demonstrate that even the best performing models struggle on this task compared to humans, with 23 absolute points of gap in accuracy. Furthermore, our analysis reveals that the models fail to reason about dialog context correctly; instead, they rely on shallow cues based on existing temporal patterns in context, motivating future research for modeling temporal concepts in text and robust contextual reasoning about them. The dataset is publicly available at: https://github.com/google-research-datasets/timedial.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TIMEDIAL: Temporal Commonsense Reasoning in Dialog<br>pdf: <a href="https://t.co/Y2yfT4qtEn">https://t.co/Y2yfT4qtEn</a><br>abs: <a href="https://t.co/QqcFv9a94r">https://t.co/QqcFv9a94r</a><br>dataset: <a href="https://t.co/U3k53i5kdf">https://t.co/U3k53i5kdf</a><br><br>a challenge set consisting of 1.1K multiple-choice cloze questions for temporal commonsense reasoning in dialog <a href="https://t.co/iEGNAaut59">pic.twitter.com/iEGNAaut59</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402427583355142144?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Data-Efficient Instance Generation from Instance Discrimination

Ceyuan Yang, Yujun Shen, Yinghao Xu, Bolei Zhou

- retweets: 42, favorites: 65 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.04566) | [pdf](https://arxiv.org/pdf/2106.04566)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative Adversarial Networks (GANs) have significantly advanced image synthesis, however, the synthesis quality drops significantly given a limited amount of training data. To improve the data efficiency of GAN training, prior work typically employs data augmentation to mitigate the overfitting of the discriminator yet still learn the discriminator with a bi-classification (i.e., real vs. fake) task. In this work, we propose a data-efficient Instance Generation (InsGen) method based on instance discrimination. Concretely, besides differentiating the real domain from the fake domain, the discriminator is required to distinguish every individual image, no matter it comes from the training set or from the generator. In this way, the discriminator can benefit from the infinite synthesized samples for training, alleviating the overfitting problem caused by insufficient training data. A noise perturbation strategy is further introduced to improve its discriminative power. Meanwhile, the learned instance discrimination capability from the discriminator is in turn exploited to encourage the generator for diverse generation. Extensive experiments demonstrate the effectiveness of our method on a variety of datasets and training settings. Noticeably, on the setting of 2K training images from the FFHQ dataset, we outperform the state-of-the-art approach with 23.5% FID improvement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Data-Efficient Instance Generation from Instance Discrimination<br>pdf: <a href="https://t.co/JOhRUEXt7U">https://t.co/JOhRUEXt7U</a><br>abs: <a href="https://t.co/XmLL6TCpkG">https://t.co/XmLL6TCpkG</a><br>project page: <a href="https://t.co/WklBPPB9Gu">https://t.co/WklBPPB9Gu</a><br><br>setting of 2K training images from the FFHQ dataset, outperform the sota approach with 23.5% FID improvement <a href="https://t.co/H5Rjod6TNW">pic.twitter.com/H5Rjod6TNW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402520676599275521?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Low-Rank Subspaces in GANs

Jiapeng Zhu, Ruili Feng, Yujun Shen, Deli Zhao, Zhengjun Zha, Jingren Zhou, Qifeng Chen

- retweets: 49, favorites: 56 (06/10/2021 18:07:50)

- links: [abs](https://arxiv.org/abs/2106.04488) | [pdf](https://arxiv.org/pdf/2106.04488)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The latent space of a Generative Adversarial Network (GAN) has been shown to encode rich semantics within some subspaces. To identify these subspaces, researchers typically analyze the statistical information from a collection of synthesized data, and the identified subspaces tend to control image attributes globally (i.e., manipulating an attribute causes the change of an entire image). By contrast, this work introduces low-rank subspaces that enable more precise control of GAN generation. Concretely, given an arbitrary image and a region of interest (e.g., eyes of face images), we manage to relate the latent space to the image region with the Jacobian matrix and then use low-rank factorization to discover steerable latent subspaces. There are three distinguishable strengths of our approach that can be aptly called LowRankGAN. First, compared to analytic algorithms in prior work, our low-rank factorization of Jacobians is able to find the low-dimensional representation of attribute manifold, making image editing more precise and controllable. Second, low-rank factorization naturally yields a null space of attributes such that moving the latent code within it only affects the outer region of interest. Therefore, local image editing can be simply achieved by projecting an attribute vector into the null space without relying on a spatial mask as existing methods do. Third, our method can robustly work with a local region from one image for analysis yet well generalize to other images, making it much easy to use in practice. Extensive experiments on state-of-the-art GAN models (including StyleGAN2 and BigGAN) trained on various datasets demonstrate the effectiveness of our LowRankGAN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Low-Rank Subspaces in GANs<br>pdf: <a href="https://t.co/fUE8bvL1y7">https://t.co/fUE8bvL1y7</a><br>abs: <a href="https://t.co/SdpaJlLSx3">https://t.co/SdpaJlLSx3</a><br><br>low-rank decomposition of the Jacobian matrix established between an arbitrary image and the<br>latent space yields a null space, enables image local editing by simply altering the latent code <a href="https://t.co/tQU3KKZQXX">pic.twitter.com/tQU3KKZQXX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402498445810749446?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Credit Assignment Through Broadcasting a Global Error Vector

David G. Clark, L. F. Abbott, SueYeon Chung

- retweets: 72, favorites: 28 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04089) | [pdf](https://arxiv.org/pdf/2106.04089)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Backpropagation (BP) uses detailed, unit-specific feedback to train deep neural networks (DNNs) with remarkable success. That biological neural circuits appear to perform credit assignment, but cannot implement BP, implies the existence of other powerful learning algorithms. Here, we explore the extent to which a globally broadcast learning signal, coupled with local weight updates, enables training of DNNs. We present both a learning rule, called global error-vector broadcasting (GEVB), and a class of DNNs, called vectorized nonnegative networks (VNNs), in which this learning rule operates. VNNs have vector-valued units and nonnegative weights past the first layer. The GEVB learning rule generalizes three-factor Hebbian learning, updating each weight by an amount proportional to the inner product of the presynaptic activation and a globally broadcast error vector when the postsynaptic unit is active. We prove that these weight updates are matched in sign to the gradient, enabling accurate credit assignment. Moreover, at initialization, these updates are exactly proportional to the gradient in the limit of infinite network width. GEVB matches the performance of BP in VNNs, and in some cases outperforms direct feedback alignment (DFA) applied in conventional networks. Unlike DFA, GEVB successfully trains convolutional layers. Altogether, our theoretical and empirical results point to a surprisingly powerful role for a global learning signal in training DNNs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint with <a href="https://twitter.com/s_y_chung?ref_src=twsrc%5Etfw">@s_y_chung</a> and Larry Abbott!<br>&quot;Credit Assignment Through Broadcasting a Global Error Vector&quot;<a href="https://t.co/2G5NgdMjLi">https://t.co/2G5NgdMjLi</a><br>(1/N)</p>&mdash; David Clark (@d_g_clark) <a href="https://twitter.com/d_g_clark/status/1402661613866471427?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Task-Generic Hierarchical Human Motion Prior using VAEs

Jiaman Li, Ruben Villegas, Duygu Ceylan, Jimei Yang, Zhengfei Kuang, Hao Li, Yajie Zhao

- retweets: 49, favorites: 48 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04004) | [pdf](https://arxiv.org/pdf/2106.04004)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

A deep generative model that describes human motions can benefit a wide range of fundamental computer vision and graphics tasks, such as providing robustness to video-based human pose estimation, predicting complete body movements for motion capture systems during occlusions, and assisting key frame animation with plausible movements. In this paper, we present a method for learning complex human motions independent of specific tasks using a combined global and local latent space to facilitate coarse and fine-grained modeling. Specifically, we propose a hierarchical motion variational autoencoder (HM-VAE) that consists of a 2-level hierarchical latent space. While the global latent space captures the overall global body motion, the local latent space enables to capture the refined poses of the different body parts. We demonstrate the effectiveness of our hierarchical motion variational autoencoder in a variety of tasks including video-based human pose estimation, motion completion from partial observations, and motion synthesis from sparse key-frames. Even though, our model has not been trained for any of these tasks specifically, it provides superior performance than task-specific alternatives. Our general-purpose human motion prior model can fix corrupted human body animations and generate complete movements from incomplete observations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Task-Generic Hierarchical Human Motion Prior using VAEs<br>pdf: <a href="https://t.co/J2PJhZokCy">https://t.co/J2PJhZokCy</a><br>abs: <a href="https://t.co/e3LynnVEBD">https://t.co/e3LynnVEBD</a> <a href="https://t.co/GlpJtrINdf">pic.twitter.com/GlpJtrINdf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402476379036397568?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural  Networks for Inverse Molecular Design

AkshatKumar Nigam, Robert Pollice, Alan Aspuru-Guzik

- retweets: 30, favorites: 60 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04011) | [pdf](https://arxiv.org/pdf/2106.04011)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Inverse molecular design, i.e., designing molecules with specific target properties, can be posed as an optimization problem. High-dimensional optimization tasks in the natural sciences are commonly tackled via population-based metaheuristic optimization algorithms such as evolutionary algorithms. However, expensive property evaluation, which is often required, can limit the widespread use of such approaches as the associated cost can become prohibitive. Herein, we present JANUS, a genetic algorithm that is inspired by parallel tempering. It propagates two populations, one for exploration and another for exploitation, improving optimization by reducing expensive property evaluations. Additionally, JANUS is augmented by a deep neural network that approximates molecular properties via active learning for enhanced sampling of the chemical space. Our method uses the SELFIES molecular representation and the STONED algorithm for the efficient generation of structures, and outperforms other generative models in common inverse molecular design tasks achieving state-of-the-art performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new article: <a href="https://t.co/CQKT7IxFKv">https://t.co/CQKT7IxFKv</a><br><br>We were inspired by parallel tempering, SELFIES &amp; S.T.O.N.E.D to create a new algorithm for inverse molecular design 😄<br><br>w/ <a href="https://twitter.com/robpollice?ref_src=twsrc%5Etfw">@robpollice</a>, <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a> <a href="https://t.co/3ZkObz5qU2">pic.twitter.com/3ZkObz5qU2</a></p>&mdash; Akshat Nigam (@akshat_ai) <a href="https://twitter.com/akshat_ai/status/1402446318312718336?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to present JANUS, a genetic algorithm for molecular design that propagates two generations, one explorative and one exploitative, and that can use a neural network for active learning and on-the-fly fitness estimation. <a href="https://t.co/9Km2AgsHzx">https://t.co/9Km2AgsHzx</a></p>&mdash; Robert Pollice (@robpollice) <a href="https://twitter.com/robpollice/status/1402615653476048899?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. NWT: Towards natural audio-to-video generation with representation  learning

Rayhane Mama, Marc S. Tyndel, Hashiam Kadhim, Cole Clifford, Ragavan Thurairatnam

- retweets: 20, favorites: 67 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04283) | [pdf](https://arxiv.org/pdf/2106.04283)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

In this work we introduce NWT, an expressive speech-to-video model. Unlike approaches that use domain-specific intermediate representations such as pose keypoints, NWT learns its own latent representations, with minimal assumptions about the audio and video content. To this end, we propose a novel discrete variational autoencoder with adversarial loss, dVAE-Adv, which learns a new discrete latent representation we call Memcodes. Memcodes are straightforward to implement, require no additional loss terms, are stable to train compared with other approaches, and show evidence of interpretability. To predict on the Memcode space, we use an autoregressive encoder-decoder model conditioned on audio. Additionally, our model can control latent attributes in the generated video that are not annotated in the data. We train NWT on clips from HBO's Last Week Tonight with John Oliver. NWT consistently scores above other approaches in Mean Opinion Score (MOS) on tests of overall video naturalness, facial naturalness and expressiveness, and lipsync quality. This work sets a strong baseline for generalized audio-to-video synthesis. Samples are available at https://next-week-tonight.github.io/NWT/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NWT: Towards natural audio-to-video generation with representation learning<br>pdf: <a href="https://t.co/cKCJplMpYO">https://t.co/cKCJplMpYO</a><br>abs: <a href="https://t.co/F0eE4r2R73">https://t.co/F0eE4r2R73</a><br>project page: <a href="https://t.co/mYp4Vgd9RL">https://t.co/mYp4Vgd9RL</a> <a href="https://t.co/5bPS6zi5Zf">pic.twitter.com/5bPS6zi5Zf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402439037831585795?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Image2Point: 3D Point-Cloud Understanding with Pretrained 2D ConvNets

Chenfeng Xu, Shijia Yang, Bohan Zhai, Bichen Wu, Xiangyu Yue, Wei Zhan, Peter Vajda, Kurt Keutzer, Masayoshi Tomizuka

- retweets: 24, favorites: 58 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04180) | [pdf](https://arxiv.org/pdf/2106.04180)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

3D point-clouds and 2D images are different visual representations of the physical world. While human vision can understand both representations, computer vision models designed for 2D image and 3D point-cloud understanding are quite different. Our paper investigates the potential for transferability between these two representations by empirically investigating whether this approach works, what factors affect the transfer performance, and how to make it work even better. We discovered that we can indeed use the same neural net model architectures to understand both images and point-clouds. Moreover, we can transfer pretrained weights from image models to point-cloud models with minimal effort. Specifically, based on a 2D ConvNet pretrained on an image dataset, we can transfer the image model to a point-cloud model by \textit{inflating} 2D convolutional filters to 3D then finetuning its input, output, and optionally normalization layers. The transferred model can achieve competitive performance on 3D point-cloud classification, indoor and driving scene segmentation, even beating a wide range of point-cloud models that adopt task-specific architectures and use a variety of tricks.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Image2Point: 3D Point-Cloud Understanding with<br>Pretrained 2D ConvNets<a href="https://t.co/09eVBMA1Y6">https://t.co/09eVBMA1Y6</a><br>面白い、適当に実験するか <a href="https://t.co/rU8jCdthkI">pic.twitter.com/rU8jCdthkI</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1402518336756326400?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Flow Network based Generative Models for Non-Iterative Diverse Candidate  Generation

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio

- retweets: 32, favorites: 50 (06/10/2021 18:07:51)

- links: [abs](https://arxiv.org/abs/2106.04399) | [pdf](https://arxiv.org/pdf/2106.04399)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper is about the problem of learning a stochastic policy for generating an object (like a molecular graph) from a sequence of actions, such that the probability of generating an object is proportional to a given positive reward for that object. Whereas standard return maximization tends to converge to a single return-maximizing sequence, there are cases where we would like to sample a diverse set of high-return solutions. These arise, for example, in black-box function optimization when few rounds are possible, each with large batches of queries, where the batches should be diverse, e.g., in the design of new molecules. One can also see this as a problem of approximately converting an energy function to a generative distribution. While MCMC methods can achieve that, they are expensive and generally only perform local exploration. Instead, training a generative policy amortizes the cost of search during training and yields to fast generation. Using insights from Temporal Difference learning, we propose GFlowNet, based on a view of the generative process as a flow network, making it possible to handle the tricky case where different trajectories can yield the same final state, e.g., there are many ways to sequentially add atoms to generate some molecular graph. We cast the set of trajectories as a flow and convert the flow consistency equations into a learning objective, akin to the casting of the Bellman equations into Temporal Difference methods. We prove that any global minimum of the proposed objectives yields a policy which samples from the desired distribution, and demonstrate the improved performance and diversity of GFlowNet on a simple domain where there are many modes to the reward function, and on a molecule synthesis task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ever wanted to generate diverse samples of discrete data based on a reward function? Our new method, GFlowNet, based on flow networks &amp; a TD-like objective, gets great results on a molecule generation domain 💊<br>paper:<a href="https://t.co/WZuQW3aetP">https://t.co/WZuQW3aetP</a> <a href="https://t.co/PfQSWLxJDX">pic.twitter.com/PfQSWLxJDX</a></p>&mdash; Emmanuel Bengio (@folinoid) <a href="https://twitter.com/folinoid/status/1402704471763922947?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. There Is No Turning Back: A Self-Supervised Approach for  Reversibility-Aware Reinforcement Learning

Nathan Grinsztajn, Johan Ferret, Olivier Pietquin, Philippe Preux, Matthieu Geist

- retweets: 40, favorites: 24 (06/10/2021 18:07:52)

- links: [abs](https://arxiv.org/abs/2106.04480) | [pdf](https://arxiv.org/pdf/2106.04480)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We propose to learn to distinguish reversible from irreversible actions for better informed decision-making in Reinforcement Learning (RL). From theoretical considerations, we show that approximate reversibility can be learned through a simple surrogate task: ranking randomly sampled trajectory events in chronological order. Intuitively, pairs of events that are always observed in the same order are likely to be separated by an irreversible sequence of actions. Conveniently, learning the temporal order of events can be done in a fully self-supervised way, which we use to estimate the reversibility of actions from experience, without any priors. We propose two different strategies that incorporate reversibility in RL agents, one strategy for exploration (RAE) and one strategy for control (RAC). We demonstrate the potential of reversibility-aware agents in several environments, including the challenging Sokoban game. In synthetic tasks, we show that we can learn control policies that never fail and reduce to zero the side-effects of interactions, even without access to the reward function.




# 22. Meta Learning for Knowledge Distillation

Wangchunshu Zhou, Canwen Xu, Julian McAuley

- retweets: 34, favorites: 29 (06/10/2021 18:07:52)

- links: [abs](https://arxiv.org/abs/2106.04570) | [pdf](https://arxiv.org/pdf/2106.04570)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present Meta Learning for Knowledge Distillation (MetaDistil), a simple yet effective alternative to traditional knowledge distillation (KD) methods where the teacher model is fixed during training. We show the teacher network can learn to better transfer knowledge to the student network (i.e., learning to teach) with the feedback from the performance of the distilled student network in a meta learning framework. Moreover, we introduce a pilot update mechanism to improve the alignment between the inner-learner and meta-learner in meta learning algorithms that focus on an improved inner-learner. Experiments on various benchmarks show that MetaDistil can yield significant improvements compared with traditional KD algorithms and is less sensitive to the choice of different student capacity and hyperparameters, facilitating the use of KD on different tasks and models. The code is available at https://github.com/JetRunner/MetaDistil

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/2) Excited to share our new paper on knowledge distillation:<br>Meta Learning for Knowledge Distillation (<a href="https://t.co/RPwB6Hba2P">https://t.co/RPwB6Hba2P</a>). We add some &quot;meta&quot; magic to KD and ask the student to do a quiz to provide feedback to the teacher!<br>w/ <a href="https://twitter.com/wangchunshu?ref_src=twsrc%5Etfw">@wangchunshu</a> and Julian McAuley <a href="https://t.co/anybEUSnfL">pic.twitter.com/anybEUSnfL</a></p>&mdash; Canwen Xu (@XuCanwen) <a href="https://twitter.com/XuCanwen/status/1402549755889164288?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 23. NISQ Algorithm for Semidefinite Programming

Kishor Bharti, Tobias Haug, Vlatko Vedral, Leong-Chuan Kwek

- retweets: 20, favorites: 41 (06/10/2021 18:07:52)

- links: [abs](https://arxiv.org/abs/2106.03891) | [pdf](https://arxiv.org/pdf/2106.03891)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

Semidefinite Programming (SDP) is a class of convex optimization programs with vast applications in control theory, quantum information, combinatorial optimization and operational research. Noisy intermediate-scale quantum (NISQ) algorithms aim to make an efficient use of the current generation of quantum hardware. However, optimizing variational quantum algorithms is a challenge as it is an NP-hard problem that in general requires an exponential time to solve and can contain many far from optimal local minima. Here, we present a current term NISQ algorithm for SDP. The classical optimization program of our NISQ solver is another SDP over a smaller dimensional ansatz space. We harness the SDP based formulation of the Hamiltonian ground state problem to design a NISQ eigensolver. Unlike variational quantum eigensolvers, the classical optimization program of our eigensolver is convex, can be solved in polynomial time with the number of ansatz parameters and every local minimum is a global minimum. Further, we demonstrate the potential of our NISQ SDP solver by finding the largest eigenvalue of up to $2^{1000}$ dimensional matrices and solving graph problems related to quantum contextuality. We also discuss NISQ algorithms for rank-constrained SDPs. Our work extends the application of NISQ computers onto one of the most successful algorithmic frameworks of the past few decades.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On arXiv today. We present a current term noisy intermediate-scale quantum algorithm for solving Semidefinite programs (SDP). <br>Side result: we present a hybrid quantum-classical eigensolver that does not suffer from the local minima problem.    (1/n)<a href="https://t.co/TX78dgk5m2">https://t.co/TX78dgk5m2</a> <a href="https://t.co/vpht0Jttmt">pic.twitter.com/vpht0Jttmt</a></p>&mdash; Kishor Bharti (@CQT_Kishor) <a href="https://twitter.com/CQT_Kishor/status/1402459817235533824?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. Parameter-efficient Multi-task Fine-tuning for Transformers via Shared  Hypernetworks

Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, James Henderson

- retweets: 25, favorites: 33 (06/10/2021 18:07:52)

- links: [abs](https://arxiv.org/abs/2106.04489) | [pdf](https://arxiv.org/pdf/2106.04489)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

State-of-the-art parameter-efficient fine-tuning methods rely on introducing adapter modules between the layers of a pretrained language model. However, such modules are trained separately for each task and thus do not enable sharing information across tasks. In this paper, we show that we can learn adapter parameters for all layers and tasks by generating them using shared hypernetworks, which condition on task, adapter position, and layer id in a transformer model. This parameter-efficient multi-task learning framework allows us to achieve the best of both worlds by sharing knowledge across tasks via hypernetworks while enabling the model to adapt to each individual task through task-specific adapters. Experiments on the well-known GLUE benchmark show improved performance in multi-task learning while adding only 0.29% parameters per task. We additionally demonstrate substantial performance improvements in few-shot domain generalization across a variety of tasks. Our code is publicly available in https://github.com/rabeehk/hyperformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks<br>pdf: <a href="https://t.co/HpsmADkHb4">https://t.co/HpsmADkHb4</a><br>abs: <a href="https://t.co/7PCgWT5bVl">https://t.co/7PCgWT5bVl</a><br><br>GLUE benchmark shows improved performance in multi-task learning while adding only 0.29% parameters per<br>task <a href="https://t.co/kmEpO1oTin">pic.twitter.com/kmEpO1oTin</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402430137531043842?ref_src=twsrc%5Etfw">June 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



