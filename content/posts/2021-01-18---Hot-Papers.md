---
title: Hot Papers 2021-01-18
date: 2021-01-19T08:57:46.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Counterfactual Generative Networks

Axel Sauer, Andreas Geiger

- retweets: 576, favorites: 138 (01/19/2021 08:57:46)

- links: [abs](https://arxiv.org/abs/2101.06046) | [pdf](https://arxiv.org/pdf/2101.06046)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural networks are prone to learning shortcuts -- they often model simple correlations, ignoring more complex ones that potentially generalize better. Prior works on image classification show that instead of learning a connection to object shape, deep classifiers tend to exploit spurious correlations with low-level texture or the background for solving the classification task. In this work, we take a step towards more robust and interpretable classifiers that explicitly expose the task's causal structure. Building on current advances in deep generative modeling, we propose to decompose the image generation process into independent causal mechanisms that we train without direct supervision. By exploiting appropriate inductive biases, these mechanisms disentangle object shape, object texture, and background; hence, they allow for generating counterfactual images. We demonstrate the ability of our model to generate such images on MNIST and ImageNet. Further, we show that the counterfactual images can improve out-of-distribution robustness with a marginal drop in performance on the original classification task, despite being synthetic. Lastly, our generative model can be trained efficiently on a single GPU, exploiting common pre-trained models as inductive biases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Counterfactual Generative Networks<br>pdf: <a href="https://t.co/VRRIIMoSIU">https://t.co/VRRIIMoSIU</a><br>abs: <a href="https://t.co/QGdRnvsTXv">https://t.co/QGdRnvsTXv</a> <a href="https://t.co/FZnHRSaoYI">pic.twitter.com/FZnHRSaoYI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1350993987507802114?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Supervised Transfer Learning at Scale for Medical Imaging

Basil Mustafa, Aaron Loh, Jan Freyberg, Patricia MacWilliams, Alan Karthikesalingam, Neil Houlsby, Vivek Natarajan

- retweets: 132, favorites: 53 (01/19/2021 08:57:46)

- links: [abs](https://arxiv.org/abs/2101.05913) | [pdf](https://arxiv.org/pdf/2101.05913)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Transfer learning is a standard technique to improve performance on tasks with limited data. However, for medical imaging, the value of transfer learning is less clear. This is likely due to the large domain mismatch between the usual natural-image pre-training (e.g. ImageNet) and medical images. However, recent advances in transfer learning have shown substantial improvements from scale. We investigate whether modern methods can change the fortune of transfer learning for medical imaging. For this, we study the class of large-scale pre-trained networks presented by Kolesnikov et al. on three diverse imaging tasks: chest radiography, mammography, and dermatology. We study both transfer performance and critical properties for the deployment in the medical domain, including: out-of-distribution generalization, data-efficiency, sub-group fairness, and uncertainty estimation. Interestingly, we find that for some of these properties transfer from natural to medical images is indeed extremely effective, but only when performed at sufficient scale.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper from our team <a href="https://twitter.com/GoogleHealth?ref_src=twsrc%5Etfw">@GoogleHealth</a>/<a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> (<a href="https://t.co/va8o7COLib">https://t.co/va8o7COLib</a>) Pre-training at scale improves AI accuracy, generalisation + fairness in many medical imaging tasks: Chest X-Ray, Dermatology &amp; Mammography! Led by <a href="https://twitter.com/_basilM?ref_src=twsrc%5Etfw">@_basilM</a>, <a href="https://twitter.com/JanFreyberg?ref_src=twsrc%5Etfw">@JanFreyberg</a>, Aaron Loh, <a href="https://twitter.com/neilhoulsby?ref_src=twsrc%5Etfw">@neilhoulsby</a>, <a href="https://twitter.com/vivnat?ref_src=twsrc%5Etfw">@vivnat</a> <a href="https://t.co/zncpXLnKhD">pic.twitter.com/zncpXLnKhD</a></p>&mdash; Alan Karthikesalingam (@alan_karthi) <a href="https://twitter.com/alan_karthi/status/1351114728907665408?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. The Geometry of Deep Generative Image Models and its Applications

Binxu Wang, Carlos R. Ponce

- retweets: 90, favorites: 65 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06006) | [pdf](https://arxiv.org/pdf/2101.06006)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [math.NA](https://arxiv.org/list/math.NA/recent)

Generative adversarial networks (GANs) have emerged as a powerful unsupervised method to model the statistical patterns of real-world data sets, such as natural images. These networks are trained to map random inputs in their latent space to new samples representative of the learned data. However, the structure of the latent space is hard to intuit due to its high dimensionality and the non-linearity of the generator, which limits the usefulness of the models. Understanding the latent space requires a way to identify input codes for existing real-world images (inversion), and a way to identify directions with known image transformations (interpretability). Here, we use a geometric framework to address both issues simultaneously. We develop an architecture-agnostic method to compute the Riemannian metric of the image manifold created by GANs. The eigen-decomposition of the metric isolates axes that account for different levels of image variability. An empirical analysis of several pretrained GANs shows that image variation around each position is concentrated along surprisingly few major axes (the space is highly anisotropic) and the directions that create this large variation are similar at different positions in the space (the space is homogeneous). We show that many of the top eigenvectors correspond to interpretable transforms in the image space, with a substantial part of eigenspace corresponding to minor transforms which could be compressed out. This geometric understanding unifies key previous results related to GAN interpretability. We show that the use of this metric allows for more efficient optimization in the latent space (e.g. GAN inversion) and facilitates unsupervised discovery of interpretable axes. Our results illustrate that defining the geometry of the GAN image manifold can serve as a general framework for understanding GANs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Geometry of Deep Generative Image Models and its Applications<br>pdf: <a href="https://t.co/MXi8fsBOBa">https://t.co/MXi8fsBOBa</a><br>abs: <a href="https://t.co/g8GL4Xtvj5">https://t.co/g8GL4Xtvj5</a> <a href="https://t.co/hNTzVwy9jY">pic.twitter.com/hNTzVwy9jY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1350991169468116994?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. SimGAN: Hybrid Simulator Identification for Domain Adaptation via  Adversarial Reinforcement Learning

Yifeng Jiang, Tingnan Zhang, Daniel Ho, Yunfei Bai, C. Karen Liu, Sergey Levine, Jie Tan

- retweets: 100, favorites: 41 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06005) | [pdf](https://arxiv.org/pdf/2101.06005)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

As learning-based approaches progress towards automating robot controllers design, transferring learned policies to new domains with different dynamics (e.g. sim-to-real transfer) still demands manual effort. This paper introduces SimGAN, a framework to tackle domain adaptation by identifying a hybrid physics simulator to match the simulated trajectories to the ones from the target domain, using a learned discriminative loss to address the limitations associated with manual loss design. Our hybrid simulator combines neural networks and traditional physics simulaton to balance expressiveness and generalizability, and alleviates the need for a carefully selected parameter set in System ID. Once the hybrid simulator is identified via adversarial reinforcement learning, it can be used to refine policies for the target domain, without the need to collect more data. We show that our approach outperforms multiple strong baselines on six robotic locomotion tasks for domain adaptation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial Reinforcement Learning<br>pdf: <a href="https://t.co/rLjq2WgAwU">https://t.co/rLjq2WgAwU</a><br>abs: <a href="https://t.co/CbAuEAapuC">https://t.co/CbAuEAapuC</a> <a href="https://t.co/EjQcP0rIUn">pic.twitter.com/EjQcP0rIUn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1350989334653128706?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The Challenge of Value Alignment: from Fairer Algorithms to AI Safety

Iason Gabriel, Vafa Ghazavi

- retweets: 72, favorites: 20 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06060) | [pdf](https://arxiv.org/pdf/2101.06060)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

This paper addresses the question of how to align AI systems with human values and situates it within a wider body of thought regarding technology and value. Far from existing in a vacuum, there has long been an interest in the ability of technology to 'lock-in' different value systems. There has also been considerable thought about how to align technologies with specific social values, including through participatory design-processes. In this paper we look more closely at the question of AI value alignment and suggest that the power and autonomy of AI systems gives rise to opportunities and challenges in the domain of value that have not been encountered before. Drawing important continuities between the work of the fairness, accountability, transparency and ethics community, and work being done by technical AI safety researchers, we suggest that more attention needs to be paid to the question of 'social value alignment' - that is, how to align AI systems with the plurality of values endorsed by groups of people, especially on the global level.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">✨Considerations of fair process and epistemic virtue point toward the need for a properly inclusive discussion around the ethics of AI alignment✨<br><br>Check out this new survey piece, co-authored with <a href="https://twitter.com/GhazaviVD?ref_src=twsrc%5Etfw">@GhazaviVD</a> + forthcoming in an OUP volume <a href="https://twitter.com/carissaveliz?ref_src=twsrc%5Etfw">@carissaveliz</a>:<a href="https://t.co/SZXR85GBoC">https://t.co/SZXR85GBoC</a></p>&mdash; Iason Gabriel (@IasonGabriel) <a href="https://twitter.com/IasonGabriel/status/1351165148875132931?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Nets with Mana: A Framework for Chemical Reaction Modelling

Fabrizio Romano Genovese, Fosco Loregian, Daniele Palombi

- retweets: 30, favorites: 51 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06234) | [pdf](https://arxiv.org/pdf/2101.06234)
- [math.CT](https://arxiv.org/list/math.CT/recent) | [cs.FL](https://arxiv.org/list/cs.FL/recent) | [q-bio.MN](https://arxiv.org/list/q-bio.MN/recent)

We use categorical methods to define a new flavor of Petri nets which could be useful in modelling chemical reactions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Borrowing the terminology from the popular Turing machine Magic:The gathering [19, 6] we propose a possible solution to this problem by endowing transitions in a [Petri] net with mana&quot;<a href="https://t.co/2acuWyfp6b">https://t.co/2acuWyfp6b</a><br><br>[19] is Wikipedia<br>[6] is arXiv:1904.09828</p>&mdash; theHigherGeometer (@HigherGeometer) <a href="https://twitter.com/HigherGeometer/status/1351102402552848391?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Temporal-Relational CrossTransformers for Few-Shot Action Recognition

Toby Perrett, Alessandro Masullo, Tilo Burghardt, Majid Mirmehdi, Dima Damen

- retweets: 27, favorites: 28 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06184) | [pdf](https://arxiv.org/pdf/2101.06184)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel approach to few-shot action recognition, finding temporally-corresponding frame tuples between the query and videos in the support set. Distinct from previous few-shot action recognition works, we construct class prototypes using the CrossTransformer attention mechanism to observe relevant sub-sequences of all support videos, rather than using class averages or single best matches. Video representations are formed from ordered tuples of varying numbers of frames, which allows sub-sequences of actions at different speeds and temporal offsets to be compared.   Our proposed Temporal-Relational CrossTransformers achieve state-of-the-art results on both Kinetics and Something-Something V2 (SSv2), outperforming prior work on SSv2 by a wide margin (6.8%) due to the method's ability to model temporal relations. A detailed ablation showcases the importance of matching to multiple support set videos and learning higher-order relational CrossTransformers. Code is available at https://github.com/tobyperrett/trx

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Temporal-Relational CrossTransformers for Few-Shot Action Recognition<br>pdf: <a href="https://t.co/H6qZrr5Fc9">https://t.co/H6qZrr5Fc9</a><br>abs: <a href="https://t.co/o0LqPP4agY">https://t.co/o0LqPP4agY</a><br>github: <a href="https://t.co/9hwa9nNWuM">https://t.co/9hwa9nNWuM</a> <a href="https://t.co/BjXyFJneDZ">pic.twitter.com/BjXyFJneDZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1350988415945338881?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Reasoning over Vision and Language: Exploring the Benefits of  Supplemental Knowledge

Violetta Shevchenko, Damien Teney, Anthony Dick, Anton van den Hengel

- retweets: 20, favorites: 34 (01/19/2021 08:57:47)

- links: [abs](https://arxiv.org/abs/2101.06013) | [pdf](https://arxiv.org/pdf/2101.06013)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The limits of applicability of vision-and-language models are defined by the coverage of their training data. Tasks like vision question answering (VQA) often require commonsense and factual information beyond what can be learned from task-specific datasets. This paper investigates the injection of knowledge from general-purpose knowledge bases (KBs) into vision-and-language transformers. We use an auxiliary training objective that encourages the learned representations to align with graph embeddings of matching entities in a KB. We empirically study the relevance of various KBs to multiple tasks and benchmarks. The technique brings clear benefits to knowledge-demanding question answering tasks (OK-VQA, FVQA) by capturing semantic and relational knowledge absent from existing models. More surprisingly, the technique also benefits visual reasoning tasks (NLVR2, SNLI-VE). We perform probing experiments and show that the injection of additional knowledge regularizes the space of embeddings, which improves the representation of lexical and semantic similarities. The technique is model-agnostic and can expand the applicability of any vision-and-language transformer with minimal computational overhead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here&#39;s a new general-purpose technique to inject knowledge in transformers for vision and language. It helps with question answering but also (surprisingly) with visual reasoning, suggesting subtle regularization effects from the additional knowledge.<a href="https://t.co/Y2bW0aMjoc">https://t.co/Y2bW0aMjoc</a> <a href="https://t.co/o9DVQycRoo">pic.twitter.com/o9DVQycRoo</a></p>&mdash; Damien Teney (@DamienTeney) <a href="https://twitter.com/DamienTeney/status/1350992510756257794?ref_src=twsrc%5Etfw">January 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



