---
title: Hot Papers 2021-07-12
date: 2021-07-13T13:51:07.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. The Bayesian Learning Rule

Mohammad Emtiyaz Khan, Håvard Rue

- retweets: 22901, favorites: 4 (07/13/2021 13:51:07)

- links: [abs](https://arxiv.org/abs/2107.04562) | [pdf](https://arxiv.org/pdf/2107.04562)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We show that many machine-learning algorithms are specific instances of a single algorithm called the Bayesian learning rule. The rule, derived from Bayesian principles, yields a wide-range of algorithms from fields such as optimization, deep learning, and graphical models. This includes classical algorithms such as ridge regression, Newton's method, and Kalman filter, as well as modern deep-learning algorithms such as stochastic-gradient descent, RMSprop, and Dropout. The key idea in deriving such algorithms is to approximate the posterior using candidate distributions estimated by using natural gradients. Different candidate distributions result in different algorithms and further approximations to natural gradients give rise to variants of those algorithms. Our work not only unifies, generalizes, and improves existing algorithms, but also helps us design new ones.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper on &quot;The Bayesian Learning Rule&quot; is now on arXiv, where we provide a common learning-principle behind a variety of learning algorithms (optimization, deep learning, and graphical models). <a href="https://t.co/Kta3EGvWba">https://t.co/Kta3EGvWba</a><br>Guess what, the principle is Bayesian. A very long🧵 <a href="https://t.co/QI0FhUyGH8">pic.twitter.com/QI0FhUyGH8</a></p>&mdash; Emtiyaz Khan (@EmtiyazKhan) <a href="https://twitter.com/EmtiyazKhan/status/1414498922584711171?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. ViTGAN: Training GANs with Vision Transformers

Kwonjoon Lee, Huiwen Chang, Lu Jiang, Han Zhang, Zhuowen Tu, Ce Liu

- retweets: 2931, favorites: 372 (07/13/2021 13:51:08)

- links: [abs](https://arxiv.org/abs/2107.04589) | [pdf](https://arxiv.org/pdf/2107.04589)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Recently, Vision Transformers (ViTs) have shown competitive performance on image recognition while requiring less vision-specific inductive biases. In this paper, we investigate if such observation can be extended to image generation. To this end, we integrate the ViT architecture into generative adversarial networks (GANs). We observe that existing regularization methods for GANs interact poorly with self-attention, causing serious instability during training. To resolve this issue, we introduce novel regularization techniques for training GANs with ViTs. Empirically, our approach, named ViTGAN, achieves comparable performance to state-of-the-art CNN-based StyleGAN2 on CIFAR-10, CelebA, and LSUN bedroom datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ViTGAN: Training GANs with Vision Transformers<br><br>Achieves comparable performance to SotA CNN-based StyleGAN2 on CelebA and LSUN bedroom datasets by introducing a novel regularization techniques for training GANs with ViTs.<a href="https://t.co/J3egmevJT3">https://t.co/J3egmevJT3</a> <a href="https://t.co/FQYMqfHjoG">pic.twitter.com/FQYMqfHjoG</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1414383203301793794?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ViTGAN: Training GANs with Vision Transformers<br>pdf: <a href="https://t.co/aKdPs3NXIf">https://t.co/aKdPs3NXIf</a><br>abs: <a href="https://t.co/sk5jkcqvjM">https://t.co/sk5jkcqvjM</a><br><br>achieves comparable performance to state-of-the-art CNN-based StyleGAN2 on CIFAR-10, CelebA, and LSUN<br>bedroom datasets <a href="https://t.co/Ff3uu2jQ13">pic.twitter.com/Ff3uu2jQ13</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414383353386700804?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. UrbanScene3D: A Large Scale Urban Scene Dataset and Simulator

Yilin Liu, Fuyou Xue, Hui Huang

- retweets: 1158, favorites: 140 (07/13/2021 13:51:08)

- links: [abs](https://arxiv.org/abs/2107.04286) | [pdf](https://arxiv.org/pdf/2107.04286)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

The ability to perceive the environments in different ways is essential to robotic research. This involves the analysis of both 2D and 3D data sources. We present a large scale urban scene dataset associated with a handy simulator based on Unreal Engine 4 and AirSim, which consists of both man-made and real-world reconstruction scenes in different scales, referred to as UrbanScene3D. Unlike previous works that purely based on 2D information or man-made 3D CAD models, UrbanScene3D contains both compact man-made models and detailed real-world models reconstructed by aerial images. Each building has been manually extracted from the entire scene model and then has been assigned with a unique label, forming an instance segmentation map. The provided 3D ground-truth textured models with instance segmentation labels in UrbanScene3D allow users to obtain all kinds of data they would like to have: instance segmentation map, depth map in arbitrary resolution, 3D point cloud/mesh in both visible and invisible places, etc. In addition, with the help of AirSim, users can also simulate the robots (cars/drones)to test a variety of autonomous tasks in the proposed city environment. Please refer to our paper and website(https://vcc.tech/UrbanScene3D/) for further details and applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">UrbanScene3D: A Large Scale Urban Scene Dataset and Simulator<br>pdf: <a href="https://t.co/0eoNr2MlpQ">https://t.co/0eoNr2MlpQ</a><br>abs: <a href="https://t.co/iJyVZXitkY">https://t.co/iJyVZXitkY</a><br>project page: <a href="https://t.co/Dp8ymSn2Qz">https://t.co/Dp8ymSn2Qz</a><br>UrbanScene3D contains both compact man-made models and detailed real-world models reconstructed by aerial images <a href="https://t.co/Mllj3KfoD5">pic.twitter.com/Mllj3KfoD5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414388354238537728?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A Systematic Survey of Text Worlds as Embodied Natural Language  Environments

Peter A Jansen

- retweets: 133, favorites: 56 (07/13/2021 13:51:08)

- links: [abs](https://arxiv.org/abs/2107.04132) | [pdf](https://arxiv.org/pdf/2107.04132)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Text Worlds are virtual environments for embodied agents that, unlike 2D or 3D environments, are rendered exclusively using textual descriptions. These environments offer an alternative to higher-fidelity 3D environments due to their low barrier to entry, providing the ability to study semantics, compositional inference, and other high-level tasks with rich high-level action spaces while controlling for perceptual input. This systematic survey outlines recent developments in tooling, environments, and agent modeling for Text Worlds, while examining recent trends in knowledge graphs, common sense reasoning, transfer learning of Text World performance to higher-fidelity environments, as well as near-term development targets that, once achieved, make Text Worlds an attractive general research paradigm for natural language processing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;ve become interested in embodied virtual environments in general (and particularly Text Worlds, those rendered entirely in text). I wrote a survey article of ~100 articles to help me get onboard. Critical feedback welcome.<a href="https://t.co/H559U5Cn4N">https://t.co/H559U5Cn4N</a> <a href="https://t.co/C6CqUSW9Kn">pic.twitter.com/C6CqUSW9Kn</a></p>&mdash; Peter Jansen (@peterjansen_ai) <a href="https://twitter.com/peterjansen_ai/status/1414628757680230404?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Improved Language Identification Through Cross-Lingual Self-Supervised  Learning

Andros Tjandra, Diptanu Gon Choudhury, Frank Zhang, Kritika Singh, Alexei Baevski, Assaf Sela, Yatharth Saraf, Michael Auli

- retweets: 59, favorites: 49 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04082) | [pdf](https://arxiv.org/pdf/2107.04082)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Language identification greatly impacts the success of downstream tasks such as automatic speech recognition. Recently, self-supervised speech representations learned by wav2vec 2.0 have been shown to be very effective for a range of speech tasks. We extend previous self-supervised work on language identification by experimenting with pre-trained models which were learned on real-world unconstrained speech in multiple languages and not just on English. We show that models pre-trained on many languages perform better and enable language identification systems that require very little labeled data to perform well. Results on a 25 languages setup show that with only 10 minutes of labeled data per language, a cross-lingually pre-trained model can achieve over 93% accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improved Language Identification Through Cross-Lingual Self-Supervised Learning<br>pdf: <a href="https://t.co/L5OuaouKjb">https://t.co/L5OuaouKjb</a><br>abs: <a href="https://t.co/kOTirYnANK">https://t.co/kOTirYnANK</a><br>Results on a 25 languages setup, 10 minutes of labeled data per language, a cross-lingually pre-trained model can achieve over 93% accuracy <a href="https://t.co/1X9GdAMofu">pic.twitter.com/1X9GdAMofu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414386148022136836?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Sensitivity analysis in differentially private machine learning using  hybrid automatic differentiation

Alexander Ziller, Dmitrii Usynin, Moritz Knolle, Kritika Prakash, Andrew Trask, Rickmer Braren, Marcus Makowski, Daniel Rueckert, Georgios Kaissis

- retweets: 36, favorites: 42 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04265) | [pdf](https://arxiv.org/pdf/2107.04265)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.SC](https://arxiv.org/list/cs.SC/recent)

In recent years, formal methods of privacy protection such as differential privacy (DP), capable of deployment to data-driven tasks such as machine learning (ML), have emerged. Reconciling large-scale ML with the closed-form reasoning required for the principled analysis of individual privacy loss requires the introduction of new tools for automatic sensitivity analysis and for tracking an individual's data and their features through the flow of computation. For this purpose, we introduce a novel \textit{hybrid} automatic differentiation (AD) system which combines the efficiency of reverse-mode AD with an ability to obtain a closed-form expression for any given quantity in the computational graph. This enables modelling the sensitivity of arbitrary differentiable function compositions, such as the training of neural networks on private data. We demonstrate our approach by analysing the individual DP guarantees of statistical database queries. Moreover, we investigate the application of our technique to the training of DP neural networks. Our approach can enable the principled reasoning about privacy loss in the setting of data processing, and further the development of automatic sensitivity analysis and privacy budgeting systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are happy that our ICML TPDP Workshop paper on sensitivity analysis using automatic differentiation got accepted and is now available on arxiv.<a href="https://t.co/79NHARKlQA">https://t.co/79NHARKlQA</a><a href="https://twitter.com/GKaissis?ref_src=twsrc%5Etfw">@GKaissis</a> <a href="https://twitter.com/moritz_K_?ref_src=twsrc%5Etfw">@moritz_K_</a> <a href="https://twitter.com/kritipraks?ref_src=twsrc%5Etfw">@kritipraks</a> <a href="https://twitter.com/iamtrask?ref_src=twsrc%5Etfw">@iamtrask</a> <a href="https://twitter.com/BrarenRickmer?ref_src=twsrc%5Etfw">@BrarenRickmer</a> <a href="https://twitter.com/DanielRueckert?ref_src=twsrc%5Etfw">@DanielRueckert</a></p>&mdash; Alex Ziller (@a1302z) <a href="https://twitter.com/a1302z/status/1414488951273009155?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. First-Generation Inference Accelerator Deployment at Facebook

Michael Anderson, Benny Chen, Stephen Chen, Summer Deng, Jordan Fix, Michael Gschwind, Aravind Kalaiah, Changkyu Kim, Jaewon Lee, Jason Liang, Haixin Liu, Yinghai Lu, Jack Montgomery, Arun Moorthy, Satish Nadathur, Sam Naghshineh, Avinash Nayak, Jongsoo Park, Chris Petersen, Martin Schatz, Narayanan Sundaram, Bangsheng Tang, Peter Tang, Amy Yang, Jiecao Yu, Hector Yuen, Ying Zhang, Aravind Anbudurai, Vandana Balan, Harsha Bojja, Joe Boyd, Matthew Breitbach, Claudio Caldato, Anna Calvo, Garret Catron, Sneh Chandwani, Panos Christeas, Brad Cottel, Brian Coutinho, Arun Dalli, Abhishek Dhanotia, Oniel Duncan, Roman Dzhabarov, Simon Elmir, Chunli Fu, Wenyin Fu, Michael Fulthorp, Adi Gangidi, Nick Gibson, Sean Gordon, Beatriz Padilla Hernandez, Daniel Ho, Yu-Cheng Huang, Olof Johansson, Shishir Juluri

- retweets: 20, favorites: 55 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04140) | [pdf](https://arxiv.org/pdf/2107.04140)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

In this paper, we provide a deep dive into the deployment of inference accelerators at Facebook. Many of our ML workloads have unique characteristics, such as sparse memory accesses, large model sizes, as well as high compute, memory and network bandwidth requirements. We co-designed a high-performance, energy-efficient inference accelerator platform based on these requirements. We describe the inference accelerator platform ecosystem we developed and deployed at Facebook: both hardware, through Open Compute Platform (OCP), and software framework and tooling, through Pytorch/Caffe2/Glow. A characteristic of this ecosystem from the start is its openness to enable a variety of AI accelerators from different vendors. This platform, with six low-power accelerator cards alongside a single-socket host CPU, allows us to serve models of high complexity that cannot be easily or efficiently run on CPUs. We describe various performance optimizations, at both platform and accelerator level, which enables this platform to serve production traffic at Facebook. We also share deployment challenges, lessons learned during performance optimization, as well as provide guidance for future inference hardware co-design.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">First-Generation Inference Accelerator Deployment<br>at Facebook<br>pdf: <a href="https://t.co/w1RmCEQrby">https://t.co/w1RmCEQrby</a><br>abs: <a href="https://t.co/iQ8FoJUjkm">https://t.co/iQ8FoJUjkm</a><br>overview of Facebook’s hardware/software inference accelerator ecosystem and accelerator deployment <a href="https://t.co/Pfqb44uZNl">pic.twitter.com/Pfqb44uZNl</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414382942038761478?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Multiaccurate Proxies for Downstream Fairness

Emily Diana, Wesley Gill, Michael Kearns, Krishnaram Kenthapadi, Aaron Roth, Saeed Sharifi-Malvajerdi

- retweets: 36, favorites: 30 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04423) | [pdf](https://arxiv.org/pdf/2107.04423)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

We study the problem of training a model that must obey demographic fairness conditions when the sensitive features are not available at training time -- in other words, how can we train a model to be fair by race when we don't have data about race? We adopt a fairness pipeline perspective, in which an "upstream" learner that does have access to the sensitive features will learn a proxy model for these features from the other attributes. The goal of the proxy is to allow a general "downstream" learner -- with minimal assumptions on their prediction task -- to be able to use the proxy to train a model that is fair with respect to the true sensitive features. We show that obeying multiaccuracy constraints with respect to the downstream model class suffices for this purpose, and provide sample- and oracle efficient-algorithms and generalization bounds for learning such proxies. In general, multiaccuracy can be much easier to satisfy than classification accuracy, and can be satisfied even when the sensitive features are hard to predict.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You want to solve a downstream fair machine learning task wrt some protected attribute z, but don&#39;t have access to z. You might have to rely on a proxy for z. What properties do you want it to have? Multiaccuracy wrt error regions of downstream models: <a href="https://t.co/dymGEv5HWF">https://t.co/dymGEv5HWF</a></p>&mdash; Aaron Roth (@Aaroth) <a href="https://twitter.com/Aaroth/status/1414598666090946560?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. An ontology for the formalization and visualization of scientific  knowledge

Vincenzo Daponte, Gilles Falquet

- retweets: 53, favorites: 1 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04347) | [pdf](https://arxiv.org/pdf/2107.04347)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DL](https://arxiv.org/list/cs.DL/recent) | [cs.LO](https://arxiv.org/list/cs.LO/recent)

The construction of an ontology of scientific knowledge objects, presented here, is part of the development of an approach oriented towards the visualization of scientific knowledge. It is motivated by the fact that the concepts of organization of scientific knowledge (theorem, law, experience, proof, etc.) appear in existing ontologies but that none of them is centered on this topic and presents a simple and easily usable organization. We present the first version built from ontological sources (ontologies of knowledge objects of certain fields, lexical and higher level ones), specialized knowledge bases and interviews with scientists. We have aligned this ontology with some of the sources used, which has allowed us to verify its consistency with respect to them. The validation of the ontology consists in using it to formalize knowledge from various sources, which we have begun to do in the field of physics.




# 10. StyleCariGAN: Caricature Generation via StyleGAN Feature Map Modulation

Wonjong Jang, Gwangjin Ju, Yucheol Jung, Jiaolong Yang, Xin Tong, Seungyong Lee

- retweets: 6, favorites: 44 (07/13/2021 13:51:09)

- links: [abs](https://arxiv.org/abs/2107.04331) | [pdf](https://arxiv.org/pdf/2107.04331)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a caricature generation framework based on shape and style manipulation using StyleGAN. Our framework, dubbed StyleCariGAN, automatically creates a realistic and detailed caricature from an input photo with optional controls on shape exaggeration degree and color stylization type. The key component of our method is shape exaggeration blocks that are used for modulating coarse layer feature maps of StyleGAN to produce desirable caricature shape exaggerations. We first build a layer-mixed StyleGAN for photo-to-caricature style conversion by swapping fine layers of the StyleGAN for photos to the corresponding layers of the StyleGAN trained to generate caricatures. Given an input photo, the layer-mixed model produces detailed color stylization for a caricature but without shape exaggerations. We then append shape exaggeration blocks to the coarse layers of the layer-mixed model and train the blocks to create shape exaggerations while preserving the characteristic appearances of the input. Experimental results show that our StyleCariGAN generates realistic and detailed caricatures compared to the current state-of-the-art methods. We demonstrate StyleCariGAN also supports other StyleGAN-based image manipulations, such as facial expression control.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">StyleCariGAN: Caricature Generation via StyleGAN Feature Map Modulation<br>pdf: <a href="https://t.co/cUAy2bsphU">https://t.co/cUAy2bsphU</a><br>abs: <a href="https://t.co/rgCAM7GOqr">https://t.co/rgCAM7GOqr</a> <a href="https://t.co/Zs9aCGivvi">pic.twitter.com/Zs9aCGivvi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1414384217983758337?ref_src=twsrc%5Etfw">July 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



