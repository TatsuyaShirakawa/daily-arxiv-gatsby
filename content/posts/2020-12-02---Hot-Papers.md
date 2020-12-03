---
title: Hot Papers 2020-12-02
date: 2020-12-03T11:30:37.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Byzantine Eventual Consistency and the Fundamental Limits of  Peer-to-Peer Databases

Martin Kleppmann, Heidi Howard

- retweets: 323, favorites: 125 (12/03/2020 11:30:37)

- links: [abs](https://arxiv.org/abs/2012.00472) | [pdf](https://arxiv.org/pdf/2012.00472)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent)

Sybil attacks, in which a large number of adversary-controlled nodes join a network, are a concern for many peer-to-peer database systems, necessitating expensive countermeasures such as proof-of-work. However, there is a category of database applications that are, by design, immune to Sybil attacks because they can tolerate arbitrary numbers of Byzantine-faulty nodes. In this paper, we characterize this category of applications using a consistency model we call Byzantine Eventual Consistency (BEC). We introduce an algorithm that guarantees BEC based on Byzantine causal broadcast, prove its correctness, and demonstrate near-optimal performance in a prototype implementation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! 😎 In which <a href="https://twitter.com/heidiann360?ref_src=twsrc%5Etfw">@heidiann360</a> and I explore Git-like hash graphs, Bloom filters, and peer-to-peer systems that are immune to Sybil attacks.<br><br>📄 Paper: <a href="https://t.co/kDXe1zIkaK">https://t.co/kDXe1zIkaK</a><br>📎 Blog post: <a href="https://t.co/pHwdjFGi7Y">https://t.co/pHwdjFGi7Y</a></p>&mdash; Martin Kleppmann (@martinkl) <a href="https://twitter.com/martinkl/status/1334166327750221827?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. We are More than Our Joints: Predicting how 3D Bodies Move

Yan Zhang, Michael J. Black, Siyu Tang

- retweets: 289, favorites: 67 (12/03/2020 11:30:37)

- links: [abs](https://arxiv.org/abs/2012.00619) | [pdf](https://arxiv.org/pdf/2012.00619)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A key step towards understanding human behavior is the prediction of 3D human motion. Successful solutions have many applications in human tracking, HCI, and graphics. Most previous work focuses on predicting a time series of future 3D joint locations given a sequence 3D joints from the past. This Euclidean formulation generally works better than predicting pose in terms of joint rotations. Body joint locations, however, do not fully constrain 3D human pose, leaving degrees of freedom undefined, making it hard to animate a realistic human from only the joints. Note that the 3D joints can be viewed as a sparse point cloud. Thus the problem of human motion prediction can be seen as point cloud prediction. With this observation, we instead predict a sparse set of locations on the body surface that correspond to motion capture markers. Given such markers, we fit a parametric body model to recover the 3D shape and pose of the person. These sparse surface markers also carry detailed information about human movement that is not present in the joints, increasing the naturalness of the predicted motions. Using the AMASS dataset, we train MOJO, which is a novel variational autoencoder that generates motions from latent frequencies. MOJO preserves the full temporal resolution of the input motion, and sampling from the latent frequencies explicitly introduces high-frequency components into the generated motion. We note that motion prediction methods accumulate errors over time, resulting in joints or markers that diverge from true human bodies. To address this, we fit SMPL-X to the predictions at each time step, projecting the solution back onto the space of valid bodies. These valid markers are then propagated in time. Experiments show that our method produces state-of-the-art results and realistic 3D body animations. The code for research purposes is at https://yz-cnsdqz.github.io/MOJO/MOJO.html

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are More than Our Joints: Predicting how 3D Bodies Move<br>pdf: <a href="https://t.co/aiHQBPVio1">https://t.co/aiHQBPVio1</a><br>abs: <a href="https://t.co/0GRLJv0f4p">https://t.co/0GRLJv0f4p</a><br>project page: <a href="https://t.co/YrPIU5R8Ov">https://t.co/YrPIU5R8Ov</a> <a href="https://t.co/3PUhBNKSHC">pic.twitter.com/3PUhBNKSHC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333969080068546560?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Low Bandwidth Video-Chat Compression using Deep Generative Models

Maxime Oquab, Pierre Stock, Oran Gafni, Daniel Haziza, Tao Xu, Peizhao Zhang, Onur Celebi, Yana Hasson, Patrick Labatut, Bobo Bose-Kolanu, Thibault Peyronel, Camille Couprie

- retweets: 225, favorites: 65 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00328) | [pdf](https://arxiv.org/pdf/2012.00328)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

To unlock video chat for hundreds of millions of people hindered by poor connectivity or unaffordable data costs, we propose to authentically reconstruct faces on the receiver's device using facial landmarks extracted at the sender's side and transmitted over the network. In this context, we discuss and evaluate the benefits and disadvantages of several deep adversarial approaches. In particular, we explore quality and bandwidth trade-offs for approaches based on static landmarks, dynamic landmarks or segmentation maps. We design a mobile-compatible architecture based on the first order animation model of Siarohin et al. In addition, we leverage SPADE blocks to refine results in important areas such as the eyes and lips. We compress the networks down to about 3MB, allowing models to run in real time on iPhone 8 (CPU). This approach enables video calling at a few kbits per second, an order of magnitude lower than currently available alternatives.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Low Bandwidth Video-Chat Compression using Deep Generative Models<br>pdf: <a href="https://t.co/JCICbRGVki">https://t.co/JCICbRGVki</a><br>abs: <a href="https://t.co/RuQpZmra73">https://t.co/RuQpZmra73</a> <a href="https://t.co/HASR5RE6Pr">pic.twitter.com/HASR5RE6Pr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333971408746057728?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Natural Evolutionary Strategies for Variational Quantum Computation

Abhinav Anand, Matthias Degroote, Alán Aspuru-Guzik

- retweets: 132, favorites: 55 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00101) | [pdf](https://arxiv.org/pdf/2012.00101)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Natural evolutionary strategies (NES) are a family of gradient-free black-box optimization algorithms. This study illustrates their use for the optimization of randomly-initialized parametrized quantum circuits (PQCs) in the region of vanishing gradients. We show that using the NES gradient estimator the exponential decrease in variance can be alleviated. We implement two specific approaches, the exponential and separable natural evolutionary strategies, for parameter optimization of PQCs and compare them against standard gradient descent. We apply them to two different problems of ground state energy estimation using variational quantum eigensolver (VQE) and state preparation with circuits of varying depth and length. We also introduce batch optimization for circuits with larger depth to extend the use of evolutionary strategies to a larger number of parameters. We achieve accuracy comparable to state-of-the-art optimization techniques in all the above cases with a lower number of circuit evaluations. Our empirical results indicate that one can use NES as a hybrid tool in tandem with other gradient-based methods for optimization of deep quantum circuits in regions with vanishing gradients.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We posted a new preprint <a href="https://t.co/L7neYHmOEa">https://t.co/L7neYHmOEa</a> today investigating the use of Natural Evolutionary Strategies for optimization of parameterized quantum circuits. <a href="https://twitter.com/whynotquantum?ref_src=twsrc%5Etfw">@whynotquantum</a> <a href="https://twitter.com/A_Aspuru_Guzik?ref_src=twsrc%5Etfw">@A_Aspuru_Guzik</a> <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a></p>&mdash; Abhinav Anand (@theanandabhinav) <a href="https://twitter.com/theanandabhinav/status/1333982054980218880?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Civic Technologies: Research, Practice and Open Challenges

Pablo Aragon, Adriana Alvarado Garcia, Christopher A. Le Dantec, Claudia Flores-Saviaga, Jorge Saldivar

- retweets: 132, favorites: 23 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00515) | [pdf](https://arxiv.org/pdf/2012.00515)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

Over the last years, civic technology projects have emerged around the world to advance open government and community action. Although Computer-Supported Cooperative Work (CSCW) and Human-Computer Interaction (HCI) communities have shown a growing interest in researching issues around civic technologies, yet most research still focuses on projects from the Global North. The goal of this workshop is, therefore, to advance CSCW research by raising awareness for the ongoing challenges and open questions around civic technology by bridging the gap between researchers and practitioners from different regions.   The workshop will be organized around three central topics: (1) discuss how the local context and infrastructure affect the design, implementation, adoption, and maintenance of civic technology; (2) identify key elements of the configuration of trust among government, citizenry, and local organizations and how these elements change depending on the sociopolitical context where community engagement takes place; (3) discover what methods and strategies are best suited for conducting research on civic technologies in different contexts. These core topics will be covered across sessions that will initiate in-depth discussions and, thereby, stimulate collaboration between the CSCW research community and practitioners of civic technologies from both Global North and South.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The report of our <a href="https://twitter.com/hashtag/CSCW2020?src=hash&amp;ref_src=twsrc%5Etfw">#CSCW2020</a> workshop in <a href="https://twitter.com/hashtag/CivicTech?src=hash&amp;ref_src=twsrc%5Etfw">#CivicTech</a> with the proposal, outcome and position papers is already available on <a href="https://twitter.com/arxiv?ref_src=twsrc%5Etfw">@arxiv</a> 🚀<br><br>📖 <a href="https://t.co/q1Fsj7ehNI">https://t.co/q1Fsj7ehNI</a><br>🌐 <a href="https://t.co/ynUmAm4O8v">https://t.co/ynUmAm4O8v</a> <a href="https://t.co/wRbMf7ra5F">pic.twitter.com/wRbMf7ra5F</a></p>&mdash; Pablo Aragón (@elaragon) <a href="https://twitter.com/elaragon/status/1334180301799641090?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution

Kelvin C.K. Chan, Xintao Wang, Xiangyu Xu, Jinwei Gu, Chen Change Loy

- retweets: 72, favorites: 74 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00739) | [pdf](https://arxiv.org/pdf/2012.00739)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We show that pre-trained Generative Adversarial Networks (GANs), e.g., StyleGAN, can be used as a latent bank to improve the restoration quality of large-factor image super-resolution (SR). While most existing SR approaches attempt to generate realistic textures through learning with adversarial loss, our method, Generative LatEnt bANk (GLEAN), goes beyond existing practices by directly leveraging rich and diverse priors encapsulated in a pre-trained GAN. But unlike prevalent GAN inversion methods that require expensive image-specific optimization at runtime, our approach only needs a single forward pass to generate the upscaled image. GLEAN can be easily incorporated in a simple encoder-bank-decoder architecture with multi-resolution skip connections. Switching the bank allows the method to deal with images from diverse categories, e.g., cat, building, human face, and car. Images upscaled by GLEAN show clear improvements in terms of fidelity and texture faithfulness in comparison to existing methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution<br>pdf: <a href="https://t.co/miiNZcFhQQ">https://t.co/miiNZcFhQQ</a><br>abs: <a href="https://t.co/4EFOHaPEVU">https://t.co/4EFOHaPEVU</a><br>project page: <a href="https://t.co/Cih4BPvIoe">https://t.co/Cih4BPvIoe</a> <a href="https://t.co/uEnb902Nbe">pic.twitter.com/uEnb902Nbe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333955873685839874?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. 6.7ms on Mobile with over 78% ImageNet Accuracy: Unified Network Pruning  and Architecture Search for Beyond Real-Time Mobile Acceleration

Zhengang Li, Geng Yuan, Wei Niu, Yanyu Li, Pu Zhao, Yuxuan Cai, Xuan Shen, Zheng Zhan, Zhenglun Kong, Qing Jin, Zhiyu Chen, Sijia Liu, Kaiyuan Yang, Bin Ren, Yanzhi Wang, Xue Lin

- retweets: 90, favorites: 43 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00596) | [pdf](https://arxiv.org/pdf/2012.00596)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

With the increasing demand to efficiently deploy DNNs on mobile edge devices, it becomes much more important to reduce unnecessary computation and increase the execution speed. Prior methods towards this goal, including model compression and network architecture search (NAS), are largely performed independently and do not fully consider compiler-level optimizations which is a must-do for mobile acceleration. In this work, we first propose (i) a general category of fine-grained structured pruning applicable to various DNN layers, and (ii) a comprehensive, compiler automatic code generation framework supporting different DNNs and different pruning schemes, which bridge the gap of model compression and NAS. We further propose NPAS, a compiler-aware unified network pruning, and architecture search. To deal with large search space, we propose a meta-modeling procedure based on reinforcement learning with fast evaluation and Bayesian optimization, ensuring the total number of training epochs comparable with representative NAS frameworks. Our framework achieves 6.7ms, 5.9ms, 3.9ms ImageNet inference times with 78.2%, 75% (MobileNet-V3 level), and 71% (MobileNet-V2 level) Top-1 accuracy respectively on an off-the-shelf mobile phone, consistently outperforming prior work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">6.7ms on Mobile with over 78% ImageNet Accuracy: Unified Network Pruning and Architecture Search for Beyond Real-Time Mobile Acceleration<br>pdf: <a href="https://t.co/sLDXw8PW7E">https://t.co/sLDXw8PW7E</a><br>abs: <a href="https://t.co/sh45urhoix">https://t.co/sh45urhoix</a> <a href="https://t.co/LC82OinXjJ">pic.twitter.com/LC82OinXjJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333975758449078274?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. CPM: A Large-scale Generative Chinese Pre-trained Language Model

Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun

- retweets: 30, favorites: 52 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00413) | [pdf](https://arxiv.org/pdf/2012.00413)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB training data, drew a lot of attention due to the capacity of few-shot (even zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is still challenging, as the training corpus of GPT-3 is primarily English, and the parameters are not publicly available. In this technical report, we release the Chinese Pre-trained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best of our knowledge, CPM, with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pre-trained language model, which could facilitate several downstream Chinese NLP tasks, such as conversation, essay generation, cloze test, and language understanding. Extensive experiments demonstrate that CPM achieves strong performance on many NLP tasks in the settings of few-shot (even zero-shot) learning. The code and parameters are available at https://github.com/TsinghuaAI/CPM-Generate.




# 9. Unpaired Image-to-Image Translation via Latent Energy Transport

Yang Zhao, Changyou Chen

- retweets: 30, favorites: 50 (12/03/2020 11:30:38)

- links: [abs](https://arxiv.org/abs/2012.00649) | [pdf](https://arxiv.org/pdf/2012.00649)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Image-to-image translation aims to preserve source contents while translating to discriminative target styles between two visual domains. Most works apply adversarial learning in the ambient image space, which could be computationally expensive and challenging to train. In this paper, we propose to deploy an energy-based model (EBM) in the latent space of a pretrained autoencoder for this task. The pretrained autoencoder serves as both a latent code extractor and an image reconstruction worker. Our model is based on the assumption that two domains share the same latent space, where latent representation is implicitly decomposed as a content code and a domain-specific style code. Instead of explicitly extracting the two codes and applying adaptive instance normalization to combine them, our latent EBM can implicitly learn to transport the source style code to the target style code while preserving the content code, which is an advantage over existing image translation methods. This simplified solution also brings us far more efficiency in the one-sided unpaired image translation setting. Qualitative and quantitative comparisons demonstrate superior translation quality and faithfulness for content preservation. To the best of our knowledge, our model is the first to be applicable to 1024$\times$1024-resolution unpaired image translation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unpaired Image-to-Image Translation via Latent Energy Transport<br>pdf: <a href="https://t.co/vDootk2vxJ">https://t.co/vDootk2vxJ</a><br>abs: <a href="https://t.co/XosgdgcZ99">https://t.co/XosgdgcZ99</a> <a href="https://t.co/q6rV4VmCV7">pic.twitter.com/q6rV4VmCV7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1333978731044622338?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Message Passing Networks for Molecules with Tetrahedral Chirality

Lagnajit Pattanaik, Octavian E. Ganea, Ian Coley, Klavs F. Jensen, William H. Green, Connor W. Coley

- retweets: 12, favorites: 63 (12/03/2020 11:30:39)

- links: [abs](https://arxiv.org/abs/2012.00094) | [pdf](https://arxiv.org/pdf/2012.00094)
- [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Molecules with identical graph connectivity can exhibit different physical and biological properties if they exhibit stereochemistry-a spatial structural characteristic. However, modern neural architectures designed for learning structure-property relationships from molecular structures treat molecules as graph-structured data and therefore are invariant to stereochemistry. Here, we develop two custom aggregation functions for message passing neural networks to learn properties of molecules with tetrahedral chirality, one common form of stereochemistry. We evaluate performance on synthetic data as well as a newly-proposed protein-ligand docking dataset with relevance to drug discovery. Results show modest improvements over a baseline sum aggregator, highlighting opportunities for further architecture development.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our work accepted to the <a href="https://twitter.com/hashtag/ML4Molecules?src=hash&amp;ref_src=twsrc%5Etfw">#ML4Molecules</a> workshop at <a href="https://twitter.com/NeurIPSConf?ref_src=twsrc%5Etfw">@NeurIPSConf</a> 🥳🎉 We develop GNN aggregation functions for molecules with tetrahedral chirality--a first attempt at tackling chirality with 2D graph networks <br>Paper: <a href="https://t.co/ubCEWf4835">https://t.co/ubCEWf4835</a><br>Code: <a href="https://t.co/OIK8Ud8rla">https://t.co/OIK8Ud8rla</a></p>&mdash; Lucky Pattanaik (@lucky_pattanaik) <a href="https://twitter.com/lucky_pattanaik/status/1334167258176819206?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint led by <a href="https://twitter.com/lucky_pattanaik?ref_src=twsrc%5Etfw">@lucky_pattanaik</a> on developing asymmetric aggregation functions for message passing networks to start exploring molecular representations between 2D graphs and 3D conformers | <a href="https://t.co/4hafap7lfl">https://t.co/4hafap7lfl</a> <a href="https://twitter.com/hashtag/ML4Molecules?src=hash&amp;ref_src=twsrc%5Etfw">#ML4Molecules</a> <a href="https://t.co/stQzNcKCfB">https://t.co/stQzNcKCfB</a></p>&mdash; Connor W. Coley (@cwcoley) <a href="https://twitter.com/cwcoley/status/1334173115874648064?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Graph Generative Adversarial Networks for Sparse Data Generation in High  Energy Physics

Raghav Kansal, Javier Duarte, Breno Orzari, Thiago Tomei, Maurizio Pierini, Mary Touranakou, Jean-Roch Vlimant, Dimitrios Gunopoulos

- retweets: 31, favorites: 24 (12/03/2020 11:30:39)

- links: [abs](https://arxiv.org/abs/2012.00173) | [pdf](https://arxiv.org/pdf/2012.00173)
- [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [hep-ex](https://arxiv.org/list/hep-ex/recent) | [hep-ph](https://arxiv.org/list/hep-ph/recent) | [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent)

We develop a graph generative adversarial network to generate sparse data sets like those produced at the CERN Large Hadron Collider (LHC). We demonstrate this approach by training on and generating sparse representations of MNIST handwritten digit images and jets of particles in proton-proton collisions like those at the LHC. We find the model successfully generates sparse MNIST digits and particle jet data. We quantify agreement between real and generated data with a graph-based Fr\'echet Inception distance, and the particle and jet feature-level 1-Wasserstein distance for the MNIST and jet datasets respectively.




# 12. Pre-Trained Image Processing Transformer

Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, Wen Gao

- retweets: 20, favorites: 33 (12/03/2020 11:30:39)

- links: [abs](https://arxiv.org/abs/2012.00364) | [pdf](https://arxiv.org/pdf/2012.00364)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

As the computing power of modern hardware is increasing strongly, pre-trained deep learning models (\eg, BERT, GPT-3) learned on large-scale datasets have shown their effectiveness over conventional methods. The big progress is mainly contributed to the representation ability of transformer and its variant architectures. In this paper, we study the low-level computer vision task (\eg, denoising, super-resolution and deraining) and develop a new pre-trained model, namely, image processing transformer (IPT). To maximally excavate the capability of transformer, we present to utilize the well-known ImageNet benchmark for generating a large amount of corrupted image pairs. The IPT model is trained on these images with multi-heads and multi-tails. In addition, the contrastive learning is introduced for well adapting to different image processing tasks. The pre-trained model can therefore efficiently employed on desired task after fine-tuning. With only one pre-trained model, IPT outperforms the current state-of-the-art methods on various low-level benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pre-Trained Image Processing Transformer<a href="https://t.co/pS7H9IqpPK">https://t.co/pS7H9IqpPK</a> <a href="https://t.co/bR1dKCELDM">pic.twitter.com/bR1dKCELDM</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1334010043646132224?ref_src=twsrc%5Etfw">December 2, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Double machine learning for (weighted) dynamic treatment effects

Hugo Bodory, Martin Huber, Lukáš Lafférs

- retweets: 36, favorites: 14 (12/03/2020 11:30:39)

- links: [abs](https://arxiv.org/abs/2012.00370) | [pdf](https://arxiv.org/pdf/2012.00370)
- [econ.EM](https://arxiv.org/list/econ.EM/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We consider evaluating the causal effects of dynamic treatments, i.e. of multiple treatment sequences in various periods, based on double machine learning to control for observed, time-varying covariates in a data-driven way under a selection-on-observables assumption. To this end, we make use of so-called Neyman-orthogonal score functions, which imply the robustness of treatment effect estimation to moderate (local) misspecifications of the dynamic outcome and treatment models. This robustness property permits approximating outcome and treatment models by double machine learning even under high dimensional covariates and is combined with data splitting to prevent overfitting. In addition to effect estimation for the total population, we consider weighted estimation that permits assessing dynamic treatment effects in specific subgroups, e.g. among those treated in the first treatment period. We demonstrate that the estimators are asymptotically normal and $\sqrt{n}$-consistent under specific regularity conditions and investigate their finite sample properties in a simulation study. Finally, we apply the methods to the Job Corps study in order to assess different sequences of training programs under a large set of covariates.



