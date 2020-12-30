---
title: Hot Papers 2020-12-29
date: 2020-12-30T10:11:08.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. Towards Fully Automated Manga Translation

Ryota Hinami, Shonosuke Ishiwatari, Kazuhiko Yasuda, Yusuke Matsui

- retweets: 3574, favorites: 306 (12/30/2020 10:11:08)

- links: [abs](https://arxiv.org/abs/2012.14271) | [pdf](https://arxiv.org/pdf/2012.14271)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We tackle the problem of machine translation of manga, Japanese comics. Manga translation involves two important problems in machine translation: context-aware and multimodal translation. Since text and images are mixed up in an unstructured fashion in Manga, obtaining context from the image is essential for manga translation. However, it is still an open problem how to extract context from image and integrate into MT models. In addition, corpus and benchmarks to train and evaluate such model is currently unavailable. In this paper, we make the following four contributions that establishes the foundation of manga translation research. First, we propose multimodal context-aware translation framework. We are the first to incorporate context information obtained from manga image. It enables us to translate texts in speech bubbles that cannot be translated without using context information (e.g., texts in other speech bubbles, gender of speakers, etc.). Second, for training the model, we propose the approach to automatic corpus construction from pairs of original manga and their translations, by which large parallel corpus can be constructed without any manual labeling. Third, we created a new benchmark to evaluate manga translation. Finally, on top of our proposed methods, we devised a first comprehensive system for fully automated manga translation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Fully Automated Manga Translation<br>pdf: <a href="https://t.co/AmFu0hfuLY">https://t.co/AmFu0hfuLY</a><br>abs: <a href="https://t.co/c3IvLwlt2j">https://t.co/c3IvLwlt2j</a> <a href="https://t.co/DXuRrASgZ5">pic.twitter.com/DXuRrASgZ5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343768439698190337?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. TransPose: Towards Explainable Human Pose Estimation by Transformer

Sen Yang, Zhibin Quan, Mu Nie, Wankou Yang

- retweets: 1225, favorites: 164 (12/30/2020 10:11:09)

- links: [abs](https://arxiv.org/abs/2012.14214) | [pdf](https://arxiv.org/pdf/2012.14214)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep Convolutional Neural Networks (CNNs) have made remarkable progress on human pose estimation task. However, there is no explicit understanding of how the locations of body keypoints are predicted by CNN, and it is also unknown what spatial dependency relationships between structural variables are learned in the model. To explore these questions, we construct an explainable model named TransPose based on Transformer architecture and low-level convolutional blocks. Given an image, the attention layers built in Transformer can capture long-range spatial relationships between keypoints and explain what dependencies the predicted keypoints locations highly rely on. We analyze the rationality of using attention as the explanation to reveal the spatial dependencies in this task. The revealed dependencies are image-specific and variable across different keypoint types, layer depths, or trained models. The experiments show that TransPose can accurately predict the positions of keypoints. It achieves state-of-the-art performance on COCO dataset, while being more interpretable, lightweight, and efficient than mainstream fully convolutional architectures.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TransPose: Towards Explainable Human Pose Estimation by Transformer<br>pdf: <a href="https://t.co/nOp5LlI2en">https://t.co/nOp5LlI2en</a><br>abs: <a href="https://t.co/GxtVcC5qIN">https://t.co/GxtVcC5qIN</a> <a href="https://t.co/MZsnqM1IAq">pic.twitter.com/MZsnqM1IAq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343748998176366593?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Improving the Generalization of End-to-End Driving through Procedural  Generation

Quanyi Li, Zhenghao Peng, Qihang Zhang, Cong Qiu, Chunxiao Liu, Bolei Zhou

- retweets: 706, favorites: 114 (12/30/2020 10:11:09)

- links: [abs](https://arxiv.org/abs/2012.13681) | [pdf](https://arxiv.org/pdf/2012.13681)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently there is a growing interest in the end-to-end training of autonomous driving where the entire driving pipeline from perception to control is modeled as a neural network and jointly optimized. The end-to-end driving is usually first developed and validated in simulators. However, most of the existing driving simulators only contain a fixed set of maps and a limited number of configurations. As a result the deep models are prone to overfitting training scenarios. Furthermore it is difficult to assess how well the trained models generalize to unseen scenarios. To better evaluate and improve the generalization of end-to-end driving, we introduce an open-ended and highly configurable driving simulator called PGDrive. PGDrive first defines multiple basic road blocks such as ramp, fork, and roundabout with configurable settings. Then a range of diverse maps can be assembled from those blocks with procedural generation, which are further turned into interactive environments. The experiments show that the driving agent trained by reinforcement learning on a small fixed set of maps generalizes poorly to unseen maps. We further validate that training with the increasing number of procedurally generated maps significantly improves the generalization of the agent across scenarios of different traffic densities and map structures. Code is available at: https://decisionforce.github.io/pgdrive

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improving the Generalization of End-to-End Driving through Procedural Generation<br>pdf: <a href="https://t.co/ismMqRj7iS">https://t.co/ismMqRj7iS</a><br>abs: <a href="https://t.co/1YNiEV6RT7">https://t.co/1YNiEV6RT7</a><br>project page: <a href="https://t.co/68XDgqKFsi">https://t.co/68XDgqKFsi</a><br>github: <a href="https://t.co/y7DY70KCc0">https://t.co/y7DY70KCc0</a> <a href="https://t.co/8txE1zstEB">pic.twitter.com/8txE1zstEB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343758590549614598?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Neural Network Training With Homomorphic Encryption

Kentaro Mihara, Ryohei Yamaguchi, Miguel Mitsuishi, Yusuke Maruyama

- retweets: 426, favorites: 24 (12/30/2020 10:11:09)

- links: [abs](https://arxiv.org/abs/2012.13552) | [pdf](https://arxiv.org/pdf/2012.13552)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

We introduce a novel method and implementation architecture to train neural networks which preserves the confidentiality of both the model and the data. Our method relies on homomorphic capability of lattice based encryption scheme. Our procedure is optimized for operations on packed ciphertexts in order to achieve efficient updates of the model parameters. Our method achieves a significant reduction of computations due to our way to perform multiplications and rotations on packed ciphertexts from a feedforward network to a back-propagation network. To verify the accuracy of the training model as well as the implementation feasibility, we tested our method on the Iris data set by using the CKKS scheme with Microsoft SEAL as a back end. Although our test implementation is for simple neural network training, we believe our basic implementation block can help the further applications for more complex neural network based use cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Network Training With Homomorphic Encryption.<a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Cloud?src=hash&amp;ref_src=twsrc%5Etfw">#Cloud</a> <a href="https://twitter.com/hashtag/DevCommunity?src=hash&amp;ref_src=twsrc%5Etfw">#DevCommunity</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/javascript?src=hash&amp;ref_src=twsrc%5Etfw">#javascript</a> <a href="https://twitter.com/hashtag/womenwhocode?src=hash&amp;ref_src=twsrc%5Etfw">#womenwhocode</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/CodeNewbie?src=hash&amp;ref_src=twsrc%5Etfw">#CodeNewbie</a> <a href="https://twitter.com/hashtag/100DaysOfMLCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfMLCode</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a><a href="https://t.co/RLU4jEjIEQ">https://t.co/RLU4jEjIEQ</a> <a href="https://t.co/26PKeNMS52">pic.twitter.com/26PKeNMS52</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1344045151791022081?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Evolution Is All You Need: Phylogenetic Augmentation for Contrastive  Learning

Amy X. Lu, Alex X. Lu, Alan Moses

- retweets: 182, favorites: 60 (12/30/2020 10:11:09)

- links: [abs](https://arxiv.org/abs/2012.13475) | [pdf](https://arxiv.org/pdf/2012.13475)
- [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Self-supervised representation learning of biological sequence embeddings alleviates computational resource constraints on downstream tasks while circumventing expensive experimental label acquisition. However, existing methods mostly borrow directly from large language models designed for NLP, rather than with bioinformatics philosophies in mind. Recently, contrastive mutual information maximization methods have achieved state-of-the-art representations for ImageNet. In this perspective piece, we discuss how viewing evolution as natural sequence augmentation and maximizing information across phylogenetic "noisy channels" is a biologically and theoretically desirable objective for pretraining encoders. We first provide a review of current contrastive learning literature, then provide an illustrative example where we show that contrastive learning using evolutionary augmentation can be used as a representation learning objective which maximizes the mutual information between biological sequences and their conserved function, and finally outline rationale for this approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Evolution Is All You Need: Phylogenetic Augmentation for Contrastive Learning<br>pdf: <a href="https://t.co/CmgYtTd8bJ">https://t.co/CmgYtTd8bJ</a><br>abs: <a href="https://t.co/KSlv6zbVX6">https://t.co/KSlv6zbVX6</a> <a href="https://t.co/GsAkiiPIzi">pic.twitter.com/GsAkiiPIzi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343754907522920450?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. On Generating Extended Summaries of Long Documents

Sajad Sotudeh, Arman Cohan, Nazli Goharian

- retweets: 156, favorites: 63 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.14136) | [pdf](https://arxiv.org/pdf/2012.14136)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Prior work in document summarization has mainly focused on generating short summaries of a document. While this type of summary helps get a high-level view of a given document, it is desirable in some cases to know more detailed information about its salient points that can't fit in a short summary. This is typically the case for longer documents such as a research paper, legal document, or a book. In this paper, we present a new method for generating extended summaries of long papers. Our method exploits hierarchical structure of the documents and incorporates it into an extractive summarization model through a multi-task learning approach. We then present our results on three long summarization datasets, arXiv-Long, PubMed-Long, and Longsumm. Our method outperforms or matches the performance of strong baselines. Furthermore, we perform a comprehensive analysis over the generated results, shedding insights on future research for long-form summary generation task. Our analysis shows that our multi-tasking approach can adjust extraction probability distribution to the favor of summary-worthy sentences across diverse sections. Our datasets, and codes are publicly available at https://github.com/Georgetown-IR-Lab/ExtendedSumm

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A new method for generating extended summaries of long papers. Exploiting hierarchical structure of the docs and incorporates it into an extractive summarization model through a multi-task learning approach<br><br>Paper <a href="https://t.co/OUBVbD2XYy">https://t.co/OUBVbD2XYy</a><br><br>GitHub <a href="https://t.co/szh0Zg7FI4">https://t.co/szh0Zg7FI4</a> <a href="https://t.co/v5HwuOz7bL">pic.twitter.com/v5HwuOz7bL</a></p>&mdash; Philip Vollet (@philipvollet) <a href="https://twitter.com/philipvollet/status/1343806074030321667?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Analysis of Short Dwell Time in Relation to User Interest in a News  Application

Ryosuke Homma, Yoshifumi Seki, Mitsuo Yoshida, Kyoji Umemura

- retweets: 121, favorites: 41 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.13992) | [pdf](https://arxiv.org/pdf/2012.13992)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.DL](https://arxiv.org/list/cs.DL/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Dwell time has been widely used in various fields to evaluate content quality and user engagement. Although many studies shown that content with long dwell time is good quality, contents with short dwell time have not been discussed in detail. We hypothesize that content with short dwell time is not always low quality and does not always have low user engagement, but is instead related to user interest. The purpose of this study is to clarify the meanings of short dwell time browsing in mobile news application. First, we analyze the relation of short dwell time to user interest using large scale user behavior logs from a mobile news application. This analysis was conducted on a vector space based on users click histories and then users and articles were mapped in the same space. The users with short dwell time are concentrated on a specific position in this space; thus, the length of dwell time is related to their interest. Moreover, we also analyze the characteristics of short dwell time browsing by excluding these browses from their click histories. Surprisingly, excluding short dwell time click history, it was found that short dwell time click history included some aspect of user interest in 30.87% of instances where the cluster of users changed. These findings demonstrate that short dwell time does not always indicate a low level of user engagement, but also level of user interest.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">ユーザのクリック履歴をもとにした特徴空間上にニュース記事と滞在時間をマッピングし、滞在時間の意味を考察。短い滞在時間が必ずしもエンゲージメントの低さを示すわけではないことを示唆<br>Analysis of Short Dwell Time in Relation to User Interest in a News Application<a href="https://t.co/Gdfs0GeXYy">https://t.co/Gdfs0GeXYy</a></p>&mdash; Mitsuo Yoshida; AI Bot (PR) (@ceekz) <a href="https://twitter.com/ceekz/status/1343774253196345344?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A Google Earth Engine-enabled Python approach to improve identification  of anthropogenic palaeo-landscape features

Filippo Brandolini, Guillem Domingo Ribas, Andrea Zerboni, Sam Turner

- retweets: 87, favorites: 18 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.14180) | [pdf](https://arxiv.org/pdf/2012.14180)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

The necessity of sustainable development for landscapes has emerged as an important theme in recent decades. Current methods take a holistic approach to landscape heritage and promote an interdisciplinary dialogue to facilitate complementary landscape management strategies. With the socio-economic values of the natural and cultural landscape heritage increasingly recognised worldwide, remote sensing tools are being used more and more to facilitate the recording and management of landscape heritage. Satellite remote sensing technologies have enabled significant improvements in landscape research. The advent of the cloud-based platform of Google Earth Engine has allowed the rapid exploration and processing of satellite imagery such as the Landsat and Copernicus Sentinel datasets. In this paper, the use of Sentinel-2 satellite data in the identification of palaeo-riverscape features has been assessed in the Po Plain, selected because it is characterized by human exploitation since the Mid-Holocene. A multi-temporal approach has been adopted to investigate the potential of satellite imagery to detect buried hydrological and anthropogenic features along with Spectral Index and Spectral Decomposition analysis. This research represents one of the first applications of the GEE Python API in landscape studies. The complete FOSS-cloud protocol proposed here consists of a Python code script developed in Google Colab which could be simply adapted and replicated in different areas of the world




# 9. Self-supervised Pre-training with Hard Examples Improves Visual  Representations

Chunyuan Li, Xiujun Li, Lei Zhang, Baolin Peng, Mingyuan Zhou, Jianfeng Gao

- retweets: 64, favorites: 38 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.13493) | [pdf](https://arxiv.org/pdf/2012.13493)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Self-supervised pre-training (SSP) employs random image transformations to generate training data for visual representation learning. In this paper, we first present a modeling framework that unifies existing SSP methods as learning to predict pseudo-labels. Then, we propose new data augmentation methods of generating training examples whose pseudo-labels are harder to predict than those generated via random image transformations. Specifically, we use adversarial training and CutMix to create hard examples (HEXA) to be used as augmented views for MoCo-v2 and DeepCluster-v2, leading to two variants HEXA_{MoCo} and HEXA_{DCluster}, respectively. In our experiments, we pre-train models on ImageNet and evaluate them on multiple public benchmarks. Our evaluation shows that the two new algorithm variants outperform their original counterparts, and achieve new state-of-the-art on a wide range of tasks where limited task supervision is available for fine-tuning. These results verify that hard examples are instrumental in improving the generalization of the pre-trained models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Pre-training with Hard Examples<br>Improves Visual Representations<br>pdf: <a href="https://t.co/cqUSDdjTvg">https://t.co/cqUSDdjTvg</a><br>abs: <a href="https://t.co/fWcK8bysab">https://t.co/fWcK8bysab</a> <a href="https://t.co/3Gqwmp2GGC">pic.twitter.com/3Gqwmp2GGC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343771874308878338?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Universal Sentence Representation Learning with Conditional Masked  Language Model

Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve

- retweets: 42, favorites: 50 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.14388) | [pdf](https://arxiv.org/pdf/2012.14388)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

This paper presents a novel training method, Conditional Masked Language Modeling (CMLM), to effectively learn sentence representations on large scale unlabeled corpora. CMLM integrates sentence representation learning into MLM training by conditioning on the encoded vectors of adjacent sentences. Our English CMLM model achieves state-of-the-art performance on SentEval, even outperforming models learned using (semi-)supervised signals. As a fully unsupervised learning method, CMLM can be conveniently extended to a broad range of languages and domains. We find that a multilingual CMLM model co-trained with bitext retrieval~(BR) and natural language inference~(NLI) tasks outperforms the previous state-of-the-art multilingual models by a large margin. We explore the same language bias of the learned representations, and propose a principle component based approach to remove the language identifying information from the representation while still retaining sentence semantics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Universal Sentence Representation Learning with Conditional Masked Language Model<br>pdf: <a href="https://t.co/COdss9Yhcp">https://t.co/COdss9Yhcp</a><br>abs: <a href="https://t.co/FHld1izR6F">https://t.co/FHld1izR6F</a> <a href="https://t.co/80b9AY4fgn">pic.twitter.com/80b9AY4fgn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343769949987348483?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Logic Tensor Networks

Samy Badreddine, Artur d'Avila Garcez, Luciano Serafini, Michael Spranger

- retweets: 64, favorites: 22 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.13635) | [pdf](https://arxiv.org/pdf/2012.13635)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Artificial Intelligence agents are required to learn from their surroundings and to reason about the knowledge that has been learned in order to make decisions. While state-of-the-art learning from data typically uses sub-symbolic distributed representations, reasoning is normally useful at a higher level of abstraction with the use of a first-order logic language for knowledge representation. As a result, attempts at combining symbolic AI and neural computation into neural-symbolic systems have been on the increase. In this paper, we present Logic Tensor Networks (LTN), a neurosymbolic formalism and computational model that supports learning and reasoning through the introduction of a many-valued, end-to-end differentiable first-order logic called Real Logic as a representation language for deep learning. We show that LTN provides a uniform language for the specification and the computation of several AI tasks such as data clustering, multi-label classification, relational learning, query answering, semi-supervised learning, regression and embedding learning. We implement and illustrate each of the above tasks with a number of simple explanatory examples using TensorFlow 2. Keywords: Neurosymbolic AI, Deep Learning and Reasoning, Many-valued Logic.




# 12. Taxonomy of multimodal self-supervised representation learning

Alex Fedorov, Tristan Sylvain, Margaux Luck, Lei Wu, Thomas P. DeRamus, Alex Kirilin, Dmitry Bleklov, Sergey M. Plis, Vince D. Calhoun

- retweets: 52, favorites: 26 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.13623) | [pdf](https://arxiv.org/pdf/2012.13623)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Sensory input from multiple sources is crucial for robust and coherent human perception. Different sources contribute complementary explanatory factors and get combined based on factors they share. This system motivated the design of powerful unsupervised representation-learning algorithms. In this paper, we unify recent work on multimodal self-supervised learning under a single framework. Observing that most self-supervised methods optimize similarity metrics between a set of model components, we propose a taxonomy of all reasonable ways to organize this process. We empirically show on two versions of multimodal MNIST and a multimodal brain imaging dataset that (1) multimodal contrastive learning has significant benefits over its unimodal counterpart, (2) the specific composition of multiple contrastive objectives is critical to performance on a downstream task, (3) maximization of the similarity between representations has a regularizing effect on a neural network, which sometimes can lead to reduced downstream performance but still can reveal multimodal relations. Consequently, we outperform previous unsupervised encoder-decoder methods based on CCA or variational mixtures MMVAE on various datasets on linear evaluation protocol.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Taxonomy of multimodal self-supervised representation learning<br>pdf: <a href="https://t.co/pn1YIFCAuf">https://t.co/pn1YIFCAuf</a><br>abs: <a href="https://t.co/w7CVAESmS0">https://t.co/w7CVAESmS0</a> <a href="https://t.co/wMWPqwAymN">pic.twitter.com/wMWPqwAymN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343774326806487040?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Latent Compass: Creation by Navigation

Sarah Schwettmann, Hendrik Strobelt, Mauro Martino

- retweets: 42, favorites: 36 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.14283) | [pdf](https://arxiv.org/pdf/2012.14283)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

In Marius von Senden's Space and Sight, a newly sighted blind patient describes the experience of a corner as lemon-like, because corners "prick" sight like lemons prick the tongue. Prickliness, here, is a dimension in the feature space of sensory experience, an effect of the perceived on the perceiver that arises where the two interact. In the account of the newly sighted, an effect familiar from one interaction translates to a novel context. Perception serves as the vehicle for generalization, in that an effect shared across different experiences produces a concrete abstraction grounded in those experiences. Cezanne and the post-impressionists, fluent in the language of experience translation, realized that the way to paint a concrete form that best reflected reality was to paint not what they saw, but what it was like to see. We envision a future of creation using AI where what it is like to see is replicable, transferrable, manipulable - part of the artist's palette that is both grounded in a particular context, and generalizable beyond it.   An active line of research maps human-interpretable features onto directions in GAN latent space. Supervised and self-supervised approaches that search for anticipated directions or use off-the-shelf classifiers to drive image manipulation in embedding space are limited in the variety of features they can uncover. Unsupervised approaches that discover useful new directions show that the space of perceptually meaningful directions is nowhere close to being fully mapped. As this space is broad and full of creative potential, we want tools for direction discovery that capture the richness and generalizability of human perception. Our approach puts creators in the discovery loop during real-time tool use, in order to identify directions that are perceptually meaningful to them, and generate interpretable image translations along those directions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Latent Compass: Creation by Navigation<br>pdf: <a href="https://t.co/biaGPSTVCD">https://t.co/biaGPSTVCD</a><br>abs: <a href="https://t.co/uLUofsnkbB">https://t.co/uLUofsnkbB</a><br>web demo: <a href="https://t.co/Kv0EG685RJ">https://t.co/Kv0EG685RJ</a> <a href="https://t.co/Mo7NgD13lL">pic.twitter.com/Mo7NgD13lL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1343765781339926531?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. A Tutorial on Sparse Gaussian Processes and Variational Inference

Felix Leibfried, Vincent Dutordoir, ST John, Nicolas Durrande

- retweets: 30, favorites: 45 (12/30/2020 10:11:10)

- links: [abs](https://arxiv.org/abs/2012.13962) | [pdf](https://arxiv.org/pdf/2012.13962)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Gaussian processes (GPs) provide a framework for Bayesian inference that can offer principled uncertainty estimates for a large range of problems. For example, if we consider regression problems with Gaussian likelihoods, a GP model can predict both the mean and variance of the posterior in closed form. However, identifying the posterior GP scales cubically with the number of training examples and requires to store all examples in memory. In order to overcome these obstacles, sparse GPs have been proposed that approximate the true posterior GP with pseudo-training examples. Importantly, the number of pseudo-training examples is user-defined and enables control over computational and memory complexity. In the general case, sparse GPs do not enjoy closed-form solutions and one has to resort to approximate inference. In this context, a convenient choice for approximate inference is variational inference (VI), where the problem of Bayesian inference is cast as an optimization problem -- namely, to maximize a lower bound of the log marginal likelihood. This paves the way for a powerful and versatile framework, where pseudo-training examples are treated as optimization arguments of the approximate posterior that are jointly identified together with hyperparameters of the generative model (i.e. prior and likelihood). The framework can naturally handle a wide scope of supervised learning problems, ranging from regression with heteroscedastic and non-Gaussian likelihoods to classification problems with discrete labels, but also multilabel problems. The purpose of this tutorial is to provide access to the basic matter for readers without prior knowledge in both GPs and VI. A proper exposition to the subject enables also access to more recent advances (like importance-weighted VI as well as inderdomain, multioutput and deep GPs) that can serve as an inspiration for new research ideas.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Tutorial on Sparse Gaussian Processes and Variational Inference. (arXiv:2012.13962v1 [cs.LG]) <a href="https://t.co/5OIVxc5Fse">https://t.co/5OIVxc5Fse</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1343750480397299719?ref_src=twsrc%5Etfw">December 29, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



