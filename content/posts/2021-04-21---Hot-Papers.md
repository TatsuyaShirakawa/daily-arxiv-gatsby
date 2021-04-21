---
title: Hot Papers 2021-04-21
date: 2021-04-22T07:52:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-21"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-21"
socialImage: "/media/flying-marine.jpg"

---

# 1. VideoGPT: Video Generation using VQ-VAE and Transformers

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas

- retweets: 5311, favorites: 391 (04/22/2021 07:52:36)

- links: [abs](https://arxiv.org/abs/2104.10157) | [pdf](https://arxiv.org/pdf/2104.10157)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present VideoGPT: a conceptually simple architecture for scaling likelihood based generative modeling to natural videos. VideoGPT uses VQ-VAE that learns downsampled discrete latent representations of a raw video by employing 3D convolutions and axial self-attention. A simple GPT-like architecture is then used to autoregressively model the discrete latents using spatio-temporal position encodings. Despite the simplicity in formulation and ease of training, our architecture is able to generate samples competitive with state-of-the-art GAN models for video generation on the BAIR Robot dataset, and generate high fidelity natural images from UCF-101 and Tumbler GIF Dataset (TGIF). We hope our proposed architecture serves as a reproducible reference for a minimalistic implementation of transformer based video generation models. Samples and code are available at https://wilson1yan.github.io/videogpt/index.html

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VideoGPT: Video Generation using VQ-VAE and Transformers<br>pdf: <a href="https://t.co/IH1Vsiotcw">https://t.co/IH1Vsiotcw</a><br>abs: <a href="https://t.co/SrGT6UNxF9">https://t.co/SrGT6UNxF9</a><br>github: <a href="https://t.co/74m6e1MQAY">https://t.co/74m6e1MQAY</a><br>project page: <a href="https://t.co/Z2ozXlPFPJ">https://t.co/Z2ozXlPFPJ</a> <a href="https://t.co/SXCYKLOeUL">pic.twitter.com/SXCYKLOeUL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384670341637738496?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VideoGPT: Video Generation using VQ-VAE and Transformers by <a href="https://twitter.com/UCBerkeley?ref_src=twsrc%5Etfw">@UCBerkeley</a> on <a href="https://twitter.com/GradioML?ref_src=twsrc%5Etfw">@GradioML</a> in <a href="https://twitter.com/PyTorch?ref_src=twsrc%5Etfw">@PyTorch</a> <br>paper: <a href="https://t.co/SrGT6UNxF9">https://t.co/SrGT6UNxF9</a><br>project page: <a href="https://t.co/Z2ozXlPFPJ">https://t.co/Z2ozXlPFPJ</a><br>github: <a href="https://t.co/74m6e1MQAY">https://t.co/74m6e1MQAY</a><br>gradio demo:  <a href="https://t.co/xDkCStYTdf">https://t.co/xDkCStYTdf</a> <a href="https://t.co/Umv48wZbHO">pic.twitter.com/Umv48wZbHO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384902425212067840?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Manipulating SGD with Data Ordering Attacks

Ilia Shumailov, Zakhar Shumaylov, Dmitry Kazhdan, Yiren Zhao, Nicolas Papernot, Murat A. Erdogdu, Ross Anderson

- retweets: 4101, favorites: 299 (04/22/2021 07:52:37)

- links: [abs](https://arxiv.org/abs/2104.09667) | [pdf](https://arxiv.org/pdf/2104.09667)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Machine learning is vulnerable to a wide variety of different attacks. It is now well understood that by changing the underlying data distribution, an adversary can poison the model trained with it or introduce backdoors. In this paper we present a novel class of training-time attacks that require no changes to the underlying model dataset or architecture, but instead only change the order in which data are supplied to the model. In particular, an attacker can disrupt the integrity and availability of a model by simply reordering training batches, with no knowledge about either the model or the dataset. Indeed, the attacks presented here are not specific to the model or dataset, but rather target the stochastic nature of modern learning procedures. We extensively evaluate our attacks to find that the adversary can disrupt model training and even introduce backdoors.   For integrity we find that the attacker can either stop the model from learning, or poison it to learn behaviours specified by the attacker. For availability we find that a single adversarially-ordered epoch can be enough to slow down model learning, or even to reset all of the learning progress. Such attacks have a long-term impact in that they decrease model performance hundreds of epochs after the attack took place. Reordering is a very powerful adversarial paradigm in that it removes the assumption that an adversary must inject adversarial data points or perturbations to perform training-time attacks. It reminds us that stochastic gradient descent relies on the assumption that data are sampled at random. If this randomness is compromised, then all bets are off.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is poisoning ML possible without inserting poison in the model&#39;s training set? Yes. <a href="https://twitter.com/iliaishacked?ref_src=twsrc%5Etfw">@iliaishacked</a> et al. just introduces &quot;data ordering attacks&quot; which are able to target both the integrity and availability of ML simply by *reordering* points during SGD<a href="https://t.co/4rErkWugiP">https://t.co/4rErkWugiP</a> <a href="https://t.co/xx3tNoS64q">pic.twitter.com/xx3tNoS64q</a></p>&mdash; Nicolas Papernot (@NicolasPapernot) <a href="https://twitter.com/NicolasPapernot/status/1384700774987538433?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. RoFormer: Enhanced Transformer with Rotary Position Embedding

Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu

- retweets: 1498, favorites: 270 (04/22/2021 07:52:37)

- links: [abs](https://arxiv.org/abs/2104.09864) | [pdf](https://arxiv.org/pdf/2104.09864)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Position encoding in transformer architecture provides supervision for dependency modeling between elements at different positions in the sequence. We investigate various methods to encode positional information in transformer-based language models and propose a novel implementation named Rotary Position Embedding(RoPE). The proposed RoPE encodes absolute positional information with rotation matrix and naturally incorporates explicit relative position dependency in self-attention formulation. Notably, RoPE comes with valuable properties such as flexibility of being expand to any sequence lengths, decaying inter-token dependency with increasing relative distances, and capability of equipping the linear self-attention with relative position encoding. As a result, the enhanced transformer with rotary position embedding, or RoFormer, achieves superior performance in tasks with long texts. We release the theoretical analysis along with some preliminary experiment results on Chinese data. The undergoing experiment for English benchmark will soon be updated.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RoFormer: Enhanced Transformer with Rotary Position Embedding<br>pdf: <a href="https://t.co/2wOHbPx6Ss">https://t.co/2wOHbPx6Ss</a><br>abs: <a href="https://t.co/LJXnpgFpIK">https://t.co/LJXnpgFpIK</a> <a href="https://t.co/HHCrQ541EV">pic.twitter.com/HHCrQ541EV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384677442787360770?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MBRL-Lib: A Modular Library for Model-based Reinforcement Learning

Luis Pineda, Brandon Amos, Amy Zhang, Nathan O. Lambert, Roberto Calandra

- retweets: 416, favorites: 110 (04/22/2021 07:52:37)

- links: [abs](https://arxiv.org/abs/2104.10159) | [pdf](https://arxiv.org/pdf/2104.10159)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

Model-based reinforcement learning is a compelling framework for data-efficient learning of agents that interact with the world. This family of algorithms has many subcomponents that need to be carefully selected and tuned. As a result the entry-bar for researchers to approach the field and to deploy it in real-world tasks can be daunting. In this paper, we present MBRL-Lib -- a machine learning library for model-based reinforcement learning in continuous state-action spaces based on PyTorch. MBRL-Lib is designed as a platform for both researchers, to easily develop, debug and compare new algorithms, and non-expert user, to lower the entry-bar of deploying state-of-the-art algorithms. MBRL-Lib is open-source at https://github.com/facebookresearch/mbrl-lib.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Today, we are happy to release MBRL-lib -- the first PyTorch library dedicated to model-based reinforcement learning. <br><br>With <a href="https://twitter.com/luisenp?ref_src=twsrc%5Etfw">@luisenp</a>, <a href="https://twitter.com/brandondamos?ref_src=twsrc%5Etfw">@brandondamos</a>, <a href="https://twitter.com/yayitsamyzhang?ref_src=twsrc%5Etfw">@yayitsamyzhang</a>, <a href="https://twitter.com/natolambert?ref_src=twsrc%5Etfw">@natolambert</a><br><br>Code: <a href="https://t.co/OtTECH6cGX">https://t.co/OtTECH6cGX</a><br>Paper: <a href="https://t.co/5Hr0Txa8SF">https://t.co/5Hr0Txa8SF</a> <a href="https://t.co/kyByK1erQN">https://t.co/kyByK1erQN</a></p>&mdash; Roberto Calandra (@RCalandra) <a href="https://twitter.com/RCalandra/status/1384921060919947271?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. GENESIS-V2: Inferring Unordered Object Representations without Iterative  Refinement

Martin Engelcke, Oiwi Parker Jones, Ingmar Posner

- retweets: 111, favorites: 53 (04/22/2021 07:52:37)

- links: [abs](https://arxiv.org/abs/2104.09958) | [pdf](https://arxiv.org/pdf/2104.09958)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Advances in object-centric generative models (OCGMs) have culminated in the development of a broad range of methods for unsupervised object segmentation and interpretable object-centric scene generation. These methods, however, are limited to simulated and real-world datasets with limited visual complexity. Moreover, object representations are often inferred using RNNs which do not scale well to large images or iterative refinement which avoids imposing an unnatural ordering on objects in an image but requires the a priori initialisation of a fixed number of object representations. In contrast to established paradigms, this work proposes an embedding-based approach in which embeddings of pixels are clustered in a differentiable fashion using a stochastic, non-parametric stick-breaking process. Similar to iterative refinement, this clustering procedure also leads to randomly ordered object representations, but without the need of initialising a fixed number of clusters a priori. This is used to develop a new model, GENESIS-V2, which can infer a variable number of object representations without using RNNs or iterative refinement. We show that GENESIS-V2 outperforms previous methods for unsupervised image segmentation and object-centric scene generation on established synthetic datasets as well as more complex real-world datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement&quot; <a href="https://twitter.com/hashtag/tweeprint?src=hash&amp;ref_src=twsrc%5Etfw">#tweeprint</a><br><br>TL;DR: Unsupervised object segmentation and object-centric image generation facilitated by a new stochastic clustering method<br><br>Link: <a href="https://t.co/mK3cB7Xd0m">https://t.co/mK3cB7Xd0m</a><br><br>1/4 <a href="https://t.co/idksN7DNmm">pic.twitter.com/idksN7DNmm</a></p>&mdash; Martin Engelcke (@martinengelcke) <a href="https://twitter.com/martinengelcke/status/1384791760329756672?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Style-Aware Normalized Loss for Improving Arbitrary Style Transfer

Jiaxin Cheng, Ayush Jaiswal, Yue Wu, Pradeep Natarajan, Prem Natarajan

- retweets: 90, favorites: 68 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.10064) | [pdf](https://arxiv.org/pdf/2104.10064)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural Style Transfer (NST) has quickly evolved from single-style to infinite-style models, also known as Arbitrary Style Transfer (AST). Although appealing results have been widely reported in literature, our empirical studies on four well-known AST approaches (GoogleMagenta, AdaIN, LinearTransfer, and SANet) show that more than 50% of the time, AST stylized images are not acceptable to human users, typically due to under- or over-stylization. We systematically study the cause of this imbalanced style transferability (IST) and propose a simple yet effective solution to mitigate this issue. Our studies show that the IST issue is related to the conventional AST style loss, and reveal that the root cause is the equal weightage of training samples irrespective of the properties of their corresponding style images, which biases the model towards certain styles. Through investigation of the theoretical bounds of the AST style loss, we propose a new loss that largely overcomes IST. Theoretical analysis and experimental results validate the effectiveness of our loss, with over 80% relative improvement in style deception rate and 98% relatively higher preference in human evaluation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Style-Aware Normalized Loss for Improving Arbitrary Style Transfer<br>pdf: <a href="https://t.co/N1LNeo4LXY">https://t.co/N1LNeo4LXY</a><br>abs: <a href="https://t.co/sKELSsJOa1">https://t.co/sKELSsJOa1</a> <a href="https://t.co/W9EpXafmhs">pic.twitter.com/W9EpXafmhs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384696466049015810?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Frustratingly Easy Edit-based Linguistic Steganography with a Masked  Language Model

Honai Ueoka, Yugo Murawaki, Sadao Kurohashi

- retweets: 101, favorites: 41 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.09833) | [pdf](https://arxiv.org/pdf/2104.09833)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

With advances in neural language models, the focus of linguistic steganography has shifted from edit-based approaches to generation-based ones. While the latter's payload capacity is impressive, generating genuine-looking texts remains challenging. In this paper, we revisit edit-based linguistic steganography, with the idea that a masked language model offers an off-the-shelf solution. The proposed method eliminates painstaking rule construction and has a high payload capacity for an edit-based model. It is also shown to be more secure against automatic detection than a generation-based method while offering better control of the security/payload capacity trade-off.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">NAACL2021 Shortに採択された論文のプレプリントを公開しました。秘密の情報をテキストに埋め込むステガノグラフィの新しい手法を提案しています。<br>Ueoka, Murawaki and Kurohashi.<br>Frustratingly Easy Edit-based Linguistic Steganography with a Masked Language Model <a href="https://t.co/QUlxP0usGg">https://t.co/QUlxP0usGg</a></p>&mdash; ほない (@_honai) <a href="https://twitter.com/_honai/status/1384833985268703232?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. SelfReg: Self-supervised Contrastive Regularization for Domain  Generalization

Daehee Kim, Seunghyun Park, Jinkyu Kim, Jaekoo Lee

- retweets: 49, favorites: 57 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.09841) | [pdf](https://arxiv.org/pdf/2104.09841)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

In general, an experimental environment for deep learning assumes that the training and the test dataset are sampled from the same distribution. However, in real-world situations, a difference in the distribution between two datasets, domain shift, may occur, which becomes a major factor impeding the generalization performance of the model. The research field to solve this problem is called domain generalization, and it alleviates the domain shift problem by extracting domain-invariant features explicitly or implicitly. In recent studies, contrastive learning-based domain generalization approaches have been proposed and achieved high performance. These approaches require sampling of the negative data pair. However, the performance of contrastive learning fundamentally depends on quality and quantity of negative data pairs. To address this issue, we propose a new regularization method for domain generalization based on contrastive learning, self-supervised contrastive regularization (SelfReg). The proposed approach use only positive data pairs, thus it resolves various problems caused by negative pair sampling. Moreover, we propose a class-specific domain perturbation layer (CDPL), which makes it possible to effectively apply mixup augmentation even when only positive data pairs are used. The experimental results show that the techniques incorporated by SelfReg contributed to the performance in a compatible manner. In the recent benchmark, DomainBed, the proposed method shows comparable performance to the conventional state-of-the-art alternatives. Codes are available at https://github.com/dnap512/SelfReg.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SelfReg: Self-supervised Contrastive Regularization for Domain Generalization<br>pdf: <a href="https://t.co/D8FYjiiCTC">https://t.co/D8FYjiiCTC</a><br>abs: <a href="https://t.co/J0UvJ1iQbK">https://t.co/J0UvJ1iQbK</a><br>github: <a href="https://t.co/mQdl28s3ou">https://t.co/mQdl28s3ou</a> <a href="https://t.co/Fq9iCbERvo">pic.twitter.com/Fq9iCbERvo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384679438584066049?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Lighting, Reflectance and Geometry Estimation from 360$^{\circ}$  Panoramic Stereo

Junxuan Li, Hongdong Li, Yasuyuki Matsushita

- retweets: 49, favorites: 34 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.09886) | [pdf](https://arxiv.org/pdf/2104.09886)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a method for estimating high-definition spatially-varying lighting, reflectance, and geometry of a scene from 360$^{\circ}$ stereo images. Our model takes advantage of the 360$^{\circ}$ input to observe the entire scene with geometric detail, then jointly estimates the scene's properties with physical constraints. We first reconstruct a near-field environment light for predicting the lighting at any 3D location within the scene. Then we present a deep learning model that leverages the stereo information to infer the reflectance and surface normal. Lastly, we incorporate the physical constraints between lighting and geometry to refine the reflectance of the scene. Both quantitative and qualitative experiments show that our method, benefiting from the 360$^{\circ}$ observation of the scene, outperforms prior state-of-the-art methods and enables more augmented reality applications such as mirror-objects insertion.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Lighting, Reflectance and Geometry Estimation from 360° Panoramic Stereo<br>pdf: <a href="https://t.co/qs1VFsqn1g">https://t.co/qs1VFsqn1g</a><br>abs: <a href="https://t.co/xKMNYyheWG">https://t.co/xKMNYyheWG</a><br>github: <a href="https://t.co/e5wQeMa62c">https://t.co/e5wQeMa62c</a> <a href="https://t.co/UfkOj3nuKw">pic.twitter.com/UfkOj3nuKw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384697884650639367?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Large Scale Interactive Motion Forecasting for Autonomous Driving : The  Waymo Open Motion Dataset

Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles Qi, Yin Zhou, Zoey Yang, Aurelien Chouard, Pei Sun, Jiquan Ngiam, Vijay Vasudevan, Alexander McCauley, Jonathon Shlens, Dragomir Anguelov

- retweets: 30, favorites: 28 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.10133) | [pdf](https://arxiv.org/pdf/2104.10133)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

As autonomous driving systems mature, motion forecasting has received increasing attention as a critical requirement for planning. Of particular importance are interactive situations such as merges, unprotected turns, etc., where predicting individual object motion is not sufficient. Joint predictions of multiple objects are required for effective route planning. There has been a critical need for high-quality motion data that is rich in both interactions and annotation to develop motion planning models. In this work, we introduce the most diverse interactive motion dataset to our knowledge, and provide specific labels for interacting objects suitable for developing joint prediction models. With over 100,000 scenes, each 20 seconds long at 10 Hz, our new dataset contains more than 570 hours of unique data over 1750 km of roadways. It was collected by mining for interesting interactions between vehicles, pedestrians, and cyclists across six cities within the United States. We use a high-accuracy 3D auto-labeling system to generate high quality 3D bounding boxes for each road agent, and provide corresponding high definition 3D maps for each scene. Furthermore, we introduce a new set of metrics that provides a comprehensive evaluation of both single agent and joint agent interaction motion forecasting models. Finally, we provide strong baseline models for individual-agent prediction and joint-prediction. We hope that this new large-scale interactive motion dataset will provide new opportunities for advancing motion forecasting models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large Scale Interactive Motion Forecasting for Autonomous Driving : The Waymo Open Motion Dataset<br>pdf: <a href="https://t.co/qFmrn6h28n">https://t.co/qFmrn6h28n</a><br>abs: <a href="https://t.co/mnYKI4CXHj">https://t.co/mnYKI4CXHj</a> <a href="https://t.co/SLHRr6tIAb">pic.twitter.com/SLHRr6tIAb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384703950516011012?ref_src=twsrc%5Etfw">April 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Asymmetric compressive learning guarantees with applications to  quantized sketches

Vincent Schellekens, Laurent Jacques

- retweets: 42, favorites: 11 (04/22/2021 07:52:38)

- links: [abs](https://arxiv.org/abs/2104.10061) | [pdf](https://arxiv.org/pdf/2104.10061)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The compressive learning framework reduces the computational cost of training on large-scale datasets. In a sketching phase, the data is first compressed to a lightweight sketch vector, obtained by mapping the data samples through a well-chosen feature map, and averaging those contributions. In a learning phase, the desired model parameters are then extracted from this sketch by solving an optimization problem, which also involves a feature map. When the feature map is identical during the sketching and learning phases, formal statistical guarantees (excess risk bounds) have been proven.   However, the desirable properties of the feature map are different during sketching and learning (e.g. quantized outputs, and differentiability, respectively). We thus study the relaxation where this map is allowed to be different for each phase. First, we prove that the existing guarantees carry over to this asymmetric scheme, up to a controlled error term, provided some Limited Projected Distortion (LPD) property holds. We then instantiate this framework to the setting of quantized sketches, by proving that the LPD indeed holds for binary sketch contributions. Finally, we further validate the approach with numerical simulations, including a large-scale application in audio event classification.



