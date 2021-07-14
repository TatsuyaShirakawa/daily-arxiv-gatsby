---
title: Hot Papers 2021-07-14
date: 2021-07-15T08:02:56.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-14"
socialImage: "/media/flying-marine.jpg"

---

# 1. Codified audio language modeling learns useful representations for music  information retrieval

Rodrigo Castellon, Chris Donahue, Percy Liang

- retweets: 2826, favorites: 217 (07/15/2021 08:02:56)

- links: [abs](https://arxiv.org/abs/2107.05677) | [pdf](https://arxiv.org/pdf/2107.05677)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We demonstrate that language models pre-trained on codified (discretely-encoded) music audio learn representations that are useful for downstream MIR tasks. Specifically, we explore representations from Jukebox (Dhariwal et al. 2020): a music generation system containing a language model trained on codified audio from 1M songs. To determine if Jukebox's representations contain useful information for MIR, we use them as input features to train shallow models on several MIR tasks. Relative to representations from conventional MIR models which are pre-trained on tagging, we find that using representations from Jukebox as input features yields 30% stronger performance on average across four MIR tasks: tagging, genre classification, emotion recognition, and key detection. For key detection, we observe that representations from Jukebox are considerably stronger than those from models pre-trained on tagging, suggesting that pre-training via codified audio language modeling may address blind spots in conventional approaches. We interpret the strength of Jukebox's representations as evidence that modeling audio instead of tags provides richer representations for MIR.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">There are 🔥 music representations lurking in Jukebox, a language model of music audio. A preview of what&#39;s to come for MIR?<br><br>Work w/ (undergrad!) Rodrigo Castellon and <a href="https://twitter.com/percyliang?ref_src=twsrc%5Etfw">@percyliang</a>. Freshly accepted <a href="https://twitter.com/ismir2021?ref_src=twsrc%5Etfw">@ismir2021</a><br><br>📜 <a href="https://t.co/evnT9PULYr">https://t.co/evnT9PULYr</a><br>⭐ <a href="https://t.co/dss93vf1J1">https://t.co/dss93vf1J1</a><br><br>🧵[1/8] <a href="https://t.co/OmsPs31745">pic.twitter.com/OmsPs31745</a></p>&mdash; Chris Donahue (@chrisdonahuey) <a href="https://twitter.com/chrisdonahuey/status/1415105016138846208?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. CMT: Convolutional Neural Networks Meet Vision Transformers

Jianyuan Guo, Kai Han, Han Wu, Chang Xu, Yehui Tang, Chunjing Xu, Yunhe Wang

- retweets: 1886, favorites: 195 (07/15/2021 08:02:57)

- links: [abs](https://arxiv.org/abs/2107.06263) | [pdf](https://arxiv.org/pdf/2107.06263)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Vision transformers have been successfully applied to image recognition tasks due to their ability to capture long-range dependencies within an image. However, there are still gaps in both performance and computational cost between transformers and existing convolutional neural networks (CNNs). In this paper, we aim to address this issue and develop a network that can outperform not only the canonical transformers, but also the high-performance convolutional models. We propose a new transformer based hybrid network by taking advantage of transformers to capture long-range dependencies, and of CNNs to model local features. Furthermore, we scale it to obtain a family of models, called CMTs, obtaining much better accuracy and efficiency than previous convolution and transformer based models. In particular, our CMT-S achieves 83.5% top-1 accuracy on ImageNet, while being 14x and 2x smaller on FLOPs than the existing DeiT and EfficientNet, respectively. The proposed CMT-S also generalizes well on CIFAR10 (99.2%), CIFAR100 (91.7%), Flowers (98.7%), and other challenging vision datasets such as COCO (44.3% mAP), with considerably less computational cost.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CMT: Convolutional Neural Networks Meet Vision Transformers<br>pdf: <a href="https://t.co/RBD7z9kyCn">https://t.co/RBD7z9kyCn</a><br>abs: <a href="https://t.co/EVN7CBJjGb">https://t.co/EVN7CBJjGb</a><br><br>achieves 83.5% top-1 accuracy on ImageNet, while being 14x and 2x smaller on FLOPs than the existing DeiT and EfficientNet, respectively <a href="https://t.co/BMU0smHvrX">pic.twitter.com/BMU0smHvrX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415108414699618305?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Per-Pixel Classification is Not All You Need for Semantic Segmentation

Bowen Cheng, Alexander G. Schwing, Alexander Kirillov

- retweets: 798, favorites: 186 (07/15/2021 08:02:57)

- links: [abs](https://arxiv.org/abs/2107.06278) | [pdf](https://arxiv.org/pdf/2107.06278)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Per-Pixel Classification is Not All You Need for Semantic Segmentation<br>pdf: <a href="https://t.co/lG6ZYV8XBp">https://t.co/lG6ZYV8XBp</a><br>github: <a href="https://t.co/bXqZ6pR3Fb">https://t.co/bXqZ6pR3Fb</a><br>outperforms both current sota semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models <a href="https://t.co/RqNInMJQOm">pic.twitter.com/RqNInMJQOm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415118835846455299?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit  Partial Observability

Dibya Ghosh, Jad Rahme, Aviral Kumar, Amy Zhang, Ryan P. Adams, Sergey Levine

- retweets: 479, favorites: 222 (07/15/2021 08:02:57)

- links: [abs](https://arxiv.org/abs/2107.06277) | [pdf](https://arxiv.org/pdf/2107.06277)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Generalization is a central challenge for the deployment of reinforcement learning (RL) systems in the real world. In this paper, we show that the sequential structure of the RL problem necessitates new approaches to generalization beyond the well-studied techniques used in supervised learning. While supervised learning methods can generalize effectively without explicitly accounting for epistemic uncertainty, we show that, perhaps surprisingly, this is not the case in RL. We show that generalization to unseen test conditions from a limited number of training conditions induces implicit partial observability, effectively turning even fully-observed MDPs into POMDPs. Informed by this observation, we recast the problem of generalization in RL as solving the induced partially observed Markov decision process, which we call the epistemic POMDP. We demonstrate the failure modes of algorithms that do not appropriately handle this partial observability, and suggest a simple ensemble-based technique for approximately solving the partially observed problem. Empirically, we demonstrate that our simple algorithm derived from the epistemic POMDP achieves significant gains in generalization over current methods on the Procgen benchmark suite.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fresh on ArXiv! <a href="https://t.co/KY9tnogl6Z">https://t.co/KY9tnogl6Z</a><br><br>TL;DR: Standard RL algos are sub-optimal for generalization. Why? When generalizing from limited training scenarios,the fully-observed env implicitly becomes partially-observed, necessitating new algos and policies to generalize well (1/n) <a href="https://t.co/BTL768Dwox">pic.twitter.com/BTL768Dwox</a></p>&mdash; Dibya Ghosh (@its_dibya) <a href="https://twitter.com/its_dibya/status/1415334901750976512?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Empirical studies observed that generalization in RL is hard. Why? In a new paper, we provide a partial answer: generalization in RL induces partial observability, even for fully observed MDPs! This makes standard RL methods suboptimal.<a href="https://t.co/u5HaeJPQ29">https://t.co/u5HaeJPQ29</a><br><br>A thread:</p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1415364606445318155?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why Generalization in RL is Difficult: Epistemic POMDPs<br>and Implicit Partial Observability<br>pdf: <a href="https://t.co/72jLkJ1ijQ">https://t.co/72jLkJ1ijQ</a><br>abs: <a href="https://t.co/sG0KNEG0cC">https://t.co/sG0KNEG0cC</a> <a href="https://t.co/KF1n3UaYAi">pic.twitter.com/KF1n3UaYAi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415176982950776835?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Combiner: Full Attention Transformer with Sparse Computation Cost

Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, Bo Dai

- retweets: 448, favorites: 175 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05768) | [pdf](https://arxiv.org/pdf/2107.05768)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Transformers provide a class of expressive architectures that are extremely effective for sequence modeling. However, the key limitation of transformers is their quadratic memory and time complexity $\mathcal{O}(L^2)$ with respect to the sequence length in attention layers, which restricts application in extremely long sequences. Most existing approaches leverage sparsity or low-rank assumptions in the attention matrix to reduce cost, but sacrifice expressiveness. Instead, we propose Combiner, which provides full attention capability in each attention head while maintaining low computation and memory complexity. The key idea is to treat the self-attention mechanism as a conditional expectation over embeddings at each location, and approximate the conditional distribution with a structured factorization. Each location can attend to all other locations, either via direct attention, or through indirect attention to abstractions, which are again conditional expectations of embeddings from corresponding local regions. We show that most sparse attention patterns used in existing sparse transformers are able to inspire the design of such factorization for full attention, resulting in the same sub-quadratic cost ($\mathcal{O}(L\log(L))$ or $\mathcal{O}(L\sqrt{L})$). Combiner is a drop-in replacement for attention layers in existing transformers and can be easily implemented in common frameworks. An experimental evaluation on both autoregressive and bidirectional sequence tasks demonstrates the effectiveness of this approach, yielding state-of-the-art results on several image and text modeling tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Combiner: Full Attention Transformer with Sparse Computation Cost<br><br>Proposes O(L log L) efficient attention Transformer that yields SotA results on several image and text modeling tasks, both autoregressive and MLM.<a href="https://t.co/RD8uY6De6m">https://t.co/RD8uY6De6m</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1415111387446013952?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Combiner: Full Attention Transformer with Sparse Computation Cost<br>pdf: <a href="https://t.co/lZ39kcGlLu">https://t.co/lZ39kcGlLu</a><br><br>achieves sota performance on both autoregressive and bidirectional tasks for image and text modeling, showing benefits in both modeling effectiveness and runtime efficiency <a href="https://t.co/AsHV8hostd">pic.twitter.com/AsHV8hostd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415109649980866562?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. The Piano Inpainting Application

Gaëtan Hadjeres, Léopold Crestel

- retweets: 324, favorites: 78 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05944) | [pdf](https://arxiv.org/pdf/2107.05944)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Autoregressive models are now capable of generating high-quality minute-long expressive MIDI piano performances. Even though this progress suggests new tools to assist music composition, we observe that generative algorithms are still not widely used by artists due to the limited control they offer, prohibitive inference times or the lack of integration within musicians' workflows. In this work, we present the Piano Inpainting Application (PIA), a generative model focused on inpainting piano performances, as we believe that this elementary operation (restoring missing parts of a piano performance) encourages human-machine interaction and opens up new ways to approach music composition. Our approach relies on an encoder-decoder Linear Transformer architecture trained on a novel representation for MIDI piano performances termed Structured MIDI Encoding. By uncovering an interesting synergy between Linear Transformers and our inpainting task, we are able to efficiently inpaint contiguous regions of a piano performance, which makes our model suitable for interactive and responsive A.I.-assisted composition. Finally, we introduce our freely-available Ableton Live PIA plugin, which allows musicians to smoothly generate or modify any MIDI clip using PIA within a widely-used professional Digital Audio Workstation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Piano Inpainting Application 🎹🎶<br>pdf: <a href="https://t.co/DdZ9zXiaMe">https://t.co/DdZ9zXiaMe</a><br>abs: <a href="https://t.co/QPbb6hXWzc">https://t.co/QPbb6hXWzc</a><br>project page: <a href="https://t.co/q5LT9AXT68">https://t.co/q5LT9AXT68</a> <a href="https://t.co/fMOE2AJqKi">pic.twitter.com/fMOE2AJqKi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415135987877961728?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Hidden Convexity of Wasserstein GANs: Interpretable Generative Models  with Closed-Form Solutions

Arda Sahiner, Tolga Ergen, Batu Ozturkler, Burak Bartan, John Pauly, Morteza Mardani, Mert Pilanci

- retweets: 169, favorites: 59 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05680) | [pdf](https://arxiv.org/pdf/2107.05680)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [math.OC](https://arxiv.org/list/math.OC/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Generative Adversarial Networks (GANs) are commonly used for modeling complex distributions of data. Both the generators and discriminators of GANs are often modeled by neural networks, posing a non-transparent optimization problem which is non-convex and non-concave over the generator and discriminator, respectively. Such networks are often heuristically optimized with gradient descent-ascent (GDA), but it is unclear whether the optimization problem contains any saddle points, or whether heuristic methods can find them in practice. In this work, we analyze the training of Wasserstein GANs with two-layer neural network discriminators through the lens of convex duality, and for a variety of generators expose the conditions under which Wasserstein GANs can be solved exactly with convex optimization approaches, or can be represented as convex-concave games. Using this convex duality interpretation, we further demonstrate the impact of different activation functions of the discriminator. Our observations are verified with numerical results demonstrating the power of the convex interpretation, with applications in progressive training of convex architectures corresponding to linear generators and quadratic-activation discriminators for CelebA image generation. The code for our experiments is available at https://github.com/ardasahiner/ProCoGAN.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our new arXiv preprint on Interpretable GANs. Leveraging convex duality we cast Wasserstein GAN optimization as a convex-concave game that amounts to moment matching.<a href="https://t.co/fUM6fdUZwn">https://t.co/fUM6fdUZwn</a> <a href="https://t.co/Ir5tdEhxN6">pic.twitter.com/Ir5tdEhxN6</a></p>&mdash; Morteza Mardani (@MardaniMorteza) <a href="https://twitter.com/MardaniMorteza/status/1415147653789208576?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval

Min Jin Chong, Wen-Sheng Chu, Abhishek Kumar

- retweets: 156, favorites: 31 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.06256) | [pdf](https://arxiv.org/pdf/2107.06256)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present Retrieve in Style (RIS), an unsupervised framework for fine-grained facial feature transfer and retrieval on real images. Recent work shows that it is possible to learn a catalog that allows local semantic transfers of facial features on generated images by capitalizing on the disentanglement property of the StyleGAN latent space. RIS improves existing art on: 1) feature disentanglement and allows for challenging transfers (i.e., hair and pose) that were not shown possible in SoTA methods. 2) eliminating the need for per-image hyperparameter tuning, and for computing a catalog over a large batch of images. 3) enabling face retrieval using the proposed facial features (e.g., eyes), and to our best knowledge, is the first work to retrieve face images at the fine-grained level. 4) robustness and natural application to real images. Our qualitative and quantitative analyses show RIS achieves both high-fidelity feature transfers and accurate fine-grained retrievals on real images. We discuss the responsible application of RIS.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Retrieve in Style: Unsupervised Facial Feature Transfer and Retrieval<br>pdf: <a href="https://t.co/hUJWidDdi9">https://t.co/hUJWidDdi9</a><br>github: <a href="https://t.co/GSXRUHHQFM">https://t.co/GSXRUHHQFM</a><br><br>qualitative and quantitative analyses show RIS achieves both high-fidelity feature transfers and accurate fine-grained retrievals on real images <a href="https://t.co/QBKf4LgTFQ">pic.twitter.com/QBKf4LgTFQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415114281956294660?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Towards Automatic Instrumentation by Learning to Separate Parts in  Symbolic Multitrack Music

Hao-Wen Dong, Chris Donahue, Taylor Berg-Kirkpatrick, Julian McAuley

- retweets: 120, favorites: 61 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05916) | [pdf](https://arxiv.org/pdf/2107.05916)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Modern keyboards allow a musician to play multiple instruments at the same time by assigning zones -- fixed pitch ranges of the keyboard -- to different instruments. In this paper, we aim to further extend this idea and examine the feasibility of automatic instrumentation -- dynamically assigning instruments to notes in solo music during performance. In addition to the online, real-time-capable setting for performative use cases, automatic instrumentation can also find applications in assistive composing tools in an offline setting. Due to the lack of paired data of original solo music and their full arrangements, we approach automatic instrumentation by learning to separate parts (e.g., voices, instruments and tracks) from their mixture in symbolic multitrack music, assuming that the mixture is to be played on a keyboard. We frame the task of part separation as a sequential multi-class classification problem and adopt machine learning to map sequences of notes into sequences of part labels. To examine the effectiveness of our proposed models, we conduct a comprehensive empirical evaluation over four diverse datasets of different genres and ensembles -- Bach chorales, string quartets, game music and pop music. Our experiments show that the proposed models outperform various baselines. We also demonstrate the potential for our proposed models to produce alternative convincing instrumentations for an existing arrangement by separating its mixture into parts. All source code and audio samples can be found at https://salu133445.github.io/arranger/ .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share that our paper &quot;Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music&quot; (work with <a href="https://twitter.com/chrisdonahuey?ref_src=twsrc%5Etfw">@chrisdonahuey</a>, Julian McAuley, <a href="https://twitter.com/BergKirkpatrick?ref_src=twsrc%5Etfw">@BergKirkpatrick</a>) was accepted to <a href="https://twitter.com/ismir2021?ref_src=twsrc%5Etfw">@ismir2021</a>🥳<br><br>📝paper: <a href="https://t.co/3f9PV4lvJP">https://t.co/3f9PV4lvJP</a><br>🎵demo: <a href="https://t.co/Ij5OFdX5gj">https://t.co/Ij5OFdX5gj</a> <a href="https://t.co/sdvlJ9RZrc">pic.twitter.com/sdvlJ9RZrc</a></p>&mdash; Hao-Wen Dong 董皓文 (@salu133445) <a href="https://twitter.com/salu133445/status/1415182100802654213?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Hyperparameter Optimization: Foundations, Algorithms, Best Practices and  Open Challenges

Bernd Bischl, Martin Binder, Michel Lang, Tobias Pielok, Jakob Richter, Stefan Coors, Janek Thomas, Theresa Ullmann, Marc Becker, Anne-Laure Boulesteix, Difan Deng, Marius Lindauer

- retweets: 110, favorites: 45 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05847) | [pdf](https://arxiv.org/pdf/2107.05847)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Most machine learning algorithms are configured by one or several hyperparameters that must be carefully chosen and often considerably impact performance. To avoid a time consuming and unreproducible manual trial-and-error process to find well-performing hyperparameter configurations, various automatic hyperparameter optimization (HPO) methods, e.g., based on resampling error estimation for supervised machine learning, can be employed. After introducing HPO from a general perspective, this paper reviews important HPO methods such as grid or random search, evolutionary algorithms, Bayesian optimization, Hyperband and racing. It gives practical recommendations regarding important choices to be made when conducting HPO, including the HPO algorithms themselves, performance evaluation, how to combine HPO with ML pipelines, runtime improvements, and parallelization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges. (arXiv:2107.05847v1 [<a href="https://t.co/zjV5HgYw5a">https://t.co/zjV5HgYw5a</a>]) <a href="https://t.co/mYtmJJFcxf">https://t.co/mYtmJJFcxf</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1415216433567502338?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Locally Enhanced Self-Attention: Rethinking Self-Attention as Local and  Context Terms

Chenglin Yang, Siyuan Qiao, Adam Kortylewski, Alan Yuille

- retweets: 90, favorites: 38 (07/15/2021 08:02:58)

- links: [abs](https://arxiv.org/abs/2107.05637) | [pdf](https://arxiv.org/pdf/2107.05637)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-Attention has become prevalent in computer vision models. Inspired by fully connected Conditional Random Fields (CRFs), we decompose it into local and context terms. They correspond to the unary and binary terms in CRF and are implemented by attention mechanisms with projection matrices. We observe that the unary terms only make small contributions to the outputs, and meanwhile standard CNNs that rely solely on the unary terms achieve great performances on a variety of tasks. Therefore, we propose Locally Enhanced Self-Attention (LESA), which enhances the unary term by incorporating it with convolutions, and utilizes a fusion module to dynamically couple the unary and binary operations. In our experiments, we replace the self-attention modules with LESA. The results on ImageNet and COCO show the superiority of LESA over convolution and self-attention baselines for the tasks of image recognition, object detection, and instance segmentation. The code is made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Locally Enhanced Self-Attention: Rethinking Self-Attention as Local and Context Terms<br>pdf: <a href="https://t.co/APtBEo6VEw">https://t.co/APtBEo6VEw</a><br>enhances the unary term by incorporating it with convolutions, and utilizes a fusion module to dynamically couple the unary and binary operations <a href="https://t.co/QmTupwcWRh">pic.twitter.com/QmTupwcWRh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1415135043786321921?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Representation Learning for Out-Of-Distribution Generalization in  Reinforcement Learning

Andrea Dittadi, Frederik Träuble, Manuel Wüthrich, Felix Widmaier, Peter Gehler, Ole Winther, Francesco Locatello, Olivier Bachem, Bernhard Schölkopf, Stefan Bauer

- retweets: 74, favorites: 20 (07/15/2021 08:02:59)

- links: [abs](https://arxiv.org/abs/2107.05686) | [pdf](https://arxiv.org/pdf/2107.05686)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Learning data representations that are useful for various downstream tasks is a cornerstone of artificial intelligence. While existing methods are typically evaluated on downstream tasks such as classification or generative image quality, we propose to assess representations through their usefulness in downstream control tasks, such as reaching or pushing objects. By training over 10,000 reinforcement learning policies, we extensively evaluate to what extent different representation properties affect out-of-distribution (OOD) generalization. Finally, we demonstrate zero-shot transfer of these policies from simulation to the real world, without any domain randomization or fine-tuning. This paper aims to establish the first systematic characterization of the usefulness of learned representations for real-world OOD downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How do properties of pre-trained representations affect the out-of-distribution generalization of reinforcement learning agents in simulation and the real world? Check out our large scale study <a href="https://t.co/KwmUvgRDXE">https://t.co/KwmUvgRDXE</a>. Joint first authors: <a href="https://twitter.com/andrea_dittadi?ref_src=twsrc%5Etfw">@andrea_dittadi</a> and <a href="https://twitter.com/f_traeuble?ref_src=twsrc%5Etfw">@f_traeuble</a>. <a href="https://t.co/v6UpOh6wuh">pic.twitter.com/v6UpOh6wuh</a></p>&mdash; Olivier Bachem (@OlivierBachem) <a href="https://twitter.com/OlivierBachem/status/1415315610288988163?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. How Could Equality and Data Protection Law Shape AI Fairness for People  with Disabilities?

Reuben Binns, Reuben Kirkham

- retweets: 30, favorites: 23 (07/15/2021 08:02:59)

- links: [abs](https://arxiv.org/abs/2107.05704) | [pdf](https://arxiv.org/pdf/2107.05704)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

This article examines the concept of 'AI fairness' for people with disabilities from the perspective of data protection and equality law. This examination demonstrates that there is a need for a distinctive approach to AI fairness that is fundamentally different to that used for other protected characteristics, due to the different ways in which discrimination and data protection law applies in respect of Disability. We articulate this new agenda for AI fairness for people with disabilities, explaining how combining data protection and equality law creates new opportunities for disabled people's organisations and assistive technology researchers alike to shape the use of AI, as well as to challenge potential harmful uses.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Formed an all-Reuben team with <a href="https://twitter.com/ReubenKirkham?ref_src=twsrc%5Etfw">@ReubenKirkham</a> to work on this paper (forthcoming in <a href="https://twitter.com/acmtaccess?ref_src=twsrc%5Etfw">@acmtaccess</a>):<br><br>&quot;How Could Equality and Data Protection Law Shape AI Fairness for People with Disabilities?&quot; <a href="https://t.co/IH1FgC8bkZ">https://t.co/IH1FgC8bkZ</a></p>&mdash; Reuben Binns (@RDBinns) <a href="https://twitter.com/RDBinns/status/1415209078708649987?ref_src=twsrc%5Etfw">July 14, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



