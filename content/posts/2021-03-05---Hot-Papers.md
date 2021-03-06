---
title: Hot Papers 2021-03-05
date: 2021-03-06T15:15:57.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. Accounting for Variance in Machine Learning Benchmarks

Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Naz Sepah, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Dmitriy Serdyuk, Tal Arbel, Chris Pal, Gaël Varoquaux, Pascal Vincent

- retweets: 3537, favorites: 189 (03/06/2021 15:15:57)

- links: [abs](https://arxiv.org/abs/2103.03098) | [pdf](https://arxiv.org/pdf/2103.03098)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Strong empirical evidence that one machine-learning algorithm A outperforms another one B ideally calls for multiple trials optimizing the learning pipeline over sources of variation such as data sampling, data augmentation, parameter initialization, and hyperparameters choices. This is prohibitively expensive, and corners are cut to reach conclusions. We model the whole benchmarking process, revealing that variance due to data sampling, parameter initialization and hyperparameter choice impact markedly the results. We analyze the predominant comparison methods used today in the light of this variance. We show a counter-intuitive result that adding more sources of variation to an imperfect estimator approaches better the ideal estimator at a 51 times reduction in compute cost. Building on these results, we study the error rate of detecting improvements, on five different deep-learning tasks/architectures. This study leads us to propose recommendations for performance comparisons.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint: Accounting for Variance in Machine Learning Benchmarks<a href="https://t.co/iu5QuMWzw8">https://t.co/iu5QuMWzw8</a><br>Lead by <a href="https://twitter.com/bouthilx?ref_src=twsrc%5Etfw">@bouthilx</a> and <a href="https://twitter.com/Mila_Quebec?ref_src=twsrc%5Etfw">@Mila_Quebec</a> friends<br><br>We show that ML benchmarks contain multiple sources of uncontrolled variation, not only inits. We propose procedure for reliable conclusion 1/8</p>&mdash; Gael Varoquaux (@GaelVaroquaux) <a href="https://twitter.com/GaelVaroquaux/status/1367839174469029899?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Out of Distribution Generalization in Machine Learning

Martin Arjovsky

- retweets: 1441, favorites: 261 (03/06/2021 15:15:58)

- links: [abs](https://arxiv.org/abs/2103.02667) | [pdf](https://arxiv.org/pdf/2103.02667)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Machine learning has achieved tremendous success in a variety of domains in recent years. However, a lot of these success stories have been in places where the training and the testing distributions are extremely similar to each other. In everyday situations when models are tested in slightly different data than they were trained on, ML algorithms can fail spectacularly. This research attempts to formally define this problem, what sets of assumptions are reasonable to make in our data and what kind of guarantees we hope to obtain from them. Then, we focus on a certain class of out of distribution problems, their assumptions, and introduce simple algorithms that follow from these assumptions that are able to provide more reliable generalization. A central topic in the thesis is the strong link between discovering the causal structure of the data, finding features that are reliable (when using them to predict) regardless of their context, and out of distribution generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Out of Distribution Generalization in Machine Learning<br><br>Martin Arjovsky&#39;s PhD thesis to review, contextualize, and clarify the current knowledge in out of distribution generalization.<a href="https://t.co/BQojqTcbPy">https://t.co/BQojqTcbPy</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1367660625929793536?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Perceiver: General Perception with Iterative Attention

Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira

- retweets: 857, favorites: 251 (03/06/2021 15:15:58)

- links: [abs](https://arxiv.org/abs/2103.03206) | [pdf](https://arxiv.org/pdf/2103.03206)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Biological systems understand the world by simultaneously processing high-dimensional inputs from modalities as diverse as vision, audition, touch, proprioception, etc. The perception models used in deep learning on the other hand are designed for individual modalities, often relying on domain-specific assumptions such as the local grid structures exploited by virtually all existing vision models. These priors introduce helpful inductive biases, but also lock models to individual modalities. In this paper we introduce the Perceiver - a model that builds upon Transformers and hence makes few architectural assumptions about the relationship between its inputs, but that also scales to hundreds of thousands of inputs, like ConvNets. The model leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck, allowing it to scale to handle very large inputs. We show that this architecture performs competitively or beyond strong, specialized models on classification tasks across various modalities: images, point clouds, audio, video and video+audio. The Perceiver obtains performance comparable to ResNet-50 on ImageNet without convolutions and by directly attending to 50,000 pixels. It also surpasses state-of-the-art results for all modalities in AudioSet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Perceiver: General Perception with Iterative Attention<br><br>- Competitive perf on classification tasks across various modalities: images, audio and video. <br><br>- Obtains perf comparable to ResNet on ImageNet w/o convs and by directly attending to 50,000 pixels. <a href="https://t.co/I8qpvFH2Z8">https://t.co/I8qpvFH2Z8</a> <a href="https://t.co/Lgf7Khfr08">pic.twitter.com/Lgf7Khfr08</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1367657530143383561?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Perceiver: General Perception with Iterative Attention　<a href="https://t.co/OHfgdlA0CD">https://t.co/OHfgdlA0CD</a><br>隠れベクトルと生の信号をcross attentionすることで高次元データをそのまま扱えるTransformer派生。画像、ビデオ、音声、ポイントクラウドなどを同じ構造で扱うことができる。ImageNetでResNet50並 <a href="https://t.co/KuHcr8cKjw">pic.twitter.com/KuHcr8cKjw</a></p>&mdash; ワクワクさん（ミジンコ） (@mosko_mule) <a href="https://twitter.com/mosko_mule/status/1367673823739351048?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Perceiver: General Perception with Iterative Attention<br>pdf: <a href="https://t.co/1EACnTm1Sj">https://t.co/1EACnTm1Sj</a><br>abs: <a href="https://t.co/JWverNYmDR">https://t.co/JWverNYmDR</a> <a href="https://t.co/cQRsBIcNWm">pic.twitter.com/cQRsBIcNWm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367657218942894081?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Improving Computational Efficiency in Visual Reinforcement Learning via  Stored Embeddings

Lili Chen, Kimin Lee, Aravind Srinivas, Pieter Abbeel

- retweets: 732, favorites: 243 (03/06/2021 15:15:58)

- links: [abs](https://arxiv.org/abs/2103.02886) | [pdf](https://arxiv.org/pdf/2103.02886)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent advances in off-policy deep reinforcement learning (RL) have led to impressive success in complex tasks from visual observations. Experience replay improves sample-efficiency by reusing experiences from the past, and convolutional neural networks (CNNs) process high-dimensional inputs effectively. However, such techniques demand high memory and computational bandwidth. In this paper, we present Stored Embeddings for Efficient Reinforcement Learning (SEER), a simple modification of existing off-policy RL methods, to address these computational and memory requirements. To reduce the computational overhead of gradient updates in CNNs, we freeze the lower layers of CNN encoders early in training due to early convergence of their parameters. Additionally, we reduce memory requirements by storing the low-dimensional latent vectors for experience replay instead of high-dimensional images, enabling an adaptive increase in the replay buffer capacity, a useful technique in constrained-memory settings. In our experiments, we show that SEER does not degrade the performance of RL agents while significantly saving computation and memory across a diverse set of DeepMind Control environments and Atari games. Finally, we show that SEER is useful for computation-efficient transfer learning in RL because lower layers of CNNs extract generalizable features, which can be used for different tasks and domains.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper, SEER,  improving both compute and memory efficiency of pixel-based RL.<br><br>Using two simple ideas: <br>(1) Freeze lower layers of CNN encoders early on in training; (2) Store latents in replay buffer instead of pixels.<br><br>🎓<a href="https://t.co/j4kDnNhO0N">https://t.co/j4kDnNhO0N</a><br>💻<a href="https://t.co/8Wxi3BaqGf">https://t.co/8Wxi3BaqGf</a> <a href="https://t.co/F97qooStgT">pic.twitter.com/F97qooStgT</a></p>&mdash; Aravind (@AravSrinivas) <a href="https://twitter.com/AravSrinivas/status/1367885931768807424?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Two ML papers with methods called &quot;SEER&quot; in 24 hours - self-supervised image recognition from Facebook: <a href="https://t.co/e65Jr00hVH">https://t.co/e65Jr00hVH</a><br><br>And more efficient visual RL from Berkeley: <a href="https://t.co/2TzzO04N9K">https://t.co/2TzzO04N9K</a></p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1367692218287366145?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improving Computational Efficiency in Visual Reinforcement Learning via Stored Embeddings<br><br>SEER saves significant computation and memory across DeepMind Control environments and Atari games without degrading the performance.<a href="https://t.co/37vJ1Otg7i">https://t.co/37vJ1Otg7i</a> <a href="https://t.co/noNhexMVYe">pic.twitter.com/noNhexMVYe</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1367658921477873664?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Catala: A Programming Language for the Law

Denis Merigoux, Nicolas Chataing, Jonathan Protzenko

- retweets: 648, favorites: 95 (03/06/2021 15:15:59)

- links: [abs](https://arxiv.org/abs/2103.03198) | [pdf](https://arxiv.org/pdf/2103.03198)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

Law at large underpins modern society, codifying and governing many aspects of citizens' daily lives. Oftentimes, law is subject to interpretation, debate and challenges throughout various courts and jurisdictions. But in some other areas, law leaves little room for interpretation, and essentially aims to rigorously describe a computation, a decision procedure or, simply said, an algorithm. Unfortunately, prose remains a woefully inadequate tool for the job. The lack of formalism leaves room for ambiguities; the structure of legal statutes, with many paragraphs and sub-sections spread across multiple pages, makes it hard to compute the intended outcome of the algorithm underlying a given text; and, as with any other piece of poorly-specified critical software, the use of informal language leaves corner cases unaddressed. We introduce Catala, a new programming language that we specifically designed to allow a straightforward and systematic translation of statutory law into an executable implementation. Catala aims to bring together lawyers and programmers through a shared medium, which together they can understand, edit and evolve, bridging a gap that often results in dramatically incorrect implementations of the law. We have implemented a compiler for Catala, and have proven the correctness of its core compilation steps using the F* proof assistant. We evaluate Catala on several legal texts that are algorithms in disguise, notably section 121 of the US federal income tax and the byzantine French family benefits; in doing so, we uncover a bug in the official implementation. We observe as a consequence of the formalization process that using Catala enables rich interactions between lawyers and programmers, leading to a greater understanding of the original legislative intent, while producing a correct-by-construction executable specification reusable by the greater software ecosystem.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Catala: a programming language for the law <a href="https://t.co/Qd9ZhwPFNH">https://t.co/Qd9ZhwPFNH</a><br><br>&quot;We evaluate Catala on several legal texts (...), notably section 121 of the US federal income tax and the byzantine French family benefits; in doing so, we uncover a bug in the official implementation.&quot;</p>&mdash; ly(s)xia (@lysxia) <a href="https://twitter.com/lysxia/status/1367860882206822401?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What is the future of legal expert systems? How to make sure taxes are computed correctly? Can we efficiently translate law into <a href="https://twitter.com/hashtag/rulesascode?src=hash&amp;ref_src=twsrc%5Etfw">#rulesascode</a>?  <br><br>New paper with <a href="https://twitter.com/NChataing?ref_src=twsrc%5Etfw">@NChataing</a> and <a href="https://twitter.com/_protz_?ref_src=twsrc%5Etfw">@_protz_</a>:<br><br>➡️ <a href="https://t.co/TMkXbpy0s4">https://t.co/TMkXbpy0s4</a><br>📖 <a href="https://t.co/WBkLp3mmz5">https://t.co/WBkLp3mmz5</a><br>🚀 <a href="https://t.co/wH1usUT9Cm">https://t.co/wH1usUT9Cm</a> <a href="https://t.co/Gs8qp9xRjA">pic.twitter.com/Gs8qp9xRjA</a></p>&mdash; Denis Merigoux (@DMerigoux) <a href="https://twitter.com/DMerigoux/status/1367780398730780675?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. GenoML: Automated Machine Learning for Genomics

Mary B. Makarious, Hampton L. Leonard, Dan Vitale, Hirotaka Iwaki, David Saffo, Lana Sargent, Anant Dadu, Eduardo Salmerón Castaño, John F. Carter, Melina Maleknia, Juan A. Botia, Cornelis Blauwendraat, Roy H. Campbell, Sayed Hadi Hashemi, Andrew B. Singleton, Mike A. Nalls, Faraz Faghri

- retweets: 589, favorites: 114 (03/06/2021 15:15:59)

- links: [abs](https://arxiv.org/abs/2103.03221) | [pdf](https://arxiv.org/pdf/2103.03221)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

GenoML is a Python package automating machine learning workflows for genomics (genetics and multi-omics) with an open science philosophy. Genomics data require significant domain expertise to clean, pre-process, harmonize and perform quality control of the data. Furthermore, tuning, validation, and interpretation involve taking into account the biology and possibly the limitations of the underlying data collection, protocols, and technology. GenoML's mission is to bring machine learning for genomics and clinical data to non-experts by developing an easy-to-use tool that automates the full development, evaluation, and deployment process. Emphasis is put on open science to make workflows easily accessible, replicable, and transferable within the scientific community. Source code and documentation is available at https://genoml.com.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GenoML: Automated Machine Learning for Genomics<br>pdf: <a href="https://t.co/E4NNpz9g4R">https://t.co/E4NNpz9g4R</a><br>abs: <a href="https://t.co/1CfAufCsZY">https://t.co/1CfAufCsZY</a><br>project page: <a href="https://t.co/5NRkDdePpf">https://t.co/5NRkDdePpf</a> <a href="https://t.co/mCLtAeAHyT">pic.twitter.com/mCLtAeAHyT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367694608117927940?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Anycost GANs for Interactive Image Synthesis and Editing

Ji Lin, Richard Zhang, Frieder Ganz, Song Han, Jun-Yan Zhu

- retweets: 441, favorites: 102 (03/06/2021 15:16:00)

- links: [abs](https://arxiv.org/abs/2103.03243) | [pdf](https://arxiv.org/pdf/2103.03243)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative adversarial networks (GANs) have enabled photorealistic image synthesis and editing. However, due to the high computational cost of large-scale generators (e.g., StyleGAN2), it usually takes seconds to see the results of a single edit on edge devices, prohibiting interactive user experience. In this paper, we take inspirations from modern rendering software and propose Anycost GAN for interactive natural image editing. We train the Anycost GAN to support elastic resolutions and channels for faster image generation at versatile speeds. Running subsets of the full generator produce outputs that are perceptually similar to the full generator, making them a good proxy for preview. By using sampling-based multi-resolution training, adaptive-channel training, and a generator-conditioned discriminator, the anycost generator can be evaluated at various configurations while achieving better image quality compared to separately trained models. Furthermore, we develop new encoder training and latent code optimization techniques to encourage consistency between the different sub-generators during image projection. Anycost GAN can be executed at various cost budgets (up to 10x computation reduction) and adapt to a wide range of hardware and latency requirements. When deployed on desktop CPUs and edge devices, our model can provide perceptually similar previews at 6-12x speedup, enabling interactive image editing. The code and demo are publicly available: https://github.com/mit-han-lab/anycost-gan.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Anycost GANs for Interactive Image Synthesis and Editing<br>pdf: <a href="https://t.co/i81p9MRaj3">https://t.co/i81p9MRaj3</a><br>abs: <a href="https://t.co/2hDmFy9OTG">https://t.co/2hDmFy9OTG</a><br>github: <a href="https://t.co/W6PHWlLWX5">https://t.co/W6PHWlLWX5</a><br>project page: <a href="https://t.co/mFoPI7PI1B">https://t.co/mFoPI7PI1B</a> <a href="https://t.co/0hJdqHqlXf">pic.twitter.com/0hJdqHqlXf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367662344667545600?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Self-supervised Geometric Perception

Heng Yang, Wei Dong, Luca Carlone, Vladlen Koltun

- retweets: 334, favorites: 116 (03/06/2021 15:16:00)

- links: [abs](https://arxiv.org/abs/2103.03114) | [pdf](https://arxiv.org/pdf/2103.03114)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present self-supervised geometric perception (SGP), the first general framework to learn a feature descriptor for correspondence matching without any ground-truth geometric model labels (e.g., camera poses, rigid transformations). Our first contribution is to formulate geometric perception as an optimization problem that jointly optimizes the feature descriptor and the geometric models given a large corpus of visual measurements (e.g., images, point clouds). Under this optimization formulation, we show that two important streams of research in vision, namely robust model fitting and deep feature learning, correspond to optimizing one block of the unknown variables while fixing the other block. This analysis naturally leads to our second contribution -- the SGP algorithm that performs alternating minimization to solve the joint optimization. SGP iteratively executes two meta-algorithms: a teacher that performs robust model fitting given learned features to generate geometric pseudo-labels, and a student that performs deep feature learning under noisy supervision of the pseudo-labels. As a third contribution, we apply SGP to two perception problems on large-scale real datasets, namely relative camera pose estimation on MegaDepth and point cloud registration on 3DMatch. We demonstrate that SGP achieves state-of-the-art performance that is on-par or superior to the supervised oracles trained using ground-truth labels.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Geometric Perception<br>pdf: <a href="https://t.co/fi3XAQLhCM">https://t.co/fi3XAQLhCM</a><br>abs: <a href="https://t.co/vLjCN1l2NO">https://t.co/vLjCN1l2NO</a> <a href="https://t.co/cJuEDeqntk">pic.twitter.com/cJuEDeqntk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367664973720268809?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Geometric Perception, with W. Dong, <a href="https://twitter.com/lucacarlone1?ref_src=twsrc%5Etfw">@lucacarlone1</a>, V. Koltun, is accepted as <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> Oral. Appreciated the discussion w/ <a href="https://twitter.com/ducha_aiki?ref_src=twsrc%5Etfw">@ducha_aiki</a> on difference b/w SGP and reconstruction-based supervised learning. Check out future research on Page 8:<a href="https://t.co/1oSejKamCA">https://t.co/1oSejKamCA</a> <a href="https://t.co/07eq2WqLl8">pic.twitter.com/07eq2WqLl8</a></p>&mdash; Heng Yang (@hankyang94) <a href="https://twitter.com/hankyang94/status/1367849991558008836?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-supervised Geometric Perception<a href="https://twitter.com/hankyang94?ref_src=twsrc%5Etfw">@hankyang94</a>, <a href="https://twitter.com/wdong397?ref_src=twsrc%5Etfw">@wdong397</a>, <a href="https://twitter.com/lucacarlone1?ref_src=twsrc%5Etfw">@lucacarlone1</a>, Vladlen Koltun<br><br>main idea: train CAPS by <a href="https://twitter.com/Jimantha?ref_src=twsrc%5Etfw">@Jimantha</a> with Es generated by RANSAC. Use SIFT descs for 1st iteration, then use learned descriptor.<a href="https://t.co/FZfFf2gB8b">https://t.co/FZfFf2gB8b</a><br><br>Short review: in the thread <a href="https://t.co/46oTwcrG4i">pic.twitter.com/46oTwcrG4i</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1367787745880510466?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. COIN: COmpression with Implicit Neural representations

Emilien Dupont, Adam Goliński, Milad Alizadeh, Yee Whye Teh, Arnaud Doucet

- retweets: 285, favorites: 88 (03/06/2021 15:16:00)

- links: [abs](https://arxiv.org/abs/2103.03123) | [pdf](https://arxiv.org/pdf/2103.03123)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a new simple approach for image compression: instead of storing the RGB values for each pixel of an image, we store the weights of a neural network overfitted to the image. Specifically, to encode an image, we fit it with an MLP which maps pixel locations to RGB values. We then quantize and store the weights of this MLP as a code for the image. To decode the image, we simply evaluate the MLP at every pixel location. We found that this simple approach outperforms JPEG at low bit-rates, even without entropy coding or learning a distribution over weights. While our framework is not yet competitive with state of the art compression methods, we show that it has various attractive properties which could make it a viable alternative to other neural data compression approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Galaxy brain compression method: &quot;instead of storing the RGB values for each pixel..., we store the weights of a neural network overfitted to the image. ... not yet competitive with state of the art compression methods [but] various attractive properties&quot;: <a href="https://t.co/LP56NeztFY">https://t.co/LP56NeztFY</a></p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1367691391082491907?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Enhanced 3D Human Pose Estimation from Videos by using Attention-Based  Neural Network with Dilated Convolutions

Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung, Vijayan K. Asari

- retweets: 203, favorites: 80 (03/06/2021 15:16:00)

- links: [abs](https://arxiv.org/abs/2103.03170) | [pdf](https://arxiv.org/pdf/2103.03170)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The attention mechanism provides a sequential prediction framework for learning spatial models with enhanced implicit temporal consistency. In this work, we show a systematic design (from 2D to 3D) for how conventional networks and other forms of constraints can be incorporated into the attention framework for learning long-range dependencies for the task of pose estimation. The contribution of this paper is to provide a systematic approach for designing and training of attention-based models for the end-to-end pose estimation, with the flexibility and scalability of arbitrary video sequences as input. We achieve this by adapting temporal receptive field via a multi-scale structure of dilated convolutions. Besides, the proposed architecture can be easily adapted to a causal model enabling real-time performance. Any off-the-shelf 2D pose estimation systems, e.g. Mocap libraries, can be easily integrated in an ad-hoc fashion. Our method achieves the state-of-the-art performance and outperforms existing methods by reducing the mean per joint position error to 33.4 mm on Human3.6M dataset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Enhanced 3D Human Pose Estimation from Videos by using Attention-Based Neural Network with Dilated Convolutions<br>pdf: <a href="https://t.co/uUqyjvGdG6">https://t.co/uUqyjvGdG6</a><br>abs: <a href="https://t.co/J2f1jHUimX">https://t.co/J2f1jHUimX</a><br>github: <a href="https://t.co/KVYpcQtHcR">https://t.co/KVYpcQtHcR</a> <a href="https://t.co/Bje0eBS7yu">pic.twitter.com/Bje0eBS7yu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367716359573020678?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Data Augmentation for Object Detection via Differentiable Neural  Rendering

Guanghan Ning, Guang Chen, Chaowei Tan, Si Luo, Liefeng Bo, Heng Huang

- retweets: 40, favorites: 61 (03/06/2021 15:16:01)

- links: [abs](https://arxiv.org/abs/2103.02852) | [pdf](https://arxiv.org/pdf/2103.02852)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

It is challenging to train a robust object detector when annotated data is scarce. Existing approaches to tackle this problem include semi-supervised learning that interpolates labeled data from unlabeled data, self-supervised learning that exploit signals within unlabeled data via pretext tasks. Without changing the supervised learning paradigm, we introduce an offline data augmentation method for object detection, which semantically interpolates the training data with novel views. Specifically, our proposed system generates controllable views of training images based on differentiable neural rendering, together with corresponding bounding box annotations which involve no human intervention. Firstly, we extract and project pixel-aligned image features into point clouds while estimating depth maps. We then re-project them with a target camera pose and render a novel-view 2d image. Objects in the form of keypoints are marked in point clouds to recover annotations in new views. It is fully compatible with online data augmentation methods, such as affine transform, image mixup, etc. Extensive experiments show that our method, as a cost-free tool to enrich images and labels, can significantly boost the performance of object detection systems with scarce training data. Code is available at \url{https://github.com/Guanghan/DANR}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Data Augmentation for Object Detection via Differentiable Neural Rendering<br>pdf: <a href="https://t.co/DbG4JX1tM2">https://t.co/DbG4JX1tM2</a><br>abs: <a href="https://t.co/OCspgEedYe">https://t.co/OCspgEedYe</a> <a href="https://t.co/Dhts6ym93t">pic.twitter.com/Dhts6ym93t</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367676717200969731?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. MOGAN: Morphologic-structure-aware Generative Learning from a Single  Image

Jinshu Chen, Qihui Xu, Qi Kang, MengChu Zhou

- retweets: 58, favorites: 25 (03/06/2021 15:16:01)

- links: [abs](https://arxiv.org/abs/2103.02997) | [pdf](https://arxiv.org/pdf/2103.02997)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In most interactive image generation tasks, given regions of interest (ROI) by users, the generated results are expected to have adequate diversities in appearance while maintaining correct and reasonable structures in original images. Such tasks become more challenging if only limited data is available. Recently proposed generative models complete training based on only one image. They pay much attention to the monolithic feature of the sample while ignoring the actual semantic information of different objects inside the sample. As a result, for ROI-based generation tasks, they may produce inappropriate samples with excessive randomicity and without maintaining the related objects' correct structures. To address this issue, this work introduces a MOrphologic-structure-aware Generative Adversarial Network named MOGAN that produces random samples with diverse appearances and reliable structures based on only one image. For training for ROI, we propose to utilize the data coming from the original image being augmented and bring in a novel module to transform such augmented data into knowledge containing both structures and appearances, thus enhancing the model's comprehension of the sample. To learn the rest areas other than ROI, we employ binary masks to ensure the generation isolated from ROI. Finally, we set parallel and hierarchical branches of the mentioned learning process. Compared with other single image GAN schemes, our approach focuses on internal features including the maintenance of rational structures and variation on appearance. Experiments confirm a better capacity of our model on ROI-based image generation tasks than its competitive peers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MOGAN: Morphologic-structure-aware Generative Learning from a Single Image<br>pdf: <a href="https://t.co/IDS2mAA8vb">https://t.co/IDS2mAA8vb</a><br>abs: <a href="https://t.co/6in9hRE5CO">https://t.co/6in9hRE5CO</a> <a href="https://t.co/XE5WaNmxSU">pic.twitter.com/XE5WaNmxSU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1367667586045972480?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Continuous Coordination As a Realistic Scenario for Lifelong Learning

Hadi Nekoei, Akilesh Badrinaaraayanan, Aaron Courville, Sarath Chandar

- retweets: 20, favorites: 32 (03/06/2021 15:16:01)

- links: [abs](https://arxiv.org/abs/2103.03216) | [pdf](https://arxiv.org/pdf/2103.03216)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

Current deep reinforcement learning (RL) algorithms are still highly task-specific and lack the ability to generalize to new environments. Lifelong learning (LLL), however, aims at solving multiple tasks sequentially by efficiently transferring and using knowledge between tasks. Despite a surge of interest in lifelong RL in recent years, the lack of a realistic testbed makes robust evaluation of LLL algorithms difficult. Multi-agent RL (MARL), on the other hand, can be seen as a natural scenario for lifelong RL due to its inherent non-stationarity, since the agents' policies change over time. In this work, we introduce a multi-agent lifelong learning testbed that supports both zero-shot and few-shot settings. Our setup is based on Hanabi -- a partially-observable, fully cooperative multi-agent game that has been shown to be challenging for zero-shot coordination. Its large strategy space makes it a desirable environment for lifelong RL tasks. We evaluate several recent MARL methods, and benchmark state-of-the-art LLL algorithms in limited memory and computation regimes to shed light on their strengths and weaknesses. This continual learning paradigm also provides us with a pragmatic way of going beyond centralized training which is the most commonly used training protocol in MARL. We empirically show that the agents trained in our setup are able to coordinate well with unseen agents, without any additional assumptions made by previous works.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are you tired of manually creating new tasks for Lifelong RL? We introduce Lifelong Hanabi in which every task is coordinating with a partner that&#39;s an expert player of Hanabi. Work led by <a href="https://twitter.com/HadiNekoei?ref_src=twsrc%5Etfw">@HadiNekoei</a> and <a href="https://twitter.com/akileshbadri?ref_src=twsrc%5Etfw">@akileshbadri</a>.<br>paper: <a href="https://t.co/8b4Gk3MlOI">https://t.co/8b4Gk3MlOI</a>  <a href="https://twitter.com/Mila_Quebec?ref_src=twsrc%5Etfw">@Mila_Quebec</a>  1/n <a href="https://t.co/LmQBrpyFxC">pic.twitter.com/LmQBrpyFxC</a></p>&mdash; sarath chandar (@apsarathchandar) <a href="https://twitter.com/apsarathchandar/status/1367920278605598727?ref_src=twsrc%5Etfw">March 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



