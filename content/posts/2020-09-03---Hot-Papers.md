---
title: Hot Papers 2020-09-03
date: 2020-09-04T09:52:41.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. WaveGrad: Estimating Gradients for Waveform Generation

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan

- retweets: 55, favorites: 279 (09/04/2020 09:52:41)

- links: [abs](https://arxiv.org/abs/2009.00713) | [pdf](https://arxiv.org/pdf/2009.00713)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

This paper introduces WaveGrad, a conditional model for waveform generation through estimating gradients of the data density. This model is built on the prior work on score matching and diffusion probabilistic models. It starts from Gaussian white noise and iteratively refines the signal via a gradient-based sampler conditioned on the mel-spectrogram. WaveGrad is non-autoregressive, and requires only a constant number of generation steps during inference. It can use as few as 6 iterations to generate high fidelity audio samples. WaveGrad is simple to train, and implicitly optimizes for the weighted variational lower-bound of the log-likelihood. Empirical experiments reveal WaveGrad to generate high fidelity audio samples matching a strong likelihood-based autoregressive baseline with less sequential operations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WaveGrad generates waveforms from spectrograms by iteratively following the log-likelihood gradient. The surprising thing is that it needs as little as 6 steps to produce good quality audio! <a href="https://t.co/No5AVcc7x1">https://t.co/No5AVcc7x1</a><br><br>Seems like the resurgence of score matching is in full swing :) <a href="https://t.co/IdpbluIAFg">https://t.co/IdpbluIAFg</a> <a href="https://t.co/nzwozv3Jcr">pic.twitter.com/nzwozv3Jcr</a></p>&mdash; Sander Dieleman (@sedielem) <a href="https://twitter.com/sedielem/status/1301328244529135621?ref_src=twsrc%5Etfw">September 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">WaveGradは条件付き波形生成において、拡散確率モデルから導出される重み付きdenoising score matchingで対数尤度勾配を学習し、それを使ったランジュバン動力学でサンプリングする。高精度かつ自己回帰モデルより高速。生成モデルで拡散確率モデルが有望になってきている <a href="https://t.co/kYkRN6eK7p">https://t.co/kYkRN6eK7p</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1301661158886129665?ref_src=twsrc%5Etfw">September 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">refine, baby, refine! <a href="https://t.co/uGUoVl1gOC">https://t.co/uGUoVl1gOC</a></p>&mdash; Kyunghyun Cho (@kchonyc) <a href="https://twitter.com/kchonyc/status/1301368566567120896?ref_src=twsrc%5Etfw">September 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Automated Storytelling via Causal, Commonsense Plot Ordering

Prithviraj Ammanabrolu, Wesley Cheung, William Broniec, Mark O. Riedl

- retweets: 10, favorites: 68 (09/04/2020 09:52:41)

- links: [abs](https://arxiv.org/abs/2009.00829) | [pdf](https://arxiv.org/pdf/2009.00829)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Automated story plot generation is the task of generating a coherent sequence of plot events. Causal relations between plot events are believed to increase the perception of story and plot coherence. In this work, we introduce the concept of soft causal relations as causal relations inferred from commonsense reasoning. We demonstrate C2PO, an approach to narrative generation that operationalizes this concept through Causal, Commonsense Plot Ordering. Using human-participant protocols, we evaluate our system against baseline systems with different commonsense reasoning reasoning and inductive biases to determine the role of soft causal relations in perceived story quality. Through these studies we also probe the interplay of how changes in commonsense norms across storytelling genres affect perceptions of story quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">C2PO: Generating commonsense story plots by querying a neural commonsense model<br><br>We introduce the concept of a “soft causal relation” to bridge between classical, symbolic story planning and neural story generation<br><br>arxiv: <a href="https://t.co/XF4j6l92b0">https://t.co/XF4j6l92b0</a><br>github: <a href="https://t.co/M4n7VfW7Pt">https://t.co/M4n7VfW7Pt</a> <a href="https://t.co/DCzYVQ5Sd8">pic.twitter.com/DCzYVQ5Sd8</a></p>&mdash; Mark O. Riedl (@mark_riedl) <a href="https://twitter.com/mark_riedl/status/1301551778509455366?ref_src=twsrc%5Etfw">September 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Java Cryptography Uses in the Wild

Mohammadreza Hazhirpasand, Mohammad Ghafari, Oscar Nierstrasz

- retweets: 17, favorites: 47 (09/04/2020 09:52:41)

- links: [abs](https://arxiv.org/abs/2009.01101) | [pdf](https://arxiv.org/pdf/2009.01101)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.SE](https://arxiv.org/list/cs.SE/recent)

[Background] Previous research has shown that developers commonly misuse cryptography APIs. [Aim] We have conducted an exploratory study to find out how crypto APIs are used in open-source Java projects, what types of misuses exist, and why developers make such mistakes. [Method] We used a static analysis tool to analyze hundreds of open-source Java projects that rely on Java Cryptography Architecture, and manually inspected half of the analysis results to assess the tool results. We also contacted the maintainers of these projects by creating an issue on the GitHub repository of each project, and discussed the misuses with developers. [Results] We learned that 85% of Cryptography APIs are misused, however, not every misuse has severe consequences. Developer feedback showed that security caveats in the documentation of crypto APIs are rare, developers may overlook misuses that originate in third-party code, and the context where a Crypto API is used should be taken into account. [Conclusion] We conclude that using Crypto APIs is still problematic for developers but blindly blaming them for such misuses may lead to erroneous conclusions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Java Cryptography Uses in the Wild&quot;<br><br>85% of cryptographic APIs are misused in Java project. The discussions included in the paper give some insightful views on the social aspects of this.<a href="https://t.co/ZOMIGnrR9i">https://t.co/ZOMIGnrR9i</a><a href="https://t.co/OMlqXPz1uX">https://t.co/OMlqXPz1uX</a><br>Dataset: <a href="https://t.co/dI2VlroHn8">https://t.co/dI2VlroHn8</a> <a href="https://t.co/31ktLCzgdn">pic.twitter.com/31ktLCzgdn</a></p>&mdash; Alexandre Dulaunoy (@adulau) <a href="https://twitter.com/adulau/status/1301473335365246981?ref_src=twsrc%5Etfw">September 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



