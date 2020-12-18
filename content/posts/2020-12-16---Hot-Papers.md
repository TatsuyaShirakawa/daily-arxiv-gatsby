---
title: Hot Papers 2020-12-16
date: 2020-12-18T15:37:35.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-16"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-16"
socialImage: "/media/flying-marine.jpg"

---

# 1. Learning Energy-Based Models by Diffusion Recovery Likelihood

Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, Diederik P. Kingma

- retweets: 1158, favorites: 193 (12/18/2020 15:37:35)

- links: [abs](https://arxiv.org/abs/2012.08125) | [pdf](https://arxiv.org/pdf/2012.08125)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

While energy-based models (EBMs) exhibit a number of desirable properties, training and sampling on high-dimensional datasets remains challenging. Inspired by recent progress on diffusion probabilistic models, we present a diffusion recovery likelihood method to tractably learn and sample from a sequence of EBMs trained on increasingly noisy versions of a dataset. Each EBM is trained by maximizing the recovery likelihood: the conditional probability of the data at a certain noise level given their noisy versions at a higher noise level. The recovery likelihood objective is more tractable than the marginal likelihood objective, since it only requires MCMC sampling from a relatively concentrated conditional distribution. Moreover, we show that this estimation method is theoretically consistent: it learns the correct conditional and marginal distributions at each noise level, given sufficient data. After training, synthesized images can be generated efficiently by a sampling process that initializes from a spherical Gaussian distribution and progressively samples the conditional distributions at decreasingly lower noise levels. Our method generates high fidelity samples on various image datasets. On unconditional CIFAR-10 our method achieves FID 9.60 and inception score 8.58, superior to the majority of GANs. Moreover, we demonstrate that unlike previous work on EBMs, our long-run MCMC samples from the conditional distributions do not diverge and still represent realistic images, allowing us to accurately estimate the normalized density of data even for high-dimensional datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pleased to share our new work on learning energy-based models: <a href="https://t.co/fX50RBGXn0">https://t.co/fX50RBGXn0</a><br>By maximizing recovery likelihoods on increasingly noisy data, the MCMC becomes more tractable. We achieve (1)high quality samples (2)stable long-run chains (3)estimated likelihoods. (1/n) <a href="https://t.co/qQpnYbpfMn">pic.twitter.com/qQpnYbpfMn</a></p>&mdash; Ruiqi Gao (@RuiqiGao) <a href="https://twitter.com/RuiqiGao/status/1339277277406760960?ref_src=twsrc%5Etfw">December 16, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



