---
title: Hot Papers 2021-05-25
date: 2021-05-26T09:34:06.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. True Few-Shot Learning with Language Models

Ethan Perez, Douwe Kiela, Kyunghyun Cho

- retweets: 7050, favorites: 2 (05/26/2021 09:34:06)

- links: [abs](https://arxiv.org/abs/2105.11447) | [pdf](https://arxiv.org/pdf/2105.11447)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Pretrained language models (LMs) perform well on many tasks even when learning from a few examples, but prior work uses many held-out examples to tune various aspects of learning, such as hyperparameters, training objectives, and natural language templates ("prompts"). Here, we evaluate the few-shot ability of LMs when such held-out examples are unavailable, a setting we call true few-shot learning. We test two model selection criteria, cross-validation and minimum description length, for choosing LM prompts and hyperparameters in the true few-shot setting. On average, both marginally outperform random selection and greatly underperform selection based on held-out examples. Moreover, selection criteria often prefer models that perform significantly worse than randomly-selected ones. We find similar results even when taking into account our uncertainty in a model's true performance during selection, as well as when varying the amount of computation and number of examples used for selection. Overall, our findings suggest that prior work significantly overestimated the true few-shot ability of LMs given the difficulty of few-shot model selection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Language models are amazing few-shot learners with the right prompt, but how do we choose the right prompt? It turns out that people use large held-out sets(!). How do models like GPT3 do in a true few-shot setting?<br><br>Much worse: <a href="https://t.co/3YsG0U98Cl">https://t.co/3YsG0U98Cl</a><br>w/ <a href="https://twitter.com/douwekiela?ref_src=twsrc%5Etfw">@douwekiela</a> <a href="https://twitter.com/kchonyc?ref_src=twsrc%5Etfw">@kchonyc</a><br>1/N <a href="https://t.co/kLPJ5WVXaO">pic.twitter.com/kLPJ5WVXaO</a></p>&mdash; Ethan Perez (@EthanJPerez) <a href="https://twitter.com/EthanJPerez/status/1397015129506541570?ref_src=twsrc%5Etfw">May 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Self-Attention Networks Can Process Bounded Hierarchical Languages

Shunyu Yao, Binghui Peng, Christos Papadimitriou, Karthik Narasimhan

- retweets: 106, favorites: 74 (05/26/2021 09:34:06)

- links: [abs](https://arxiv.org/abs/2105.11115) | [pdf](https://arxiv.org/pdf/2105.11115)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.FL](https://arxiv.org/list/cs.FL/recent)

Despite their impressive performance in NLP, self-attention networks were recently proved to be limited for processing formal languages with hierarchical structure, such as $\mathsf{Dyck}_k$, the language consisting of well-nested parentheses of $k$ types. This suggested that natural language can be approximated well with models that are too weak for formal languages, or that the role of hierarchy and recursion in natural language might be limited. We qualify this implication by proving that self-attention networks can process $\mathsf{Dyck}_{k, D}$, the subset of $\mathsf{Dyck}_{k}$ with depth bounded by $D$, which arguably better captures the bounded hierarchical structure of natural language. Specifically, we construct a hard-attention network with $D+1$ layers and $O(\log k)$ memory size (per token per layer) that recognizes $\mathsf{Dyck}_{k, D}$, and a soft-attention network with two layers and $O(\log k)$ memory size that generates $\mathsf{Dyck}_{k, D}$. Experiments show that self-attention networks trained on $\mathsf{Dyck}_{k, D}$ generalize to longer inputs with near-perfect accuracy, and also verify the theoretical memory advantage of self-attention networks over recurrent networks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Attention Networks Can Process Bounded Hierarchical Languages<br>pdf: <a href="https://t.co/aFQoYe2424">https://t.co/aFQoYe2424</a><br>abs: <a href="https://t.co/Jp5pBICgAg">https://t.co/Jp5pBICgAg</a> <a href="https://t.co/iabm6IWVdH">pic.twitter.com/iabm6IWVdH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1397009933804183559?ref_src=twsrc%5Etfw">May 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical structure is a core aspect of language syntax. Recurrent networks can systematically process recursion by emulating stacks, but can self-attention networks? If so, how?<br><br>Our <a href="https://twitter.com/hashtag/ACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021</a> paper shed lights into this fundamental issue!<a href="https://t.co/AX1e15vl0s">https://t.co/AX1e15vl0s</a><br><br>(1/5) <a href="https://t.co/MVMT3kMdSp">pic.twitter.com/MVMT3kMdSp</a></p>&mdash; Shunyu Yao (@ShunyuYao12) <a href="https://twitter.com/ShunyuYao12/status/1397047887763099650?ref_src=twsrc%5Etfw">May 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Embracing New Techniques in Deep Learning for Estimating Image  Memorability

Coen D. Needell, Wilma A. Bainbridge

- retweets: 132, favorites: 31 (05/26/2021 09:34:07)

- links: [abs](https://arxiv.org/abs/2105.10598) | [pdf](https://arxiv.org/pdf/2105.10598)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Various work has suggested that the memorability of an image is consistent across people, and thus can be treated as an intrinsic property of an image. Using computer vision models, we can make specific predictions about what people will remember or forget. While older work has used now-outdated deep learning architectures to predict image memorability, innovations in the field have given us new techniques to apply to this problem. Here, we propose and evaluate five alternative deep learning models which exploit developments in the field from the last five years, largely the introduction of residual neural networks, which are intended to allow the model to use semantic information in the memorability estimation process. These new models were tested against the prior state of the art with a combined dataset built to optimize both within-category and across-category predictions. Our findings suggest that the key prior memorability network had overstated its generalizability and was overfit on its training set. Our new models outperform this prior model, leading us to conclude that Residual Networks outperform simpler convolutional neural networks in memorability regression. We make our new state-of-the-art model readily available to the research community, allowing memory researchers to make predictions about memorability on a wider range of images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint w/ <a href="https://twitter.com/CoenNeedell?ref_src=twsrc%5Etfw">@CoenNeedell</a>: improving DNN predictions of image memorability! It uses conceptual info in residual networks to reach 68% prediction accuracy &amp; visualize the features. The model&#39;s easy to use--just go here! <a href="https://t.co/QuI4HuHPz3">https://t.co/QuI4HuHPz3</a><a href="https://t.co/0Jbvgo1Q5y">https://t.co/0Jbvgo1Q5y</a></p>&mdash; Wilma Bainbridge (@WilmaBainbridge) <a href="https://twitter.com/WilmaBainbridge/status/1396998986205175810?ref_src=twsrc%5Etfw">May 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Homotopies in Multiway (Non-Deterministic) Rewriting Systems as $n$-Fold  Categories

Xerxes D. Arsiwalla, Jonathan Gorard, Hatem Elshatlawy

- retweets: 35, favorites: 42 (05/26/2021 09:34:07)

- links: [abs](https://arxiv.org/abs/2105.10822) | [pdf](https://arxiv.org/pdf/2105.10822)
- [math.CT](https://arxiv.org/list/math.CT/recent) | [cs.DM](https://arxiv.org/list/cs.DM/recent) | [cs.LO](https://arxiv.org/list/cs.LO/recent) | [math-ph](https://arxiv.org/list/math-ph/recent) | [math.CO](https://arxiv.org/list/math.CO/recent)

We investigate the algebraic and compositional properties of multiway (non-deterministic) abstract rewriting systems, which are the archetypical structures underlying the formalism of the so-called Wolfram model. We demonstrate the existence of higher homotopies in this class of rewriting systems, where these homotopic maps are induced by the inclusion of appropriate rewriting rules taken from an abstract rulial space of all possible such rules. Furthermore, we show that a multiway rewriting system with homotopies up to order $n$ may naturally be formalized as an $n$-fold category, such that (upon inclusion of appropriate inverse morphisms via invertible rewriting relations) the infinite limit of this structure yields an ${\infty}$-groupoid. Via Grothendieck's homotopy hypothesis, this ${\infty}$-groupoid thus inherits the structure of a formal homotopy space. We conclude with some comments on how this computational framework of multiway rewriting systems may potentially be used for making formal connections to homotopy spaces upon which models of physics can be instantiated.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ICYMI, our two submissions to ACT 2021 are now on arXiv. 1st presents our work on multiway string diagram rewriting, with applications to quantum information: <a href="https://t.co/JD3IPSN8s4">https://t.co/JD3IPSN8s4</a><br>2nd is an exploration of multiway systems as models for cohesive HoTT: <a href="https://t.co/JpvlZ7FTZE">https://t.co/JpvlZ7FTZE</a> <a href="https://t.co/gLAcK10Kz5">pic.twitter.com/gLAcK10Kz5</a></p>&mdash; Jonathan Gorard (@getjonwithit) <a href="https://twitter.com/getjonwithit/status/1397097253773393920?ref_src=twsrc%5Etfw">May 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



