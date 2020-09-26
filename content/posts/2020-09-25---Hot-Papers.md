---
title: Hot Papers 2020-09-25
date: 2020-09-26T09:23:19.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. Grounded Compositional Outputs for Adaptive Language Modeling

Nikolaos Pappas, Phoebe Mulcaire, Noah A. Smith

- retweets: 210, favorites: 114 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11523) | [pdf](https://arxiv.org/pdf/2009.11523)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Language models have emerged as a central component across NLP, and a great deal of progress depends on the ability to cheaply adapt them (e.g., through finetuning) to new domains and tasks. A language model's \emph{vocabulary}---typically selected before training and permanently fixed later---affects its size and is part of what makes it resistant to such adaptation. Prior work has used compositional input embeddings based on surface forms to ameliorate this issue. In this work, we go one step beyond and propose a fully compositional output embedding layer for language models, which is further grounded in information from a structured lexicon (WordNet), namely semantically related words and free-text definitions. To our knowledge, the result is the first word-level language model with a size that does not depend on the training vocabulary. We evaluate the model on conventional language modeling as well as challenging cross-domain settings with an open vocabulary, finding that it matches or outperforms previous state-of-the-art output embedding methods and adaptation approaches. Our analysis attributes the improvements to sample efficiency: our model is more accurate for low-frequency words.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GroC:  a new (word-level) language modeling approach using compositional outputs, so it&#39;s ready to score new words not seen in training and it doesn&#39;t need to grow with the vocabulary size.  Work by <a href="https://twitter.com/nik0spapp?ref_src=twsrc%5Etfw">@nik0spapp</a> <a href="https://twitter.com/PhoebeNLP?ref_src=twsrc%5Etfw">@PhoebeNLP</a> <a href="https://twitter.com/nlpnoah?ref_src=twsrc%5Etfw">@nlpnoah</a> to appear at EMNLP 2020.  <a href="https://t.co/K2wekA7u4K">https://t.co/K2wekA7u4K</a></p>&mdash; Noah A Smith (@nlpnoah) <a href="https://twitter.com/nlpnoah/status/1309305766923218945?ref_src=twsrc%5Etfw">September 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. A Unifying Review of Deep and Shallow Anomaly Detection

Lukas Ruff, Jacob R. Kauffmann, Robert A. Vandermeulen, Grégoire Montavon, Wojciech Samek, Marius Kloft, Thomas G. Dietterich, Klaus-Robert Müller

- retweets: 120, favorites: 72 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11732) | [pdf](https://arxiv.org/pdf/2009.11732)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Deep learning approaches to anomaly detection have recently improved the state of the art in detection performance on complex datasets such as large collections of images or text. These results have sparked a renewed interest in the anomaly detection problem and led to the introduction of a great variety of new methods. With the emergence of numerous such methods, including approaches based on generative models, one-class classification, and reconstruction, there is a growing need to bring methods of this field into a systematic and unified perspective. In this review we aim to identify the common underlying principles as well as the assumptions that are often made implicitly by various methods. In particular, we draw connections between classic 'shallow' and novel deep approaches and show how this relation might cross-fertilize or extend both directions. We further provide an empirical assessment of major existing methods that is enriched by the use of recent explainability techniques, and present specific worked-through examples together with practical advice. Finally, we outline critical open challenges and identify specific paths for future research in anomaly detection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: <a href="https://t.co/BO6xu6C1uT">https://t.co/BO6xu6C1uT</a> <a href="https://twitter.com/lukasruff?ref_src=twsrc%5Etfw">@lukasruff</a> led our team in this attempt to unify various perspectives on deep anomaly detection within a probabilistic framework. Jacob Kauffmann, <a href="https://twitter.com/robvdm?ref_src=twsrc%5Etfw">@robvdm</a><br>, Grégoire Montavon, <a href="https://twitter.com/WojciechSamek?ref_src=twsrc%5Etfw">@WojciechSamek</a><br>, <a href="https://twitter.com/KloftMarius?ref_src=twsrc%5Etfw">@KloftMarius</a><br>, and Klaus-Robert Müller.</p>&mdash; Thomas G. Dietterich (@tdietterich) <a href="https://twitter.com/tdietterich/status/1309522438032486402?ref_src=twsrc%5Etfw">September 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. A Unified Analysis of First-Order Methods for Smooth Games via Integral  Quadratic Constraints

Guodong Zhang, Xuchao Bao, Laurent Lessard, Roger Grosse

- retweets: 78, favorites: 51 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11359) | [pdf](https://arxiv.org/pdf/2009.11359)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The theory of integral quadratic constraints (IQCs) allows the certification of exponential convergence of interconnected systems containing nonlinear or uncertain elements. In this work, we adapt the IQC theory to study first-order methods for smooth and strongly-monotone games and show how to design tailored quadratic constraints to get tight upper bounds of convergence rates. Using this framework, we recover the existing bound for the gradient method~(GD), derive sharper bounds for the proximal point method~(PPM) and optimistic gradient method~(OG), and provide \emph{for the first time} a global convergence rate for the negative momentum method~(NM) with an iteration complexity $\bigo(\kappa^{1.5})$, which matches its known lower bound. In addition, for time-varying systems, we prove that the gradient method with optimal step size achieves the fastest provable worst-case convergence rate with quadratic Lyapunov functions. Finally, we further extend our analysis to stochastic games and study the impact of multiplicative noise on different algorithms. We show that it is impossible for an algorithm with one step of memory to achieve acceleration if it only queries the gradient once per batch (in contrast with the stochastic strongly-convex optimization setting, where such acceleration has been demonstrated). However, we exhibit an algorithm which achieves acceleration with two gradient queries per batch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper alert: <a href="https://t.co/qfmEYSrcPh">https://t.co/qfmEYSrcPh</a><br><br>We provide a unified and automated method to analyze first-order methods for smooth &amp; strongly-monotone games. The convergence rate for any first-order method can be obtained via a mechanical procedure of deriving and solving an SDP. <a href="https://t.co/moo3Ebx7t8">pic.twitter.com/moo3Ebx7t8</a></p>&mdash; Guodong Zhang (@Guodzh) <a href="https://twitter.com/Guodzh/status/1309300896208171009?ref_src=twsrc%5Etfw">September 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A Personal Perspective on Numerical Analysis and Optimization

Desmond J. Higham

- retweets: 72, favorites: 29 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11369) | [pdf](https://arxiv.org/pdf/2009.11369)
- [math.NA](https://arxiv.org/list/math.NA/recent)

I give a brief, non-technical, historical perspective on numerical analysis and optimization. I also touch on emerging trends and future challenges. This content is based on the short presentation that I made at the opening ceremony of \emph{The International Conference on Numerical Analysis and Optimization}, which was held at Sultan Qaboos University, Muscat, Oman, on January 6--9, 2020. Of course, the material covered here is necessarily incomplete and biased towards my own interests and comfort zones. My aim is to give a feel for how the area has developed over the past few decades and how it may continue.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;A Personal Perspective on Numerical Analysis and Optimization&quot; (by Desmond J. Higham): <a href="https://t.co/T1aSSQvBOp">https://t.co/T1aSSQvBOp</a></p>&mdash; DynamicalSystemsSIAM (@DynamicsSIAM) <a href="https://twitter.com/DynamicsSIAM/status/1309299093802381313?ref_src=twsrc%5Etfw">September 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Structure Aware Negative Sampling in Knowledge Graphs

Kian Ahrabian, Aarash Feizi, Yasmin Salehi, William L. Hamilton, Avishek Joey Bose

- retweets: 54, favorites: 32 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11355) | [pdf](https://arxiv.org/pdf/2009.11355)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Learning low-dimensional representations for entities and relations in knowledge graphs using contrastive estimation represents a scalable and effective method for inferring connectivity patterns. A crucial aspect of contrastive learning approaches is the choice of corruption distribution that generates hard negative samples, which force the embedding model to learn discriminative representations and find critical characteristics of observed data. While earlier methods either employ too simple corruption distributions, i.e. uniform, yielding easy uninformative negatives or sophisticated adversarial distributions with challenging optimization schemes, they do not explicitly incorporate known graph structure resulting in suboptimal negatives. In this paper, we propose Structure Aware Negative Sampling (SANS), an inexpensive negative sampling strategy that utilizes the rich graph structure by selecting negative samples from a node's k-hop neighborhood. Empirically, we demonstrate that SANS finds high-quality negatives that are highly competitive with SOTA methods, and requires no additional parameters nor difficult adversarial optimization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> short paper on &quot;Structure Aware Negative Sampling on Knowledge Graphs&quot; is now available: <a href="https://t.co/4ZrhWVol7S">https://t.co/4ZrhWVol7S</a> <br>This work was led by impressive graduate students: <a href="https://twitter.com/kahrabian?ref_src=twsrc%5Etfw">@kahrabian</a>, <a href="https://twitter.com/aarashfeizi?ref_src=twsrc%5Etfw">@aarashfeizi</a>, <a href="https://twitter.com/SalehiYasmin?ref_src=twsrc%5Etfw">@SalehiYasmin</a>, and also with my amazing supervisor <a href="https://twitter.com/williamleif?ref_src=twsrc%5Etfw">@williamleif</a>.</p>&mdash; Joey Bose (@bose_joey) <a href="https://twitter.com/bose_joey/status/1309312316786307073?ref_src=twsrc%5Etfw">September 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Investigating Applications on the A64FX

Adrian Jackson, Michèle Weiland, Nick Brown, Andrew Turner, Mark Parsons

- retweets: 44, favorites: 15 (09/26/2020 09:23:19)

- links: [abs](https://arxiv.org/abs/2009.11806) | [pdf](https://arxiv.org/pdf/2009.11806)
- [cs.PF](https://arxiv.org/list/cs.PF/recent)

The A64FX processor from Fujitsu, being designed for computational simulation and machine learning applications, has the potential for unprecedented performance in HPC systems. In this paper, we evaluate the A64FX by benchmarking against a range of production HPC platforms that cover a number of processor technologies. We investigate the performance of complex scientific applications across multiple nodes, as well as single node and mini-kernel benchmarks. This paper finds that the performance of the A64FX processor across our chosen benchmarks often significantly exceeds other platforms, even without specific application optimisations for the processor instruction set or hardware. However, this is not true for all the benchmarks we have undertaken. Furthermore, the specific configuration of applications can have an impact on the runtime and performance experienced.



