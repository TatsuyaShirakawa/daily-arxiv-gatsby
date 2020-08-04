---
title: Hot Papers 2020-08-03
date: 2020-08-04T08:08:54.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. Self-supervised learning through the eyes of a child

A. Emin Orhan, Vaibhav V. Gupta, Brenden M. Lake

- retweets: 69, favorites: 333 (08/04/2020 08:08:54)

- links: [abs](https://arxiv.org/abs/2007.16189) | [pdf](https://arxiv.org/pdf/2007.16189)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Within months of birth, children have meaningful expectations about the world around them. How much of this early knowledge can be explained through generic learning mechanisms applied to sensory data, and how much of it requires more substantive innate inductive biases? Addressing this fundamental question in its full generality is currently infeasible, but we can hope to make real progress in more narrowly defined domains, such as the development of high-level visual categories, thanks to improvements in data collecting technology and recent progress in deep learning. In this paper, our goal is to achieve such progress by utilizing modern self-supervised deep learning methods and a recent longitudinal, egocentric video dataset recorded from the perspective of several young children (Sullivan et al., 2020). Our results demonstrate the emergence of powerful, high-level visual representations from developmentally realistic natural videos using generic self-supervised learning objectives.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We train large-scale neural nets &quot;through the eyes&quot; of one baby across 2 years of development. New paper from Emin Orhan shows how high-level visual representations emerge from a subset of one baby&#39;s experience, through only self-supervised learning. <a href="https://t.co/2cBMRZUJS8">https://t.co/2cBMRZUJS8</a> (1/2)</p>&mdash; Brenden Lake (@LakeBrenden) <a href="https://twitter.com/LakeBrenden/status/1290269277140836360?ref_src=twsrc%5Etfw">August 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. BERT Learns (and Teaches) Chemistry

Josh Payne, Mario Srouji, Dian Ang Yap, Vineet Kosaraju

- retweets: 24, favorites: 111 (08/04/2020 08:08:54)

- links: [abs](https://arxiv.org/abs/2007.16012) | [pdf](https://arxiv.org/pdf/2007.16012)
- [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Modern computational organic chemistry is becoming increasingly data-driven. There remain a large number of important unsolved problems in this area such as product prediction given reactants, drug discovery, and metric-optimized molecule synthesis, but efforts to solve these problems using machine learning have also increased in recent years. In this work, we propose the use of attention to study functional groups and other property-impacting molecular substructures from a data-driven perspective, using a transformer-based model (BERT) on datasets of string representations of molecules and analyzing the behavior of its attention heads. We then apply the representations of functional groups and atoms learned by the model to tackle problems of toxicity, solubility, drug-likeness, and synthesis accessibility on smaller datasets using the learned representations as features for graph convolution and attention models on the graph structure of molecules, as well as fine-tuning of BERT. Finally, we propose the use of attention visualization as a helpful tool for chemistry practitioners and students to quickly identify important substructures in various chemical properties.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">BERT Learns (and Teaches) Chemistry<br>pdf: <a href="https://t.co/sWkSoq7vlu">https://t.co/sWkSoq7vlu</a><br>abs: <a href="https://t.co/oOCiDTPJqq">https://t.co/oOCiDTPJqq</a> <a href="https://t.co/zKkxwWUHNm">pic.twitter.com/zKkxwWUHNm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1290091398864220160?ref_src=twsrc%5Etfw">August 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Uncovering the structure of clinical EEG signals with self-supervised  learning

Hubert Banville, Omar Chehab, Aapo Hyvärinen, Denis-Alexander Engemann, Alexandre Gramfort

- retweets: 28, favorites: 100 (08/04/2020 08:08:54)

- links: [abs](https://arxiv.org/abs/2007.16104) | [pdf](https://arxiv.org/pdf/2007.16104)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

Objective. Supervised learning paradigms are often limited by the amount of labeled data that is available. This phenomenon is particularly problematic in clinically-relevant data, such as electroencephalography (EEG), where labeling can be costly in terms of specialized expertise and human processing time. Consequently, deep learning architectures designed to learn on EEG data have yielded relatively shallow models and performances at best similar to those of traditional feature-based approaches. However, in most situations, unlabeled data is available in abundance. By extracting information from this unlabeled data, it might be possible to reach competitive performance with deep neural networks despite limited access to labels. Approach. We investigated self-supervised learning (SSL), a promising technique for discovering structure in unlabeled data, to learn representations of EEG signals. Specifically, we explored two tasks based on temporal context prediction as well as contrastive predictive coding on two clinically-relevant problems: EEG-based sleep staging and pathology detection. We conducted experiments on two large public datasets with thousands of recordings and performed baseline comparisons with purely supervised and hand-engineered approaches. Main results. Linear classifiers trained on SSL-learned features consistently outperformed purely supervised deep neural networks in low-labeled data regimes while reaching competitive performance when all labels were available. Additionally, the embeddings learned with each method revealed clear latent structures related to physiological and clinical phenomena, such as age effects. Significance. We demonstrate the benefit of self-supervised learning approaches on EEG data. Our results suggest that SSL may pave the way to a wider use of deep learning models on EEG data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New <a href="https://twitter.com/hashtag/preprint?src=hash&amp;ref_src=twsrc%5Etfw">#preprint</a> out! Led by <a href="https://twitter.com/hubertjbanville?ref_src=twsrc%5Etfw">@hubertjbanville</a> —joint work with <a href="https://twitter.com/lomarchehab?ref_src=twsrc%5Etfw">@lomarchehab</a> Aapo Hyvärinen &amp; <a href="https://twitter.com/agramfort?ref_src=twsrc%5Etfw">@agramfort</a> <a href="https://t.co/wGr6NQeqfS">https://t.co/wGr6NQeqfS</a> we introduce and benchmark self-supervised <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> architectures to automatically uncover the structure of <a href="https://twitter.com/hashtag/clinical?src=hash&amp;ref_src=twsrc%5Etfw">#clinical</a> <a href="https://twitter.com/hashtag/EEG?src=hash&amp;ref_src=twsrc%5Etfw">#EEG</a> — Stay tuned !!! <a href="https://t.co/pvPuDhGgEb">pic.twitter.com/pvPuDhGgEb</a></p>&mdash; Denis A. Engemann (@dngman) <a href="https://twitter.com/dngman/status/1290180842380120064?ref_src=twsrc%5Etfw">August 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can self-supervised learning be used to learn representations of EEG data? Excited to share our new preprint <a href="https://t.co/rc4OhLvT7U">https://t.co/rc4OhLvT7U</a> - joint work with <a href="https://twitter.com/lomarchehab?ref_src=twsrc%5Etfw">@lomarchehab</a> Aapo Hyvärinen <a href="https://twitter.com/dngman?ref_src=twsrc%5Etfw">@dngman</a> &amp; <a href="https://twitter.com/agramfort?ref_src=twsrc%5Etfw">@agramfort</a>! We used self-supervision to improve classification of clinical EEG. Thread: <a href="https://t.co/WQxRAKoYjD">pic.twitter.com/WQxRAKoYjD</a></p>&mdash; Hubert Banville (@hubertjbanville) <a href="https://twitter.com/hubertjbanville/status/1290366419323019264?ref_src=twsrc%5Etfw">August 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. BasicBlocker: Redesigning ISAs to Eliminate Speculative-Execution  Attacks

Jan Philipp Thoma, Jakob Feldtkeller, Markus Krausz, Tim Güneysu, Daniel J. Bernstein

- retweets: 16, favorites: 36 (08/04/2020 08:08:54)

- links: [abs](https://arxiv.org/abs/2007.15919) | [pdf](https://arxiv.org/pdf/2007.15919)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent)

Recent research has revealed an ever-growing class of microarchitectural attacks that exploit speculative execution, a standard feature in modern processors. Proposed and deployed countermeasures involve a variety of compiler updates, firmware updates, and hardware updates. None of the deployed countermeasures have convincing security arguments, and many of them have already been broken.   The obvious way to simplify the analysis of speculative-execution attacks is to eliminate speculative execution. This is normally dismissed as being unacceptably expensive, but the underlying cost analyses consider only software written for current instruction-set architectures, so they do not rule out the possibility of a new instruction-set architecture providing acceptable performance without speculative execution. A new ISA requires compiler updates and hardware updates, but those are happening in any case.   This paper introduces BasicBlocker, a generic ISA modification that works for all common ISAs and that removes most of the performance benefit of speculative execution. To demonstrate feasibility of BasicBlocker, this paper defines a BBRISC-V variant of the RISC-V ISA, reports implementations of a BBRISC-V soft core and an associated compiler, and presents a performance comparison for a variety of benchmark programs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper &quot;BasicBlocker: Redesigning ISAs to Eliminate Speculative-Execution Attacks&quot; online from <a href="https://twitter.com/sec_janthoma?ref_src=twsrc%5Etfw">@sec_janthoma</a>, Jakob Feldtkeller, Markus Krausz, Tim Güneysu, and me: <a href="https://t.co/WEWFyTLhVw">https://t.co/WEWFyTLhVw</a> Speculative execution is much less important for performance than commonly believed.</p>&mdash; Daniel J. Bernstein (@hashbreaker) <a href="https://twitter.com/hashbreaker/status/1290276576597176320?ref_src=twsrc%5Etfw">August 3, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



