---
title: Hot Papers 2020-09-08
date: 2020-09-09T09:29:44.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-08"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-08"
socialImage: "/media/flying-marine.jpg"

---

# 1. Measuring Massive Multitask Language Understanding

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt

- retweets: 77, favorites: 344 (09/09/2020 09:29:44)

- links: [abs](https://arxiv.org/abs/2009.03300) | [pdf](https://arxiv.org/pdf/2009.03300)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach human-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How multipurpose is <a href="https://twitter.com/hashtag/GPT3?src=hash&amp;ref_src=twsrc%5Etfw">#GPT3</a>? We gave it questions about elementary math, history, law, and more. We found that GPT-3 is now better than random chance across many tasks, but for all 57 tasks it still has wide room for improvement.<a href="https://t.co/cykeuuQpNo">https://t.co/cykeuuQpNo</a><a href="https://t.co/uVliu9oYv5">https://t.co/uVliu9oYv5</a> <a href="https://t.co/jCqFvdPeSv">pic.twitter.com/jCqFvdPeSv</a></p>&mdash; Dan Hendrycks (@DanHendrycks) <a href="https://twitter.com/DanHendrycks/status/1303332260318457857?ref_src=twsrc%5Etfw">September 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Measuring Massive Multitask Language Understanding<br>pdf: <a href="https://t.co/C9S2bbmcPm">https://t.co/C9S2bbmcPm</a><br>abs: <a href="https://t.co/dHA63hXeCj">https://t.co/dHA63hXeCj</a><br>github: <a href="https://t.co/fjjanMEeKb">https://t.co/fjjanMEeKb</a> <a href="https://t.co/qNtu54SFQs">pic.twitter.com/qNtu54SFQs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1303134344840712193?ref_src=twsrc%5Etfw">September 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Non-exponentially weighted aggregation: regret bounds for unbounded loss  functions

Pierre Alquier

- retweets: 21, favorites: 108 (09/09/2020 09:29:44)

- links: [abs](https://arxiv.org/abs/2009.03017) | [pdf](https://arxiv.org/pdf/2009.03017)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We tackle the problem of online optimization with a general, possibly unbounded, loss function. It is well known that the exponentially weighted aggregation strategy (EWA) leads to a regret in $\sqrt{T}$ after $T$ steps, under the assumption that the loss is bounded. The online gradient algorithm (OGA) has a regret in $\sqrt{T}$ when the loss is convex and Lipschitz. In this paper, we study a generalized aggregation strategy, where the weights do no longer necessarily depend exponentially on the losses. Our strategy can be interpreted as the minimization of the expected losses plus a penalty term. When the penalty term is the Kullback-Leibler divergence, we obtain EWA as a special case, but using alternative divergences lead to a regret bounds for unbounded, not necessarily convex losses. However, the cost is a worst regret bound in some cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint: &quot;Non-exponentially weighted aggregation: regret bounds for unbounded loss functions&quot;. <br><br>- study of (1.4), extends Bayes &amp; exponential weights (1.3),<br>- explicit formula (3.4),<br>- regret analysis, without boundedness of the loss.<a href="https://t.co/aD5DWchfGK">https://t.co/aD5DWchfGK</a> <a href="https://t.co/yBKQqHA8wc">pic.twitter.com/yBKQqHA8wc</a></p>&mdash; Pierre Alquier (@PierreAlquier) <a href="https://twitter.com/PierreAlquier/status/1303133165230288896?ref_src=twsrc%5Etfw">September 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. E-BERT: A Phrase and Product Knowledge Enhanced Language Model for  E-commerce

Denghui Zhang, Zixuan Yuan, Yanchi Liu, Fuzhen Zhuang, Hui Xiong

- retweets: 22, favorites: 78 (09/09/2020 09:29:44)

- links: [abs](https://arxiv.org/abs/2009.02835) | [pdf](https://arxiv.org/pdf/2009.02835)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Pre-trained language models such as BERT have achieved great success in a broad range of natural language processing tasks. However, BERT cannot well support E-commerce related tasks due to the lack of two levels of domain knowledge, i.e., phrase-level and product-level. On one hand, many E-commerce tasks require an accurate understanding of domain phrases, whereas such fine-grained phrase-level knowledge is not explicitly modeled by BERT's training objective. On the other hand, product-level knowledge like product associations can enhance the language modeling of E-commerce, but they are not factual knowledge thus using them indiscriminately may introduce noise. To tackle the problem, we propose a unified pre-training framework, namely, E-BERT. Specifically, to preserve phrase-level knowledge, we introduce Adaptive Hybrid Masking, which allows the model to adaptively switch from learning preliminary word knowledge to learning complex phrases, based on the fitting progress of two modes. To utilize product-level knowledge, we introduce Neighbor Product Reconstruction, which trains E-BERT to predict a product's associated neighbors with a denoising cross attention layer. Our investigation reveals promising results in four downstream tasks, i.e., review-based question answering, aspect extraction, aspect sentiment classification, and product classification.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">E-BERT proposes ideas on how to improve a BERT based language model for e-commerce use cases. <br><br>The proposed method enhances BERT by modeling product &amp; phrase-level knowledge enabling capabilities such as product classification and review-based QA.<a href="https://t.co/sIcFfheTsq">https://t.co/sIcFfheTsq</a> <a href="https://t.co/tTGq929QV4">pic.twitter.com/tTGq929QV4</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1303295506404323328?ref_src=twsrc%5Etfw">September 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Fast simulation of planar Clifford circuits

David Gosset, Daniel Grier, Alex Kerzner, Luke Schaeffer

- retweets: 5, favorites: 50 (09/09/2020 09:29:45)

- links: [abs](https://arxiv.org/abs/2009.03218) | [pdf](https://arxiv.org/pdf/2009.03218)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CC](https://arxiv.org/list/cs.CC/recent)

A general quantum circuit can be simulated in exponential time on a classical computer. If it has a planar layout, then a tensor-network contraction algorithm due to Markov and Shi has a runtime exponential in the square root of its size, or more generally exponential in the treewidth of the underlying graph. Separately, Gottesman and Knill showed that if all gates are restricted to be Clifford, then there is a polynomial time simulation. We combine these two ideas and show that treewidth and planarity can be exploited to improve Clifford circuit simulation. Our main result is a classical algorithm with runtime scaling asymptotically as $ n^{\omega/2}<n^{1.19}$ which samples from the output distribution obtained by measuring all $n$ qubits of a planar graph state in given Pauli bases. Here $\omega$ is the matrix multiplication exponent. We also provide a classical algorithm with the same asymptotic runtime which samples from the output distribution of any constant-depth Clifford circuit in a planar geometry. Our work improves known classical algorithms with cubic runtime. A key ingredient is a mapping which, given a tree decomposition of some graph $G$, produces a Clifford circuit with a structure that mirrors the tree decomposition and which emulates measurement of the quantum graph state corresponding to $G$. We provide a classical simulation of this circuit with the runtime stated above for planar graphs and otherwise $n t^{\omega-1}$ where $t$ is the width of the tree decomposition. The algorithm also incorporates a matrix-multiplication-time version of the Gottesman-Knill simulation of multi-qubit measurement on stabilizer states, which may be of independent interest.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper with Daniel Grier, Alex Kerzner, and Luke Schaeffer gives a fast classical algorithm to simulate multi-qubit measurement on planar graph states, with applications to Clifford circuit simulation.<a href="https://t.co/SlU6qmZjxA">https://t.co/SlU6qmZjxA</a></p>&mdash; David Gosset (@QuantumGosset) <a href="https://twitter.com/QuantumGosset/status/1303136309276487681?ref_src=twsrc%5Etfw">September 8, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



