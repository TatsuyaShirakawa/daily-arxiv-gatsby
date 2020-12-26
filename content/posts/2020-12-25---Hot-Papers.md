---
title: Hot Papers 2020-12-25
date: 2020-12-26T16:50:40.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. Solving Mixed Integer Programs Using Neural Networks

Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O'Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, Ravichandra Addanki, Tharindi Hapuarachchi, Thomas Keck, James Keeling, Pushmeet Kohli, Ira Ktena, Yujia Li, Oriol Vinyals, Yori Zwols

- retweets: 12223, favorites: 47 (12/26/2020 16:50:40)

- links: [abs](https://arxiv.org/abs/2012.13349) | [pdf](https://arxiv.org/pdf/2012.13349)
- [math.OC](https://arxiv.org/list/math.OC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DM](https://arxiv.org/list/cs.DM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Mixed Integer Programming (MIP) solvers rely on an array of sophisticated heuristics developed with decades of research to solve large-scale MIP instances encountered in practice. Machine learning offers to automatically construct better heuristics from data by exploiting shared structure among instances in the data. This paper applies learning to the two key sub-tasks of a MIP solver, generating a high-quality joint variable assignment, and bounding the gap in objective value between that assignment and an optimal one. Our approach constructs two corresponding neural network-based components, Neural Diving and Neural Branching, to use in a base MIP solver such as SCIP. Neural Diving learns a deep neural network to generate multiple partial assignments for its integer variables, and the resulting smaller MIPs for un-assigned variables are solved with SCIP to construct high quality joint assignments. Neural Branching learns a deep neural network to make variable selection decisions in branch-and-bound to bound the objective value gap with a small tree. This is done by imitating a new variant of Full Strong Branching we propose that scales to large instances using GPUs. We evaluate our approach on six diverse real-world datasets, including two Google production datasets and MIPLIB, by training separate neural networks on each. Most instances in all the datasets combined have $10^3-10^6$ variables and constraints after presolve, which is significantly larger than previous learning approaches. Comparing solvers with respect to primal-dual gap averaged over a held-out set of instances, the learning-augmented SCIP is 2x to 10x better on all datasets except one on which it is $10^5$x better, at large time limits. To the best of our knowledge, ours is the first learning approach to demonstrate such large improvements over SCIP on both large-scale real-world application datasets and MIPLIB.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Solving Mixed Integer Programs Using Neural Networks<br><br>The first learning-based method to substantially outperform SCIP (a mixed interger program solver) on various large-scale real-world application datasets.<a href="https://t.co/ixIZQS1UKC">https://t.co/ixIZQS1UKC</a> <a href="https://t.co/pBxe8UQmL1">pic.twitter.com/pBxe8UQmL1</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1342293320412762112?ref_src=twsrc%5Etfw">December 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Soft-IntroVAE: Analyzing and Improving the Introspective Variational  Autoencoder

Tal Daniel, Aviv Tamar

- retweets: 504, favorites: 112 (12/26/2020 16:50:40)

- links: [abs](https://arxiv.org/abs/2012.13253) | [pdf](https://arxiv.org/pdf/2012.13253)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

The recently introduced introspective variational autoencoder (IntroVAE) exhibits outstanding image generations, and allows for amortized inference using an image encoder. The main idea in IntroVAE is to train a VAE adversarially, using the VAE encoder to discriminate between generated and real data samples. However, the original IntroVAE loss function relied on a particular hinge-loss formulation that is very hard to stabilize in practice, and its theoretical convergence analysis ignored important terms in the loss. In this work, we take a step towards better understanding of the IntroVAE model, its practical implementation, and its applications. We propose the Soft-IntroVAE, a modified IntroVAE that replaces the hinge-loss terms with a smooth exponential loss on generated samples. This change significantly improves training stability, and also enables theoretical analysis of the complete algorithm. Interestingly, we show that the IntroVAE converges to a distribution that minimizes a sum of KL distance from the data distribution and an entropy term. We discuss the implications of this result, and demonstrate that it induces competitive image generation and reconstruction. Finally, we describe two applications of Soft-IntroVAE to unsupervised image translation and out-of-distribution detection, and demonstrate compelling results. Code and additional information is available on the project website -- https://taldatech.github.io/soft-intro-vae-web

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Soft-IntroVAE: Analyzing and Improving the Introspective Variational Autoencoder<br>pdf: <a href="https://t.co/ptyRmVE5JP">https://t.co/ptyRmVE5JP</a><br>abs: <a href="https://t.co/uJ8jMTc8dH">https://t.co/uJ8jMTc8dH</a><br>project page: <a href="https://t.co/U62Bd30R5X">https://t.co/U62Bd30R5X</a><br>github: <a href="https://t.co/FkywlKgpfz">https://t.co/FkywlKgpfz</a> <a href="https://t.co/Zt5qA48xGr">pic.twitter.com/Zt5qA48xGr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1342297437373935618?ref_src=twsrc%5Etfw">December 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Nine Best Practices for Research Software Registries and Repositories: A  Concise Guide

Task Force on Best Practices for Software Registries, Alain Monteil, Alejandra Gonzalez-Beltran, Alexandros Ioannidis, Alice Allen, Allen Lee, Anita Bandrowski, Bruce E. Wilson, Bryce Mecum, Cai Fan Du, Carly Robinson, Daniel Garijo, Daniel S. Katz, David Long, Genevieve Milliken, Hervé Ménager, Jessica Hausman

- retweets: 182, favorites: 28 (12/26/2020 16:50:40)

- links: [abs](https://arxiv.org/abs/2012.13117) | [pdf](https://arxiv.org/pdf/2012.13117)
- [cs.DL](https://arxiv.org/list/cs.DL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Scientific software registries and repositories serve various roles in their respective disciplines. These resources improve software discoverability and research transparency, provide information for software citations, and foster preservation of computational methods that might otherwise be lost over time, thereby supporting research reproducibility and replicability. However, developing these resources takes effort, and few guidelines are available to help prospective creators of registries and repositories. To address this need, we present a set of nine best practices that can help managers define the scope, practices, and rules that govern individual registries and repositories. These best practices were distilled from the experiences of the creators of existing resources, convened by a Task Force of the FORCE11 Software Citation Implementation Working Group during the years 2019-2020. We believe that putting in place specific policies such as those presented here will help scientific software registries and repositories better serve their users and their disciplines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m happy to see this out:<br><br>Nine Best Practices for Research Software Registries and Repositories: A Concise Guide<br><br>by a <a href="https://twitter.com/force11rescomm?ref_src=twsrc%5Etfw">@force11rescomm</a> Software Citation Impl WG task force led by <a href="https://twitter.com/owlice?ref_src=twsrc%5Etfw">@owlice</a> <a href="https://twitter.com/mhucka?ref_src=twsrc%5Etfw">@mhucka</a> <a href="https://twitter.com/temorrell?ref_src=twsrc%5Etfw">@temorrell</a> <a href="https://t.co/Ox38AiULaY">https://t.co/Ox38AiULaY</a><br><br>(thx <a href="https://twitter.com/SloanFoundation?ref_src=twsrc%5Etfw">@SloanFoundation</a> &amp; <a href="https://twitter.com/epistemographer?ref_src=twsrc%5Etfw">@epistemographer</a>) <a href="https://t.co/ALg0WbS4FJ">pic.twitter.com/ALg0WbS4FJ</a></p>&mdash; Daniel S. Katz (@danielskatz) <a href="https://twitter.com/danielskatz/status/1342482159832080385?ref_src=twsrc%5Etfw">December 25, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MobileSal: Extremely Efficient RGB-D Salient Object Detection

Yu-Huan Wu, Yun Liu, Jun Xu, Jia-Wang Bian, Yuchao Gu, Ming-Ming Cheng

- retweets: 49, favorites: 14 (12/26/2020 16:50:40)

- links: [abs](https://arxiv.org/abs/2012.13095) | [pdf](https://arxiv.org/pdf/2012.13095)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The high computational cost of neural networks has prevented recent successes in RGB-D salient object detection (SOD) from benefiting real-world applications. Hence, this paper introduces a novel network, \methodname, which focuses on efficient RGB-D SOD by using mobile networks for deep feature extraction. The problem is that mobile networks are less powerful in feature representation than cumbersome networks. To this end, we observe that the depth information of color images can strengthen the feature representation related to SOD if leveraged properly. Therefore, we propose an implicit depth restoration (IDR) technique to strengthen the feature representation capability of mobile networks for RGB-D SOD. IDR is only adopted in the training phase and is omitted during testing, so it is computationally free. Besides, we propose compact pyramid refinement (CPR) for efficient multi-level feature aggregation so that we can derive salient objects with clear boundaries. With IDR and CPR incorporated, \methodname~performs favorably against \sArt methods on seven challenging RGB-D SOD datasets with much faster speed (450fps) and fewer parameters (6.5M). The code will be released.




# 5. Concurrency measures in the era of temporal network epidemiology: A  review

Naoki Masuda, Joel C. Miller, Petter Holme

- retweets: 30, favorites: 31 (12/26/2020 16:50:40)

- links: [abs](https://arxiv.org/abs/2012.13317) | [pdf](https://arxiv.org/pdf/2012.13317)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Diseases spread over temporal networks of contacts between individuals. Structures of these temporal networks hold the keys to understanding epidemic propagation. One early concept of the literature to aid in discussing these structures is concurrency -- quantifying individuals' tendency to form time-overlapping "partnerships". Although conflicting evaluations and an overabundance of operational definitions have marred the history of concurrency, it remains important, especially in the area of sexually transmitted infections. Today, much of theoretical epidemiology uses more direct models of contact patterns, and there is an emerging body of literature trying to connect methods to the concurrency literature. In this review, we will cover the development of the concept of concurrency and these new approaches.



