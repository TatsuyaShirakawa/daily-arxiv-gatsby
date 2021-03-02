---
title: Hot Papers 2021-03-01
date: 2021-03-02T09:04:21.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-01"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-01"
socialImage: "/media/flying-marine.jpg"

---

# 1. Convolution-Free Medical Image Segmentation using Transformers

Davood Karimi, Serge Vasylechko, Ali Gholipour

- retweets: 1760, favorites: 240 (03/02/2021 09:04:21)

- links: [abs](https://arxiv.org/abs/2102.13645) | [pdf](https://arxiv.org/pdf/2102.13645)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Like other applications in computer vision, medical image segmentation has been most successfully addressed using deep learning models that rely on the convolution operation as their main building block. Convolutions enjoy important properties such as sparse interactions, weight sharing, and translation equivariance. These properties give convolutional neural networks (CNNs) a strong and useful inductive bias for vision tasks. In this work we show that a different method, based entirely on self-attention between neighboring image patches and without any convolution operations, can achieve competitive or better results. Given a 3D image block, our network divides it into $n^3$ 3D patches, where $n=3 \text{ or } 5$ and computes a 1D embedding for each patch. The network predicts the segmentation map for the center patch of the block based on the self-attention between these patch embeddings. We show that the proposed model can achieve segmentation accuracies that are better than the state of the art CNNs on three datasets. We also propose methods for pre-training this model on large corpora of unlabeled images. Our experiments show that with pre-training the advantage of our proposed network over CNNs can be significant when labeled training data is small.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Convolution-Free Medical Image Segmentation<br>using Transformers<br>pdf: <a href="https://t.co/pqXHoVMlUD">https://t.co/pqXHoVMlUD</a><br>abs: <a href="https://t.co/jS3S3YySMB">https://t.co/jS3S3YySMB</a> <a href="https://t.co/cSAGzdRBVI">pic.twitter.com/cSAGzdRBVI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366201702123200519?ref_src=twsrc%5Etfw">March 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Named Tensor Notation

David Chiang, Alexander M. Rush, Boaz Barak

- retweets: 1305, favorites: 220 (03/02/2021 09:04:21)

- links: [abs](https://arxiv.org/abs/2102.13196) | [pdf](https://arxiv.org/pdf/2102.13196)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We propose a notation for tensors with named axes, which relieves the author, reader, and future implementers from the burden of keeping track of the order of axes and the purpose of each. It also makes it easy to extend operations on low-order tensors to higher order ones (e.g., to extend an operation on images to minibatches of images, or extend the attention mechanism to multiple attention heads). After a brief overview of our notation, we illustrate it through several examples from modern machine learning, from building blocks like attention and convolution to full models like Transformers and LeNet. Finally, we give formal definitions and describe some extensions. Our proposals build on ideas from many previous papers and software libraries. We hope that this document will encourage more authors to use named tensors, resulting in clearer papers and less bug-prone implementations.   The source code for this document can be found at https://github.com/namedtensor/notation/. We invite anyone to make comments on this proposal by submitting issues or pull requests on this repository.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Named Tensor Notation (v1.0 release w/ <a href="https://twitter.com/davidweichiang?ref_src=twsrc%5Etfw">@davidweichiang</a>,  <a href="https://twitter.com/boazbaraktcs?ref_src=twsrc%5Etfw">@boazbaraktcs</a> ) - a &quot;dangerous and irresponsible&quot; proposal for reproducible math in deep learning. <br><br>PDF: <a href="https://t.co/1jvhpG7yCH">https://t.co/1jvhpG7yCH</a> <br>Comments: <a href="https://t.co/76h9E1bUWT">https://t.co/76h9E1bUWT</a><br>Why not Einsum? <a href="https://t.co/St0anL4v74">https://t.co/St0anL4v74</a> <a href="https://t.co/gbdGUBSS0C">pic.twitter.com/gbdGUBSS0C</a></p>&mdash; Sasha Rush (@srush_nlp) <a href="https://twitter.com/srush_nlp/status/1366421347438632960?ref_src=twsrc%5Etfw">March 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Iterative SE(3)-Transformers

Fabian B. Fuchs, Edward Wagstaff, Justas Dauparas, Ingmar Posner

- retweets: 484, favorites: 101 (03/02/2021 09:04:22)

- links: [abs](https://arxiv.org/abs/2102.13419) | [pdf](https://arxiv.org/pdf/2102.13419)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

When manipulating three-dimensional data, it is possible to ensure that rotational and translational symmetries are respected by applying so-called SE(3)-equivariant models. Protein structure prediction is a prominent example of a task which displays these symmetries. Recent work in this area has successfully made use of an SE(3)-equivariant model, applying an iterative SE(3)-equivariant attention mechanism. Motivated by this application, we implement an iterative version of the SE(3)-Transformer, an SE(3)-equivariant attention-based model for graph data. We address the additional complications which arise when applying the SE(3)-Transformer in an iterative fashion, compare the iterative and single-pass versions on a toy problem, and consider why an iterative model may be beneficial in some problem settings. We make the code for our implementation available to the community.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Iterative SE(3)-Transformers<br>pdf: <a href="https://t.co/slOxl8Yi2I">https://t.co/slOxl8Yi2I</a><br>abs: <a href="https://t.co/rSblKEEvFZ">https://t.co/rSblKEEvFZ</a> <a href="https://t.co/xLMitdhSTr">pic.twitter.com/xLMitdhSTr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1366200847139549193?ref_src=twsrc%5Etfw">March 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Swift for TensorFlow: A portable, flexible platform for deep learning

Brennan Saeta, Denys Shabalin, Marc Rasi, Brad Larson, Xihui Wu, Parker Schuh, Michelle Casbon, Daniel Zheng, Saleem Abdulrasool, Aleksandr Efremov, Dave Abrahams, Chris Lattner, Richard Wei

- retweets: 172, favorites: 54 (03/02/2021 09:04:22)

- links: [abs](https://arxiv.org/abs/2102.13243) | [pdf](https://arxiv.org/pdf/2102.13243)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Swift for TensorFlow is a deep learning platform that scales from mobile devices to clusters of hardware accelerators in data centers. It combines a language-integrated automatic differentiation system and multiple Tensor implementations within a modern ahead-of-time compiled language oriented around mutable value semantics. The resulting platform has been validated through use in over 30 deep learning models and has been employed across data center and mobile applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">As promised ~2 weeks ago, some academic papers about <a href="https://twitter.com/hashtag/S4TF?src=hash&amp;ref_src=twsrc%5Etfw">#S4TF</a> are now available! First up is “the overview paper&quot; (<a href="https://t.co/IsqFJAhuBZ">https://t.co/IsqFJAhuBZ</a>); highlights include: (1) a discussion on how mutable value semantics is incredibly powerful (especially for autodiff &amp; hw acclrs), and …</p>&mdash; Brennan Saeta (@bsaeta) <a href="https://twitter.com/bsaeta/status/1366271405285801984?ref_src=twsrc%5Etfw">March 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. RbSyn: Type- and Effect-Guided Program Synthesis

Sankha Narayan Guria, Jeffrey S. Foster, David Van Horn

- retweets: 72, favorites: 52 (03/02/2021 09:04:22)

- links: [abs](https://arxiv.org/abs/2102.13183) | [pdf](https://arxiv.org/pdf/2102.13183)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

In recent years, researchers have explored component-based synthesis, which aims to automatically construct programs that operate by composing calls to existing APIs. However, prior work has not considered efficient synthesis of methods with side effects, e.g., web app methods that update a database. In this paper, we introduce RbSyn, a novel type- and effect-guided synthesis tool for Ruby. An RbSyn synthesis goal is specified as the type for the target method and a series of test cases it must pass. RbSyn works by recursively generating well-typed candidate method bodies whose write effects match the read effects of the test case assertions. After finding a set of candidates that separately satisfy each test, RbSyn synthesizes a solution that branches to execute the correct candidate code under the appropriate conditions. We formalize RbSyn on a core, object-oriented language $\lambda_{syn}$ and describe how the key ideas of the model are scaled-up in our implementation for Ruby. We evaluated RbSyn on 19 benchmarks, 12 of which come from popular, open-source Ruby apps. We found that RbSyn synthesizes correct solutions for all benchmarks, with 15 benchmarks synthesizing in under 9 seconds, while the slowest benchmark takes 83 seconds. Using observed reads to guide synthesize is effective: using type-guidance alone times out on 10 of 12 app benchmarks. We also found that using less precise effect annotations leads to worse synthesis performance. In summary, we believe type- and effect-guided synthesis is an important step forward in synthesis of effectful methods from test cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am super excited to announce that my paper &quot;RbSyn: Type- and Effect-Guided Program Synthesis&quot; with Jeff Foster and <a href="https://twitter.com/lambda_calculus?ref_src=twsrc%5Etfw">@lambda_calculus</a> was conditionally accepted to <a href="https://twitter.com/PLDI?ref_src=twsrc%5Etfw">@PLDI</a> 2021.<br><br>Early preprint: <a href="https://t.co/ERyRvA0SVj">https://t.co/ERyRvA0SVj</a></p>&mdash; Sankha Narayan Guria (@ngsankha) <a href="https://twitter.com/ngsankha/status/1366460331854282753?ref_src=twsrc%5Etfw">March 1, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Evolution of collective fairness in complex networks through  degree-based role assignment

Andreia Sofia Teixeira, Francisco C. Santos, Alexandre P. Francisco, Fernando P. Santos

- retweets: 30, favorites: 28 (03/02/2021 09:04:22)

- links: [abs](https://arxiv.org/abs/2102.13597) | [pdf](https://arxiv.org/pdf/2102.13597)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent)

From social contracts to climate agreements, individuals engage in groups that must collectively reach decisions with varying levels of equality and fairness. These dilemmas also pervade Distributed Artificial Intelligence, in domains such as automated negotiation, conflict resolution or resource allocation. As evidenced by the well-known Ultimatum Game -- where a Proposer has to divide a resource with a Responder -- payoff-maximizing outcomes are frequently at odds with fairness. Eliciting equality in populations of self-regarding agents requires judicious interventions. Here we use knowledge about agents' social networks to implement fairness mechanisms, in the context of Multiplayer Ultimatum Games. We focus on network-based role assignment and show that preferentially attributing the role of Proposer to low-connected nodes increases the fairness levels in a population. We evaluate the effectiveness of low-degree Proposer assignment considering networks with different average connectivity, group sizes, and group voting rules when accepting proposals (e.g. majority or unanimity). We further show that low-degree Proposer assignment is efficient, not only optimizing fairness, but also the average payoff level in the population. Finally, we show that stricter voting rules (i.e., imposing an accepting consensus as requirement for collectives to accept a proposal) attenuates the unfairness that results from situations where high-degree nodes (hubs) are the natural candidates to play as Proposers. Our results suggest new routes to use role assignment and voting mechanisms to prevent unfair behaviors from spreading on complex networks.



