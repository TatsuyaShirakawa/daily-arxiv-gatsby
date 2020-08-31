---
title: Hot Papers 2020-08-31
date: 2020-09-01T08:26:39.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-31"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-31"
socialImage: "/media/flying-marine.jpg"

---

# 1. On the model-based stochastic value gradient for continuous  reinforcement learning

Brandon Amos, Samuel Stanton, Denis Yarats, Andrew Gordon Wilson

- retweets: 35, favorites: 155 (09/01/2020 08:26:39)

- links: [abs](https://arxiv.org/abs/2008.12775) | [pdf](https://arxiv.org/pdf/2008.12775)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Model-based reinforcement learning approaches add explicit domain knowledge to agents in hopes of improving the sample-efficiency in comparison to model-free agents. However, in practice model-based methods are unable to achieve the same asymptotic performance on challenging continuous control tasks due to the complexity of learning and controlling an explicit world model. In this paper we investigate the stochastic value gradient (SVG), which is a well-known family of methods for controlling continuous systems which includes model-based approaches that distill a model-based value expansion into a model-free policy. We consider a variant of the model-based SVG that scales to larger systems and uses 1) an entropy regularization to help with exploration, 2) a learned deterministic world model to improve the short-horizon value estimate, and 3) a learned model-free value estimate after the model's rollout. This SVG variation captures the model-free soft actor-critic method as an instance when the model rollout horizon is zero, and otherwise uses short-horizon model rollouts to improve the value estimate for the policy update. We surpass the asymptotic performance of other model-based methods on the proprioceptive MuJoCo locomotion tasks from the OpenAI gym, including a humanoid. We notably achieve these results with a simple deterministic world model without requiring an ensemble.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our new paper we scale model-based reinforcement learning to the gym humanoid by using short-horizon model rollouts followed by a learned model-free value estimate.<br><br>Paper: <a href="https://t.co/UytqwsqKdz">https://t.co/UytqwsqKdz</a><br>Videos: <a href="https://t.co/cjZ60UJg6Z">https://t.co/cjZ60UJg6Z</a><br><br>With <a href="https://twitter.com/sam_d_stanton?ref_src=twsrc%5Etfw">@sam_d_stanton</a> <a href="https://twitter.com/denisyarats?ref_src=twsrc%5Etfw">@denisyarats</a> <a href="https://twitter.com/andrewgwils?ref_src=twsrc%5Etfw">@andrewgwils</a> <a href="https://t.co/6RIZ3YuIHh">pic.twitter.com/6RIZ3YuIHh</a></p>&mdash; Brandon Amos (@brandondamos) <a href="https://twitter.com/brandondamos/status/1300450571439144961?ref_src=twsrc%5Etfw">August 31, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Causal blankets: Theory and algorithmic framework

Fernando E. Rosas, Pedro A.M. Mediano, Martin Biehl, Shamil Chandaria, Daniel Polani

- retweets: 23, favorites: 79 (09/01/2020 08:26:39)

- links: [abs](https://arxiv.org/abs/2008.12568) | [pdf](https://arxiv.org/pdf/2008.12568)
- [nlin.AO](https://arxiv.org/list/nlin.AO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent)

We introduce a novel framework to identify perception-action loops (PALOs) directly from data based on the principles of computational mechanics. Our approach is based on the notion of causal blanket, which captures sensory and active variables as dynamical sufficient statistics -- i.e. as the "differences that make a difference." Moreover, our theory provides a broadly applicable procedure to construct PALOs that requires neither a steady-state nor Markovian dynamics. Using our theory, we show that every bipartite stochastic process has a causal blanket, but the extent to which this leads to an effective PALO formulation varies depending on the integrated information of the bipartition.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Preprint time: “Causal Blankets: Theory and algorithmic framework”<a href="https://t.co/DAicqZpNKX">https://t.co/DAicqZpNKX</a><br><br>Imagine having data from two interactive systems, and wondering if it can be described as a perception-action loop? We think it can always be, but depends on its integrated information.</p>&mdash; Fernando Rosas (@_fernando_rosas) <a href="https://twitter.com/_fernando_rosas/status/1300342345657995264?ref_src=twsrc%5Etfw">August 31, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. AllenAct: A Framework for Embodied AI Research

Luca Weihs, Jordi Salvador, Klemen Kotar, Unnat Jain, Kuo-Hao Zeng, Roozbeh Mottaghi, Aniruddha Kembhavi

- retweets: 7, favorites: 61 (09/01/2020 08:26:39)

- links: [abs](https://arxiv.org/abs/2008.12760) | [pdf](https://arxiv.org/pdf/2008.12760)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

The domain of Embodied AI, in which agents learn to complete tasks through interaction with their environment from egocentric observations, has experienced substantial growth with the advent of deep reinforcement learning and increased interest from the computer vision, NLP, and robotics communities. This growth has been facilitated by the creation of a large number of simulated environments (such as AI2-THOR, Habitat and CARLA), tasks (like point navigation, instruction following, and embodied question answering), and associated leaderboards. While this diversity has been beneficial and organic, it has also fragmented the community: a huge amount of effort is required to do something as simple as taking a model trained in one environment and testing it in another. This discourages good science. We introduce AllenAct, a modular and flexible learning framework designed with a focus on the unique requirements of Embodied AI research. AllenAct provides first-class support for a growing collection of embodied environments, tasks and algorithms, provides reproductions of state-of-the-art models and includes extensive documentation, tutorials, start-up code, and pre-trained models. We hope that our framework makes Embodied AI more accessible and encourages new researchers to join this exciting area. The framework can be accessed at: https://allenact.org/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This graph shows the growth of Embodied AI over the past 5 years. We used the data from <a href="https://twitter.com/SemanticScholar?ref_src=twsrc%5Etfw">@SemanticScholar</a> to create it.<br><br>More details here: <a href="https://t.co/zokjsbKHK1">https://t.co/zokjsbKHK1</a><br><br>figure credit: <a href="https://twitter.com/anikembhavi?ref_src=twsrc%5Etfw">@anikembhavi</a> <a href="https://t.co/qeN7pAPsqC">pic.twitter.com/qeN7pAPsqC</a></p>&mdash; Roozbeh Mottaghi (@RoozbehMottaghi) <a href="https://twitter.com/RoozbehMottaghi/status/1300492224572973057?ref_src=twsrc%5Etfw">August 31, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



