---
title: Hot Papers 2020-11-11
date: 2020-11-12T09:14:17.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-11"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-11"
socialImage: "/media/flying-marine.jpg"

---

# 1. Principles of Quantum Communication Theory: A Modern Approach

Sumeet Khatri, Mark M. Wilde

- retweets: 1242, favorites: 174 (11/12/2020 09:14:17)

- links: [abs](https://arxiv.org/abs/2011.04672) | [pdf](https://arxiv.org/pdf/2011.04672)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [hep-th](https://arxiv.org/list/hep-th/recent) | [math-ph](https://arxiv.org/list/math-ph/recent)

This is a preliminary version of a book in progress on the theory of quantum communication. We adopt an information-theoretic perspective throughout and give a comprehensive account of fundamental results in quantum communication theory from the past decade (and earlier), with an emphasis on the modern one-shot-to-asymptotic approach that underlies much of today's state-of-the-art research in this field. In Part I, we cover mathematical preliminaries and provide a detailed study of quantum mechanics from an information-theoretic perspective. We also provide an extensive and thorough review of the quantum entropy zoo, and we devote an entire chapter to the study of entanglement measures. Equipped with these essential tools, in Part II we study classical communication (with and without entanglement assistance), entanglement distillation, quantum communication, secret key distillation, and private communication. In Part III, we cover the latest developments in feedback-assisted communication tasks, such as quantum and classical feedback-assisted communication, LOCC-assisted quantum communication, and secret key agreement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m very pleased to announce a preliminary version of a new textbook on quantum information theory. Done in collaboration with <a href="https://twitter.com/markwilde?ref_src=twsrc%5Etfw">@markwilde</a> and available here:<a href="https://t.co/xQqTyb4kaF">https://t.co/xQqTyb4kaF</a><br><br>We&#39;ve been working on this project for the past three years and are open to any feedback!</p>&mdash; Sumeet Khatri (@SumeetKhatri6) <a href="https://twitter.com/SumeetKhatri6/status/1326345589496242176?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. An Attack on InstaHide: Is Private Learning Possible with Instance  Encoding?

Nicholas Carlini, Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Shuang Song, Abhradeep Thakurta, Florian Tramer

- retweets: 1190, favorites: 98 (11/12/2020 09:14:17)

- links: [abs](https://arxiv.org/abs/2011.05315) | [pdf](https://arxiv.org/pdf/2011.05315)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A learning algorithm is private if the produced model does not reveal (too much) about its training set. InstaHide [Huang, Song, Li, Arora, ICML'20] is a recent proposal that claims to preserve privacy by an encoding mechanism that modifies the inputs before being processed by the normal learner.   We present a reconstruction attack on InstaHide that is able to use the encoded images to recover visually recognizable versions of the original images. Our attack is effective and efficient, and empirically breaks InstaHide on CIFAR-10, CIFAR-100, and the recently released InstaHide Challenge.   We further formalize various privacy notions of learning through instance encoding and investigate the possibility of achieving these notions. We prove barriers against achieving (indistinguishability based notions of) privacy through any learning protocol that uses instance encoding.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is a very nice and complete break of InstaHide by some of my Google colleagues and others: <a href="https://t.co/k2vvGNxtVd">https://t.co/k2vvGNxtVd</a><br><br>TL;DR: This scheme does not work, as the &quot;encrypted&quot; images can be largely recovered. ☹️<br><br>1/5 <a href="https://t.co/6bXdRIeK3t">pic.twitter.com/6bXdRIeK3t</a></p>&mdash; Thomas Steinke (@shortstein) <a href="https://twitter.com/shortstein/status/1326550963054735366?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Deep Reinforcement Learning for Navigation in AAA Video Games

Eloi Alonso, Maxim Peter, David Goumard, Joshua Romoff

- retweets: 290, favorites: 75 (11/12/2020 09:14:17)

- links: [abs](https://arxiv.org/abs/2011.04764) | [pdf](https://arxiv.org/pdf/2011.04764)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

In video games, non-player characters (NPCs) are used to enhance the players' experience in a variety of ways, e.g., as enemies, allies, or innocent bystanders. A crucial component of NPCs is navigation, which allows them to move from one point to another on the map. The most popular approach for NPC navigation in the video game industry is to use a navigation mesh (NavMesh), which is a graph representation of the map, with nodes and edges indicating traversable areas. Unfortunately, complex navigation abilities that extend the character's capacity for movement, e.g., grappling hooks, jetpacks, teleportation, or double-jumps, increases the complexity of the NavMesh, making it intractable in many practical scenarios. Game designers are thus constrained to only add abilities that can be handled by a NavMesh if they want to have NPC navigation. As an alternative, we propose to use Deep Reinforcement Learning (Deep RL) to learn how to navigate 3D maps using any navigation ability. We test our approach on complex 3D environments in the Unity game engine that are notably an order of magnitude larger than maps typically used in the Deep RL literature. One of these maps is directly modeled after a Ubisoft AAA game. We find that our approach performs surprisingly well, achieving at least $90\%$ success rate on all tested scenarios. A video of our results is available at https://youtu.be/WFIf9Wwlq8M.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Reinforcement Learning for Navigation in AAA Video Games<br>pdf: <a href="https://t.co/sUY0w5cBPj">https://t.co/sUY0w5cBPj</a><br>abs: <a href="https://t.co/FxFXnberya">https://t.co/FxFXnberya</a> <a href="https://t.co/Fhc0YAteWv">pic.twitter.com/Fhc0YAteWv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1326356125663457282?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Satellite Constellation Internet Affordability and Need

Meredith L. Rawls, Heidi B. Thiemann, Victor Chemin, Lucianne Walkowicz, Mike W. Peel, Yan G. Grange

- retweets: 274, favorites: 40 (11/12/2020 09:14:18)

- links: [abs](https://arxiv.org/abs/2011.05168) | [pdf](https://arxiv.org/pdf/2011.05168)
- [physics.pop-ph](https://arxiv.org/list/physics.pop-ph/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Large satellite constellations in low-Earth orbit seek to be the infrastructure for global broadband Internet and other telecommunication needs. We briefly review the impacts of satellite constellations on astronomy and show that the Internet service offered by these satellites will primarily target populations where it is unaffordable, not needed, or both. The harm done by tens to hundreds of thousands of low-Earth orbit satellites to astronomy, stargazers worldwide, and the environment is not acceptable.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s taken a while but our paper on satellite constellation internet, affordability, and astronomy is now on arXiv! 🛰️<br><br>Check it out here: <a href="https://t.co/2aMeO3mOR8">https://t.co/2aMeO3mOR8</a><br><br>Thanks again to my lovely co-authors <a href="https://twitter.com/merrdiff?ref_src=twsrc%5Etfw">@merrdiff</a> <a href="https://twitter.com/chmn_victor?ref_src=twsrc%5Etfw">@chmn_victor</a> <a href="https://twitter.com/RocketToLulu?ref_src=twsrc%5Etfw">@RocketToLulu</a> <a href="https://twitter.com/mike_peel?ref_src=twsrc%5Etfw">@mike_peel</a> <a href="https://twitter.com/yggrnge?ref_src=twsrc%5Etfw">@yggrnge</a> ✨ <a href="https://t.co/9CuQabJ77r">https://t.co/9CuQabJ77r</a></p>&mdash; Heidi Thiemann (@heidi_teaman) <a href="https://twitter.com/heidi_teaman/status/1326539253266804737?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Decentralized Structural-RNN for Robot Crowd Navigation with Deep  Reinforcement Learning

Shuijing Liu, Peixin Chang, Weihang Liang, Neeloy Chakraborty, Katherine Driggs-Campbell

- retweets: 290, favorites: 9 (11/12/2020 09:14:18)

- links: [abs](https://arxiv.org/abs/2011.04820) | [pdf](https://arxiv.org/pdf/2011.04820)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Safe and efficient navigation through human crowds is an essential capability for mobile robots. Previous work on robot crowd navigation assumes that the dynamics of all agents are known and well-defined. In addition, the performance of previous methods deteriorates in partially observable environments and environments with dense crowds. To tackle these problems, we propose decentralized structural-Recurrent Neural Network (DS-RNN), a novel network that reasons about spatial and temporal relationships for robot decision making in crowd navigation. We train our network with model-free deep reinforcement learning without any expert supervision. We demonstrate that our model outperforms previous methods and successfully transfer the policy learned in the simulator to a real-world TurtleBot 2i.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning. <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/ReinforcementLearning?src=hash&amp;ref_src=twsrc%5Etfw">#ReinforcementLearning</a><a href="https://t.co/8yF42y2AQA">https://t.co/8yF42y2AQA</a> <a href="https://t.co/A2QIVk6cs5">pic.twitter.com/A2QIVk6cs5</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1326669155236515840?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Adversarial Semantic Collisions

Congzheng Song, Alexander M. Rush, Vitaly Shmatikov

- retweets: 90, favorites: 40 (11/12/2020 09:14:18)

- links: [abs](https://arxiv.org/abs/2011.04743) | [pdf](https://arxiv.org/pdf/2011.04743)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

We study semantic collisions: texts that are semantically unrelated but judged as similar by NLP models. We develop gradient-based approaches for generating semantic collisions and demonstrate that state-of-the-art models for many tasks which rely on analyzing the meaning and similarity of texts-- including paraphrase identification, document retrieval, response suggestion, and extractive summarization-- are vulnerable to semantic collisions. For example, given a target query, inserting a crafted collision into an irrelevant document can shift its retrieval rank from 1000 to top 3. We show how to generate semantic collisions that evade perplexity-based filtering and discuss other potential mitigations. Our code is available at https://github.com/csong27/collision-bert.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Adversarial Semantic Collisions (<a href="https://t.co/sl1SgKCdoy">https://t.co/sl1SgKCdoy</a>, w/ Congzheng Song) <br><br>Explores threats to semantic similarity models (retrieval, summary, chat, ...) by an adversary trying to sneak in malicious secret messages.  👿 <br><br>Code: <a href="https://t.co/S3F7hcydAO">https://t.co/S3F7hcydAO</a> <a href="https://t.co/9NzlnNUwxi">pic.twitter.com/9NzlnNUwxi</a></p>&mdash; Sasha Rush (@srush_nlp) <a href="https://twitter.com/srush_nlp/status/1326588644635930626?ref_src=twsrc%5Etfw">November 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



