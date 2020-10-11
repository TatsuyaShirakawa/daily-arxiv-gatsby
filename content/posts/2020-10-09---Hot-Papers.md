---
title: Hot Papers 2020-10-09
date: 2020-10-11T10:22:29.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-09"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-09"
socialImage: "/media/flying-marine.jpg"

---

# 1. Energy-based Out-of-distribution Detection

Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li

- retweets: 25400, favorites: 0 (10/11/2020 10:22:29)

- links: [abs](https://arxiv.org/abs/2010.03759) | [pdf](https://arxiv.org/pdf/2010.03759)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Determining whether inputs are out-of-distribution (OOD) is an essential building block for safely deploying machine learning models in the open world. However, previous methods relying on the softmax confidence score suffer from overconfident posterior distributions for OOD data. We propose a unified framework for OOD detection that uses an energy score. We show that energy scores better distinguish in- and out-of-distribution samples than the traditional approach using the softmax scores. Unlike softmax confidence scores, energy scores are theoretically aligned with the probability density of the inputs and are less susceptible to the overconfidence issue. Within this framework, energy can be flexibly used as a scoring function for any pre-trained neural classifier as well as a trainable cost function to shape the energy surface explicitly for OOD detection. On a CIFAR-10 pre-trained WideResNet, using the energy score reduces the average FPR (at TPR 95%) by 18.03% compared to the softmax confidence score. With energy-based training, our method outperforms the state-of-the-art on common benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Suffering from overconfident softmax scores? Time to use energy scores!<br> <br>Excited to release our NeurIPS paper on &quot;Energy-based Out-of-distribution Detection&quot;, a theoretically motivated framework for OOD detection. 1/n<br> <br>Paper:  <a href="https://t.co/0DOLbUR8D5">https://t.co/0DOLbUR8D5</a> (w/ code included) <a href="https://t.co/OSwiJlcfPA">pic.twitter.com/OSwiJlcfPA</a></p>&mdash; Sharon Y. Li (@SharonYixuanLi) <a href="https://twitter.com/SharonYixuanLi/status/1314564710164049921?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Large Product Key Memory for Pretrained Language Models

Gyuwan Kim, Tae-Hwan Jung

- retweets: 2872, favorites: 360 (10/11/2020 10:22:30)

- links: [abs](https://arxiv.org/abs/2010.03881) | [pdf](https://arxiv.org/pdf/2010.03881)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Product key memory (PKM) proposed by Lample et al. (2019) enables to improve prediction accuracy by increasing model capacity efficiently with insignificant computational overhead. However, their empirical application is only limited to causal language modeling. Motivated by the recent success of pretrained language models (PLMs), we investigate how to incorporate large PKM into PLMs that can be finetuned for a wide variety of downstream NLP tasks. We define a new memory usage metric, and careful observation using this metric reveals that most memory slots remain outdated during the training of PKM-augmented models. To train better PLMs by tackling this issue, we propose simple but effective solutions: (1) initialization from the model weights pretrained without memory and (2) augmenting PKM by addition rather than replacing a feed-forward network. We verify that both of them are crucial for the pretraining of PKM-augmented PLMs, enhancing memory utilization and downstream performance. Code and pretrained weights are available at https://github.com/clovaai/pkm-transformers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Last year, we showed that you can outperform a 24-layer transformer in language modeling with just 12 layers and 1 Product-key memory layer. <a href="https://t.co/wjZvgBdgbh">https://t.co/wjZvgBdgbh</a> show that these results also transfer to downstream tasks: BERT large performance with a PKM-augmented BERT base! <a href="https://t.co/ORYsJSdJVL">https://t.co/ORYsJSdJVL</a></p>&mdash; Guillaume Lample (@GuillaumeLample) <a href="https://twitter.com/GuillaumeLample/status/1314597157694042113?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. What Can We Do to Improve Peer Review in NLP?

Anna Rogers, Isabelle Augenstein

- retweets: 2758, favorites: 261 (10/11/2020 10:22:30)

- links: [abs](https://arxiv.org/abs/2010.03863) | [pdf](https://arxiv.org/pdf/2010.03863)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Peer review is our best tool for judging the quality of conference submissions, but it is becoming increasingly spurious. We argue that a part of the problem is that the reviewers and area chairs face a poorly defined task forcing apples-to-oranges comparisons. There are several potential ways forward, but the key difficulty is creating the incentives and mechanisms for their consistent implementation in the NLP community.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper📜: What Can We Do to Improve Peer Review in NLP? <a href="https://t.co/GW8pzbIyLv">https://t.co/GW8pzbIyLv</a><br>with <a href="https://twitter.com/IAugenstein?ref_src=twsrc%5Etfw">@IAugenstein</a> <br><br>TLDR: In its current form, peer review is a poorly defined task with apples-to-oranges comparisons and unrealistic expectations. /1 <a href="https://t.co/pETzijb3OX">pic.twitter.com/pETzijb3OX</a></p>&mdash; Anna Rogers (@annargrs) <a href="https://twitter.com/annargrs/status/1314363840256188417?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Online Safety Assurance for Deep Reinforcement Learning

Noga H. Rotman, Michael Schapira, Aviv Tamar

- retweets: 1228, favorites: 18 (10/11/2020 10:22:30)

- links: [abs](https://arxiv.org/abs/2010.03625) | [pdf](https://arxiv.org/pdf/2010.03625)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.NI](https://arxiv.org/list/cs.NI/recent)

Recently, deep learning has been successfully applied to a variety of networking problems. A fundamental challenge is that when the operational environment for a learning-augmented system differs from its training environment, such systems often make badly informed decisions, leading to bad performance. We argue that safely deploying learning-driven systems requires being able to determine, in real time, whether system behavior is coherent, for the purpose of defaulting to a reasonable heuristic when this is not so. We term this the online safety assurance problem (OSAP). We present three approaches to quantifying decision uncertainty that differ in terms of the signal used to infer uncertainty. We illustrate the usefulness of online safety assurance in the context of the proposed deep reinforcement learning (RL) approach to video streaming. While deep RL for video streaming bests other approaches when the operational and training environments match, it is dominated by simple heuristics when the two differ. Our preliminary findings suggest that transitioning to a default policy when decision uncertainty is detected is key to enjoying the performance benefits afforded by leveraging ML without compromising on safety.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Online Safety Assurance for Deep Reinforcement Learning. <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/Coding?src=hash&amp;ref_src=twsrc%5Etfw">#Coding</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/DataScientist?src=hash&amp;ref_src=twsrc%5Etfw">#DataScientist</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a><a href="https://t.co/G8x7OFb3UQ">https://t.co/G8x7OFb3UQ</a> <a href="https://t.co/enz16SJACS">pic.twitter.com/enz16SJACS</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1314708734858403842?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Olympus: a benchmarking framework for noisy optimization and experiment  planning

Florian Häse, Matteo Aldeghi, Riley J. Hickman, Loïc M. Roch, Melodie Christensen, Elena Liles, Jason E. Hein, Alán Aspuru-Guzik

- retweets: 680, favorites: 112 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.04153) | [pdf](https://arxiv.org/pdf/2010.04153)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

Research challenges encountered across science, engineering, and economics can frequently be formulated as optimization tasks. In chemistry and materials science, recent growth in laboratory digitization and automation has sparked interest in optimization-guided autonomous discovery and closed-loop experimentation. Experiment planning strategies based on off-the-shelf optimization algorithms can be employed in fully autonomous research platforms to achieve desired experimentation goals with the minimum number of trials. However, the experiment planning strategy that is most suitable to a scientific discovery task is a priori unknown while rigorous comparisons of different strategies are highly time and resource demanding. As optimization algorithms are typically benchmarked on low-dimensional synthetic functions, it is unclear how their performance would translate to noisy, higher-dimensional experimental tasks encountered in chemistry and materials science. We introduce Olympus, a software package that provides a consistent and easy-to-use framework for benchmarking optimization algorithms against realistic experiments emulated via probabilistic deep-learning models. Olympus includes a collection of experimentally derived benchmark sets from chemistry and materials science and a suite of experiment planning strategies that can be easily accessed via a user-friendly python interface. Furthermore, Olympus facilitates the integration, testing, and sharing of custom algorithms and user-defined datasets. In brief, Olympus mitigates the barriers associated with benchmarking optimization algorithms on realistic experimental scenarios, promoting data sharing and the creation of a standard framework for evaluating the performance of experiment planning strategies

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are ready to share the preprint of <a href="https://twitter.com/hashtag/Olympus?src=hash&amp;ref_src=twsrc%5Etfw">#Olympus</a> a <a href="https://twitter.com/hashtag/machinelearning?src=hash&amp;ref_src=twsrc%5Etfw">#machinelearning</a> software platform for benchmarking optimization algorithms in noisy surfaces. <a href="https://twitter.com/hashtag/matterlab?src=hash&amp;ref_src=twsrc%5Etfw">#matterlab</a> <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a>  <a href="https://twitter.com/chemuoft?ref_src=twsrc%5Etfw">@chemuoft</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@uoft</a> <a href="https://twitter.com/UBC?ref_src=twsrc%5Etfw">@ubc</a> New tool for <a href="https://twitter.com/hashtag/selfdrivinglabs?src=hash&amp;ref_src=twsrc%5Etfw">#selfdrivinglabs</a> <a href="https://t.co/FvzHhTkUOZ">https://t.co/FvzHhTkUOZ</a></p>&mdash; Alan Aspuru-Guzik (@A_Aspuru_Guzik) <a href="https://twitter.com/A_Aspuru_Guzik/status/1314563051580948481?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. DiffTune: Optimizing CPU Simulator Parameters with Learned  Differentiable Surrogates

Alex Renda, Yishen Chen, Charith Mendis, Michael Carbin

- retweets: 361, favorites: 55 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.04017) | [pdf](https://arxiv.org/pdf/2010.04017)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.PL](https://arxiv.org/list/cs.PL/recent)

CPU simulators are useful tools for modeling CPU execution behavior. However, they suffer from inaccuracies due to the cost and complexity of setting their fine-grained parameters, such as the latencies of individual instructions. This complexity arises from the expertise required to design benchmarks and measurement frameworks that can precisely measure the values of parameters at such fine granularity. In some cases, these parameters do not necessarily have a physical realization and are therefore fundamentally approximate, or even unmeasurable.   In this paper we present DiffTune, a system for learning the parameters of x86 basic block CPU simulators from coarse-grained end-to-end measurements. Given a simulator, DiffTune learns its parameters by first replacing the original simulator with a differentiable surrogate, another function that approximates the original function; by making the surrogate differentiable, DiffTune is then able to apply gradient-based optimization techniques even when the original function is non-differentiable, such as is the case with CPU simulators. With this differentiable surrogate, DiffTune then applies gradient-based optimization to produce values of the simulator's parameters that minimize the simulator's error on a dataset of ground truth end-to-end performance measurements. Finally, the learned parameters are plugged back into the original simulator.   DiffTune is able to automatically learn the entire set of microarchitecture-specific parameters within the Intel x86 simulation model of llvm-mca, a basic block CPU simulator based on LLVM's instruction scheduling model. DiffTune's learned parameters lead llvm-mca to an average error that not only matches but lowers that of its original, expert-provided parameter values.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper is presented DiffTune, a system for learning the parameters of x86 basic block CPU simulators from coarse-grained end-to-end measurements, showing be able to learn the entire set of 11265 μarch-specific parameters from scratch in LLVM-MCA.<a href="https://t.co/jIMSXlMDRd">https://t.co/jIMSXlMDRd</a> <a href="https://t.co/nYnl4JQsOp">pic.twitter.com/nYnl4JQsOp</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1314997349412986882?ref_src=twsrc%5Etfw">October 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Maximum Reward Formulation In Reinforcement Learning

Sai Krishna Gottipati, Yashaswi Pathak, Rohan Nuttall, Sahir, Raviteja Chunduru, Ahmed Touati, Sriram Ganapathi Subramanian, Matthew E. Taylor, Sarath Chandar

- retweets: 116, favorites: 87 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.03744) | [pdf](https://arxiv.org/pdf/2010.03744)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Reinforcement learning (RL) algorithms typically deal with maximizing the expected cumulative return (discounted or undiscounted, finite or infinite horizon). However, several crucial applications in the real world, such as drug discovery, do not fit within this framework because an RL agent only needs to identify states (molecules) that achieve the highest reward within a trajectory and does not need to optimize for the expected cumulative return. In this work, we formulate an objective function to maximize the expected maximum reward along a trajectory, derive a novel functional form of the Bellman equation, introduce the corresponding Bellman operators, and provide a proof of convergence. Using this formulation, we achieve state-of-the-art results on the task of molecule generation that mimics a real-world drug discovery pipeline.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new work on maximum reward formulation in <a href="https://twitter.com/hashtag/RL?src=hash&amp;ref_src=twsrc%5Etfw">#RL</a> is out: <a href="https://t.co/xZdjiRR308">https://t.co/xZdjiRR308</a> We formulate the objective function to maximize the expected maximum reward in a trajectory (instead of the traditional expected cumulative return), derive a new functional form of the Bellman 1/n</p>&mdash; Sai Krishna G.V. (@saikrishna_gvs) <a href="https://twitter.com/saikrishna_gvs/status/1314394487184150533?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Flipping the Perspective in Contact Tracing

Po-Shen Loh

- retweets: 132, favorites: 26 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.03806) | [pdf](https://arxiv.org/pdf/2010.03806)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [q-bio.PE](https://arxiv.org/list/q-bio.PE/recent)

Contact tracing has been a widely-discussed technique for controlling COVID-19. The traditional test-trace-isolate-support paradigm focuses on identifying people after they have been exposed to positive individuals, and isolating them to protect others. This article introduces an alternative and complementary approach, which appears to be the first to notify people before exposure happens, in the context of their interaction network, so that they can directly take actions to avoid exposure themselves, without using personally identifiable information. Our system has just become achievable with present technology: for each positive case, do not only notify their direct contacts, but inform thousands of people of how far away they are from the positive case, as measured in network-theoretic distance in their physical relationship network. This fundamentally different approach has already been deployed in a publicly downloadable app. It brings a new tool to bear on the pandemic, powered by network theory. Like a weather satellite providing early warning of incoming hurricanes, it empowers individuals to see transmission approaching from far away, and to directly avoid exposure in the first place. This flipped perspective engages natural self-interested instincts of self-preservation, reducing reliance on altruism. Consequently, our new system could solve the behavior coordination problem which has hampered many other app-based interventions to date. We also provide a heuristic mathematical analysis that shows how our system already achieves critical mass from the user perspective at very low adoption thresholds (likely below 10% in some common types of communities as indicated empirically in the first practical deployment); after that point, the design of our system naturally accelerates further adoption, while also alerting even non-users of the app.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wrote analysis for new &amp; fundamentally different (much more powerful) approach to digital <a href="https://twitter.com/hashtag/ContactTracing?src=hash&amp;ref_src=twsrc%5Etfw">#ContactTracing</a>. <a href="https://twitter.com/hashtag/COVID?src=hash&amp;ref_src=twsrc%5Etfw">#COVID</a> radar: for each case, don&#39;t only tell direct contacts, but anonymously tell everyone how many relationships away they are from it! <a href="https://t.co/8ECHpffRW1">https://t.co/8ECHpffRW1</a><a href="https://twitter.com/hashtag/mathchat?src=hash&amp;ref_src=twsrc%5Etfw">#mathchat</a></p>&mdash; Po-Shen Loh (@PoShenLoh) <a href="https://twitter.com/PoShenLoh/status/1314405666564583429?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Text-based RL Agents with Commonsense Knowledge: New Challenges,  Environments and Baselines

Keerthiram Murugesan, Mattia Atzeni, Pavan Kapanipathi, Pushkar Shukla, Sadhana Kumaravel, Gerald Tesauro, Kartik Talamadupula, Mrinmaya Sachan, Murray Campbell

- retweets: 80, favorites: 68 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.03790) | [pdf](https://arxiv.org/pdf/2010.03790)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Text-based games have emerged as an important test-bed for Reinforcement Learning (RL) research, requiring RL agents to combine grounded language understanding with sequential decision making. In this paper, we examine the problem of infusing RL agents with commonsense knowledge. Such knowledge would allow agents to efficiently act in the world by pruning out implausible actions, and to perform look-ahead planning to determine how current actions might affect future world states. We design a new text-based gaming environment called TextWorld Commonsense (TWC) for training and evaluating RL agents with a specific kind of commonsense knowledge about objects, their attributes, and affordances. We also introduce several baseline RL agents which track the sequential context and dynamically retrieve the relevant commonsense knowledge from ConceptNet. We show that agents which incorporate commonsense knowledge in TWC perform better, while acting more efficiently. We conduct user-studies to estimate human performance on TWC and show that there is ample room for future improvement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Good evening. Did you know that the principal research scientist behind the Deep Blue chess playing agent is now working on Text-Adventure game playing agents? <a href="https://t.co/2dzmHLJ95m">https://t.co/2dzmHLJ95m</a><br><br>Welcome <a href="https://twitter.com/murraycampbell?ref_src=twsrc%5Etfw">@murraycampbell</a> and watch out for Grues! 😄 <a href="https://t.co/Z4d6p7iFNz">pic.twitter.com/Z4d6p7iFNz</a></p>&mdash; Mark O. Riedl (@mark_riedl) <a href="https://twitter.com/mark_riedl/status/1314717021708398593?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. MolDesigner: Interactive Design of Efficacious Drugs with Deep Learning

Kexin Huang, Tianfan Fu, Dawood Khan, Ali Abid, Ali Abdalla, Abubakar Abid, Lucas M. Glass, Marinka Zitnik, Cao Xiao, Jimeng Sun

- retweets: 90, favorites: 52 (10/11/2020 10:22:31)

- links: [abs](https://arxiv.org/abs/2010.03951) | [pdf](https://arxiv.org/pdf/2010.03951)
- [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The efficacy of a drug depends on its binding affinity to the therapeutic target and pharmacokinetics. Deep learning (DL) has demonstrated remarkable progress in predicting drug efficacy. We develop MolDesigner, a human-in-the-loop web user-interface (UI), to assist drug developers leverage DL predictions to design more effective drugs. A developer can draw a drug molecule in the interface. In the backend, more than 17 state-of-the-art DL models generate predictions on important indices that are crucial for a drug's efficacy. Based on these predictions, drug developers can edit the drug molecule and reiterate until satisfaction. MolDesigner can make predictions in real-time with a latency of less than a second.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MolDesigner is in <a href="https://twitter.com/hashtag/NeurIPS2020?src=hash&amp;ref_src=twsrc%5Etfw">#NeurIPS2020</a> Demo! <br><br>-Interactive molecule design with DL, powered by DeepPurpose and <a href="https://twitter.com/GradioML?ref_src=twsrc%5Etfw">@GradioML</a>! <br>-Predict binding affinity and 17 ADMET properties from 50+ DL models! <br>-Less than 1 sec latency! <br><br>Video: <a href="https://t.co/qx3p3hkAkh">https://t.co/qx3p3hkAkh</a><br>Paper: <a href="https://t.co/weUTZKYCT5">https://t.co/weUTZKYCT5</a> <a href="https://t.co/3GTEUUvuRO">pic.twitter.com/3GTEUUvuRO</a></p>&mdash; Kexin Huang (@KexinHuang5) <a href="https://twitter.com/KexinHuang5/status/1314736615957307397?ref_src=twsrc%5Etfw">October 10, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Fast Stencil-Code Computation on a Wafer-Scale Processor

Kamil Rocki, Dirk Van Essendelft, Ilya Sharapov, Robert Schreiber, Michael Morrison, Vladimir Kibardin, Andrey Portnoy, Jean Francois Dietiker, Madhava Syamlal, Michael James

- retweets: 90, favorites: 24 (10/11/2020 10:22:32)

- links: [abs](https://arxiv.org/abs/2010.03660) | [pdf](https://arxiv.org/pdf/2010.03660)
- [cs.DC](https://arxiv.org/list/cs.DC/recent)

The performance of CPU-based and GPU-based systems is often low for PDE codes, where large, sparse, and often structured systems of linear equations must be solved. Iterative solvers are limited by data movement, both between caches and memory and between nodes. Here we describe the solution of such systems of equations on the Cerebras Systems CS-1, a wafer-scale processor that has the memory bandwidth and communication latency to perform well. We achieve 0.86 PFLOPS on a single wafer-scale system for the solution by BiCGStab of a linear system arising from a 7-point finite difference stencil on a 600 X 595 X 1536 mesh, achieving about one third of the machine's peak performance. We explain the system, its architecture and programming, and its performance on this problem and related problems. We discuss issues of memory capacity and floating point precision. We outline plans to extend this work towards full applications.




# 12. Detecting Fine-Grained Cross-Lingual Semantic Divergences without  Supervision by Learning to Rank

Eleftheria Briakou, Marine Carpuat

- retweets: 60, favorites: 27 (10/11/2020 10:22:32)

- links: [abs](https://arxiv.org/abs/2010.03662) | [pdf](https://arxiv.org/pdf/2010.03662)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Detecting fine-grained differences in content conveyed in different languages matters for cross-lingual NLP and multilingual corpora analysis, but it is a challenging machine learning problem since annotation is expensive and hard to scale. This work improves the prediction and annotation of fine-grained semantic divergences. We introduce a training strategy for multilingual BERT models by learning to rank synthetic divergent examples of varying granularity. We evaluate our models on the Rationalized English-French Semantic Divergences, a new dataset released with this work, consisting of English-French sentence-pairs annotated with semantic divergence classes and token-level rationales. Learning to rank helps detect fine-grained sentence-level divergences more accurately than a strong sentence-level similarity model, while token-level predictions have the potential of further distinguishing between coarse and fine-grained divergences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> paper w/ <a href="https://twitter.com/MarineCarpuat?ref_src=twsrc%5Etfw">@MarineCarpuat</a> at <a href="https://twitter.com/umdclip?ref_src=twsrc%5Etfw">@umdclip</a> on &quot;Detecting Fine-grained Cross-lingual Semantic Divergences without supervision by Learning to Rank&quot; is now on arxiv: <a href="https://t.co/bVPjw9UZDq">https://t.co/bVPjw9UZDq</a><br>Code and data available: <a href="https://t.co/PpPxLnqjc5">https://t.co/PpPxLnqjc5</a></p>&mdash; Eleftheria Briakou (@ebriakou) <a href="https://twitter.com/ebriakou/status/1314578548636745738?ref_src=twsrc%5Etfw">October 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Robust Semi-Supervised Learning with Out of Distribution Data

Xujiang Zhao, Killamsetty Krishnateja, Rishabh Iyer, Feng Chen

- retweets: 36, favorites: 16 (10/11/2020 10:22:32)

- links: [abs](https://arxiv.org/abs/2010.03658) | [pdf](https://arxiv.org/pdf/2010.03658)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Semi-supervised learning (SSL) based on deep neural networks (DNNs) has recently been proven effective. However, recent work [Oliver et al., 2018] shows that the performance of SSL could degrade substantially when the unlabeled set has out-of-distribution examples (OODs). In this work, we first study the key causes about the negative impact of OOD on SSL. We found that (1) OODs close to the decision boundary have a larger effect on the performance of existing SSL algorithms than the OODs far away from the decision boundary and (2) Batch Normalization (BN), a popular module in deep networks, could degrade the performance of a DNN for SSL substantially when the unlabeled set contains OODs. To address these causes, we proposed a novel unified robust SSL approach for many existing SSL algorithms in order to improve their robustness against OODs. In particular, we proposed a simple modification to batch normalization, called weighted batch normalization, capable of improving the robustness of BN against OODs. We developed two efficient hyperparameter optimization algorithms that have different tradeoffs in computational efficiency and accuracy. The first is meta-approximation and the second is implicit-differentiation based approximation. Both algorithms learn to reweight the unlabeled samples in order to improve the robustness of SSL against OODs. Extensive experiments on both synthetic and real-world datasets demonstrate that our proposed approach significantly improves the robustness of four representative SSL algorithms against OODs, in comparison with four state-of-the-art robust SSL approaches. We performed an ablation study to demonstrate which components of our approach are most important for its success.



