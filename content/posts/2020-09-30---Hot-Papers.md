---
title: Hot Papers 2020-09-30
date: 2020-10-01T09:04:17.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-30"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-30"
socialImage: "/media/flying-marine.jpg"

---

# 1. Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution

Vihang P. Patil, Markus Hofmarcher, Marius-Constantin Dinu, Matthias Dorfer, Patrick M. Blies, Johannes Brandstetter, Jose A. Arjona-Medina, Sepp Hochreiter

- retweets: 496, favorites: 50 (10/01/2020 09:04:17)

- links: [abs](https://arxiv.org/abs/2009.14108) | [pdf](https://arxiv.org/pdf/2009.14108)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Reinforcement Learning algorithms require a large number of samples to solve complex tasks with sparse and delayed rewards. Complex tasks can often be hierarchically decomposed into sub-tasks. A step in the Q-function can be associated with solving a sub-task, where the expectation of the return increases. RUDDER has been introduced to identify these steps and then redistribute reward to them, thus immediately giving reward if sub-tasks are solved. Since the problem of delayed rewards is mitigated, learning is considerably sped up. However, for complex tasks, current exploration strategies as deployed in RUDDER struggle with discovering episodes with high rewards. Therefore, we assume that episodes with high rewards are given as demonstrations and do not have to be discovered by exploration. Typically the number of demonstrations is small and RUDDER's LSTM model as a deep learning method does not learn well. Hence, we introduce Align-RUDDER, which is RUDDER with two major modifications. First, Align-RUDDER assumes that episodes with high rewards are given as demonstrations, replacing RUDDER's safe exploration and lessons replay buffer. Second, we replace RUDDER's LSTM model by a profile model that is obtained from multiple sequence alignment of demonstrations. Profile models can be constructed from as few as two demonstrations as known from bioinformatics. Align-RUDDER inherits the concept of reward redistribution, which considerably reduces the delay of rewards, thus speeding up learning. Align-RUDDER outperforms competitors on complex artificial tasks with delayed reward and few demonstrations. On the MineCraft ObtainDiamond task, Align-RUDDER is able to mine a diamond, though not frequently. Github: https://github.com/ml-jku/align-rudder, YouTube: https://youtu.be/HO-_8ZUl-UY

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We introduce Align-RUDDER, which enables Reinforcement Learning from few demonstrations by <br>reward redistribution via multiple sequence alignment.<br>Paper: <a href="https://t.co/nos4JxAWuZ">https://t.co/nos4JxAWuZ</a><br>Blog post, including a demonstration video of mining a diamond in Minecraft: <a href="https://t.co/xa8WccPBHL">https://t.co/xa8WccPBHL</a></p>&mdash; Vihang Patil (@wehungpatil) <a href="https://twitter.com/wehungpatil/status/1311210392849444869?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Utility is in the Eye of the User: A Critique of NLP Leaderboards

Kawin Ethayarajh, Dan Jurafsky

- retweets: 318, favorites: 123 (10/01/2020 09:04:17)

- links: [abs](https://arxiv.org/abs/2009.13888) | [pdf](https://arxiv.org/pdf/2009.13888)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Benchmarks such as GLUE have helped drive advances in NLP by incentivizing the creation of more accurate models. While this leaderboard paradigm has been remarkably successful, a historical focus on performance-based evaluation has been at the expense of other qualities that the NLP community values in models, such as compactness, fairness, and energy efficiency. In this opinion paper, we study the divergence between what is incentivized by leaderboards and what is useful in practice through the lens of microeconomic theory. We frame both the leaderboard and NLP practitioners as consumers and the benefit they get from a model as its utility to them. With this framing, we formalize how leaderboards -- in their current form -- can be poor proxies for the NLP community at large. For example, a highly inefficient model would provide less utility to practitioners but not to a leaderboard, since it is a cost that only the former must bear. To allow practitioners to better estimate a model's utility to them, we advocate for more transparency on leaderboards, such as the reporting of statistics that are of practical concern (e.g., model size, energy efficiency, and inference latency).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A great opinion piece on <a href="https://twitter.com/hashtag/leaderboardism?src=hash&amp;ref_src=twsrc%5Etfw">#leaderboardism</a> in <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> by <a href="https://twitter.com/ethayarajh?ref_src=twsrc%5Etfw">@ethayarajh</a> and <a href="https://twitter.com/jurafsky?ref_src=twsrc%5Etfw">@jurafsky</a>:<br><br>Title: Utility is in the Eye of the User: A Critique of NLP Leaderboards <br>Preprint: <a href="https://t.co/nk96biljF5">https://t.co/nk96biljF5</a> /1</p>&mdash; Anna Rogers (@annargrs) <a href="https://twitter.com/annargrs/status/1311207137448689664?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A succinct read from <a href="https://twitter.com/ethayarajh?ref_src=twsrc%5Etfw">@ethayarajh</a> (and <a href="https://twitter.com/jurafsky?ref_src=twsrc%5Etfw">@jurafsky</a>) at EMNLP 2020, echoing some of the ideas that folks like <a href="https://twitter.com/tallinzen?ref_src=twsrc%5Etfw">@tallinzen</a>, <a href="https://twitter.com/emilymbender?ref_src=twsrc%5Etfw">@emilymbender</a>, and <a href="https://twitter.com/annargrs?ref_src=twsrc%5Etfw">@annargrs</a> have been bringing up regarding leaderboards in <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a>. <a href="https://t.co/mlMnOlJsVt">https://t.co/mlMnOlJsVt</a></p>&mdash; Rishi Bommasani (@RishiBommasani) <a href="https://twitter.com/RishiBommasani/status/1311112987571228672?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A leaderboard-driven NLP culture has helped create more accurate models, but at what cost?<br><br>Through the lens of microeconomics, our <a href="https://twitter.com/hashtag/EMNLP?src=hash&amp;ref_src=twsrc%5Etfw">#EMNLP</a> paper contrasts what&#39;s incentivized by leaderboards with what&#39;s useful in practice: <a href="https://t.co/9cxB68v91H">https://t.co/9cxB68v91H</a><br><br>w/ <a href="https://twitter.com/jurafsky?ref_src=twsrc%5Etfw">@jurafsky</a> <a href="https://twitter.com/stanfordnlp?ref_src=twsrc%5Etfw">@stanfordnlp</a> <br><br>⬇️1/ <a href="https://t.co/7BUb9rEpMa">pic.twitter.com/7BUb9rEpMa</a></p>&mdash; Kawin Ethayarajh (@ethayarajh) <a href="https://twitter.com/ethayarajh/status/1311399326413922304?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. TinyGAN: Distilling BigGAN for Conditional Image Generation

Ting-Yun Chang, Chi-Jen Lu

- retweets: 182, favorites: 51 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.13829) | [pdf](https://arxiv.org/pdf/2009.13829)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative Adversarial Networks (GANs) have become a powerful approach for generative image modeling. However, GANs are notorious for their training instability, especially on large-scale, complex datasets. While the recent work of BigGAN has significantly improved the quality of image generation on ImageNet, it requires a huge model, making it hard to deploy on resource-constrained devices. To reduce the model size, we propose a black-box knowledge distillation framework for compressing GANs, which highlights a stable and efficient training process. Given BigGAN as the teacher network, we manage to train a much smaller student network to mimic its functionality, achieving competitive performance on Inception and FID scores with the generator having $16\times$ fewer parameters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TinyGAN: Distilling BigGAN for Conditional Image Generation<br>pdf: <a href="https://t.co/Qfa29v8BM6">https://t.co/Qfa29v8BM6</a><br>abs: <a href="https://t.co/26cT7S6T21">https://t.co/26cT7S6T21</a><br>github: <a href="https://t.co/69D1NZbyau">https://t.co/69D1NZbyau</a> <a href="https://t.co/oT7SlayppN">pic.twitter.com/oT7SlayppN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1311120205112803330?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Fast Fréchet Inception Distance

Alexander Mathiasen, Frederik Hvilshøj

- retweets: 169, favorites: 39 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.14075) | [pdf](https://arxiv.org/pdf/2009.14075)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The Fr\'echet Inception Distance (FID) has been used to evaluate thousands of generative models. We present a novel algorithm, FastFID, which allows fast computation and backpropagation for FID. FastFID can efficiently (1) evaluate generative model *during* training and (2) construct adversarial examples for FID.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Fast Fréchet Inception Distance<br>pdf: <a href="https://t.co/3BG0lIo3jm">https://t.co/3BG0lIo3jm</a><br>abs: <a href="https://t.co/f9BpIKgE4v">https://t.co/f9BpIKgE4v</a> <a href="https://t.co/QTPZaVruVc">pic.twitter.com/QTPZaVruVc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1311128961796239365?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Tracking Mixed Bitcoins

Tin Tironsakkul, Manuel Maarek, Andrea Eross, Mike Just

- retweets: 169, favorites: 14 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.14007) | [pdf](https://arxiv.org/pdf/2009.14007)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Mixer services purportedly remove all connections between the input (deposited) Bitcoins and the output (withdrawn) mixed Bitcoins, seemingly rendering taint analysis tracking ineffectual. In this paper, we introduce and explore a novel tracking strategy, called \emph{Address Taint Analysis}, that adapts from existing transaction-based taint analysis techniques for tracking Bitcoins that have passed through a mixer service. We also investigate the potential of combining address taint analysis with address clustering and backward tainting. We further introduce a set of filtering criteria that reduce the number of false-positive results based on the characteristics of withdrawn transactions and evaluate our solution with verifiable mixing transactions of nine mixer services from previous reverse-engineering studies. Our finding shows that it is possible to track the mixed Bitcoins from the deposited Bitcoins using address taint analysis and the number of potential transaction outputs can be significantly reduced with the filtering criteria.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://t.co/rMnBSAdtHN">https://t.co/rMnBSAdtHN</a> &quot;Tracking Mixed Bitcoins&quot; <a href="https://t.co/TRPPblWCa6">pic.twitter.com/TRPPblWCa6</a></p>&mdash; Alexandre Dulaunoy (@adulau) <a href="https://twitter.com/adulau/status/1311319975681028096?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Comparative Study of Deep Learning Loss Functions for Multi-Label  Remote Sensing Image Classification

Hichame Yessou, Gencer Sumbul, Begüm Demir

- retweets: 66, favorites: 17 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.13935) | [pdf](https://arxiv.org/pdf/2009.13935)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper analyzes and compares different deep learning loss functions in the framework of multi-label remote sensing (RS) image scene classification problems. We consider seven loss functions: 1) cross-entropy loss; 2) focal loss; 3) weighted cross-entropy loss; 4) Hamming loss; 5) Huber loss; 6) ranking loss; and 7) sparseMax loss. All the considered loss functions are analyzed for the first time in RS. After a theoretical analysis, an experimental analysis is carried out to compare the considered loss functions in terms of their: 1) overall accuracy; 2) class imbalance awareness (for which the number of samples associated to each class significantly varies); 3) convexibility and differentiability; and 4) learning efficiency (i.e., convergence speed). On the basis of our analysis, some guidelines are derived for a proper selection of a loss function in multi-label RS scene classification problems.




# 7. A Ranking-based, Balanced Loss Function Unifying Classification and  Localisation in Object Detection

Kemal Oksuz, Baris Can Cam, Emre Akbas, Sinan Kalkan

- retweets: 21, favorites: 37 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.13592) | [pdf](https://arxiv.org/pdf/2009.13592)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose average Localization-Recall-Precision (aLRP), a unified, bounded, balanced and ranking-based loss function for both classification and localisation tasks in object detection. aLRP extends the Localization-Recall-Precision (LRP) performance metric (Oksuz et al., 2018) inspired from how Average Precision (AP) Loss extends precision to a ranking-based loss function for classification (Chen et al., 2020). aLRP has the following distinct advantages: (i) aLRP is the first ranking-based loss function for both classification and localisation tasks. (ii) Thanks to using ranking for both tasks, aLRP naturally enforces high-quality localisation for high-precision classification. (iii) aLRP provides provable balance between positives and negatives. (iv) Compared to on average $\sim 6$ hyperparameters in the loss functions of state-of-the-art detectors, aLRP has only one hyperparameter, which we did not tune in practice. On the COCO dataset, aLRP improves its ranking-based predecessor, AP Loss, more than $4$ AP points and outperforms all one-stage detectors. The code is available at: https://github.com/kemaloksuz/aLRPLoss .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! &quot;A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection&quot; by <a href="https://twitter.com/kemaloksz?ref_src=twsrc%5Etfw">@kemaloksz</a>, <a href="https://twitter.com/camcanbaris?ref_src=twsrc%5Etfw">@camcanbaris</a>, <a href="https://twitter.com/eakbas2?ref_src=twsrc%5Etfw">@eakbas2</a> and <a href="https://twitter.com/kalkansinan?ref_src=twsrc%5Etfw">@kalkansinan</a> accepted to <a href="https://twitter.com/hashtag/NeurIPS2020?src=hash&amp;ref_src=twsrc%5Etfw">#NeurIPS2020</a> as spotlight. Paper: <a href="https://t.co/Ha3oGhsXAK">https://t.co/Ha3oGhsXAK</a><br>Code: <a href="https://t.co/SGNSDNIClT">https://t.co/SGNSDNIClT</a> <a href="https://t.co/ix2TQptN5w">pic.twitter.com/ix2TQptN5w</a></p>&mdash; METU ImageLab (@metu_imagelab) <a href="https://twitter.com/metu_imagelab/status/1311267914369183745?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Breaking the Memory Wall for AI Chip with a New Dimension

Eugene Tam, Shenfei Jiang, Paul Duan, Shawn Meng, Yue Pang, Cayden Huang, Yi Han, Jacke Xie, Yuanjun Cui, Jinsong Yu, Minggui Lu

- retweets: 20, favorites: 33 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.13664) | [pdf](https://arxiv.org/pdf/2009.13664)
- [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Recent advancements in deep learning have led to the widespread adoption of artificial intelligence (AI) in applications such as computer vision and natural language processing. As neural networks become deeper and larger, AI modeling demands outstrip the capabilities of conventional chip architectures. Memory bandwidth falls behind processing power. Energy consumption comes to dominate the total cost of ownership. Currently, memory capacity is insufficient to support the most advanced NLP models. In this work, we present a 3D AI chip, called Sunrise, with near-memory computing architecture to address these three challenges. This distributed, near-memory computing architecture allows us to tear down the performance-limiting memory wall with an abundance of data bandwidth. We achieve the same level of energy efficiency on 40nm technology as competing chips on 7nm technology. By moving to similar technologies as other AI chips, we project to achieve more than ten times the energy efficiency, seven times the performance of the current state-of-the-art chips, and twenty times of memory capacity as compared with the best chip in each benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper is presented Sunrise, a near-memory AI computing architecture, implemented in 40nm, which overcome slow DRAM latency and completely replace SRAM with high-capacity DRAM, achieve the same level of energy efficiency than competing chips on 7nm<a href="https://t.co/WdZrd1S9LG">https://t.co/WdZrd1S9LG</a> <a href="https://t.co/ltcsDty0e8">pic.twitter.com/ltcsDty0e8</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1311316544014741504?ref_src=twsrc%5Etfw">September 30, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented  Dialogue

Shikib Mehri, Mihail Eric, Dilek Hakkani-Tur

- retweets: 36, favorites: 14 (10/01/2020 09:04:18)

- links: [abs](https://arxiv.org/abs/2009.13570) | [pdf](https://arxiv.org/pdf/2009.13570)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

A long-standing goal of task-oriented dialogue research is the ability to flexibly adapt dialogue models to new domains. To progress research in this direction, we introduce \textbf{DialoGLUE} (Dialogue Language Understanding Evaluation), a public benchmark consisting of 7 task-oriented dialogue datasets covering 4 distinct natural language understanding tasks, designed to encourage dialogue research in representation-based transfer, domain adaptation, and sample-efficient task learning. We release several strong baseline models, demonstrating performance improvements over a vanilla BERT architecture and state-of-the-art results on 5 out of 7 tasks, by pre-training on a large open-domain dialogue corpus and task-adaptive self-supervised training. Through the DialoGLUE benchmark, the baseline methods, and our evaluation scripts, we hope to facilitate progress towards the goal of developing more general task-oriented dialogue models.



