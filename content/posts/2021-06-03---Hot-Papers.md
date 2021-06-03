---
title: Hot Papers 2021-06-03
date: 2021-06-04T06:43:24.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-03"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-03"
socialImage: "/media/flying-marine.jpg"

---

# 1. Decision Transformer: Reinforcement Learning via Sequence Modeling

Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch

- retweets: 5626, favorites: 368 (06/04/2021 06:43:24)

- links: [abs](https://arxiv.org/abs/2106.01345) | [pdf](https://arxiv.org/pdf/2106.01345)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Decision Transformer: Reinforcement Learning via Sequence Modeling<br><br>A nice result in the paper: By training a language model on a training dataset of random walk trajectories, it can figure out optimal trajectories by just conditioning on a large reward. <a href="https://t.co/XnkZG4eiIU">https://t.co/XnkZG4eiIU</a> <a href="https://t.co/MeBT1LbCTh">https://t.co/MeBT1LbCTh</a> <a href="https://t.co/eIc9sDtvTL">pic.twitter.com/eIc9sDtvTL</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1400281766254956544?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. SAINT: Improved Neural Networks for Tabular Data via Row Attention and  Contrastive Pre-Training

Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, Tom Goldstein

- retweets: 1560, favorites: 187 (06/04/2021 06:43:24)

- links: [abs](https://arxiv.org/abs/2106.01342) | [pdf](https://arxiv.org/pdf/2106.01342)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Tabular data underpins numerous high-impact applications of machine learning from fraud detection to genomics and healthcare. Classical approaches to solving tabular problems, such as gradient boosting and random forests, are widely used by practitioners. However, recent deep learning methods have achieved a degree of performance competitive with popular techniques. We devise a hybrid deep learning approach to solving tabular data problems. Our method, SAINT, performs attention over both rows and columns, and it includes an enhanced embedding method. We also study a new contrastive self-supervised pre-training method for use when labels are scarce. SAINT consistently improves performance over previous deep learning methods, and it even outperforms gradient boosting methods, including XGBoost, CatBoost, and LightGBM, on average over a variety of benchmark tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training<br>pdf: <a href="https://t.co/FHJYzwavHr">https://t.co/FHJYzwavHr</a><br>abs: <a href="https://t.co/dHlA7M0nnT">https://t.co/dHlA7M0nnT</a><br><br>performs attention over both rows and columns, and it includes an enhanced embedding method <a href="https://t.co/mJtDHFjuhp">pic.twitter.com/mJtDHFjuhp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400259639120437248?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Towards Deeper Deep Reinforcement Learning

Johan Bjorck, Carla P. Gomes, Kilian Q. Weinberger

- retweets: 1507, favorites: 236 (06/04/2021 06:43:24)

- links: [abs](https://arxiv.org/abs/2106.01151) | [pdf](https://arxiv.org/pdf/2106.01151)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

In computer vision and natural language processing, innovations in model architecture that lead to increases in model capacity have reliably translated into gains in performance. In stark contrast with this trend, state-of-the-art reinforcement learning (RL) algorithms often use only small MLPs, and gains in performance typically originate from algorithmic innovations. It is natural to hypothesize that small datasets in RL necessitate simple models to avoid overfitting; however, this hypothesis is untested. In this paper we investigate how RL agents are affected by exchanging the small MLPs with larger modern networks with skip connections and normalization, focusing specifically on soft actor-critic (SAC) algorithms. We verify, empirically, that na\"ively adopting such architectures leads to instabilities and poor performance, likely contributing to the popularity of simple models in practice. However, we show that dataset size is not the limiting factor, and instead argue that intrinsic instability from the actor in SAC taking gradients through the critic is the culprit. We demonstrate that a simple smoothing method can mitigate this issue, which enables stable training with large modern architectures. After smoothing, larger models yield dramatic performance improvements for state-of-the-art agents -- suggesting that more "easy" gains may be had by focusing on model architectures in addition to algorithmic innovations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Deeper Deep Reinforcement Learning<br><br>Proposes a simple smoothing method that allows RL model to increase its model size without instability, which leads to dramatic perf improvements on SotA angets.<a href="https://t.co/uQs5t72Yg5">https://t.co/uQs5t72Yg5</a> <a href="https://t.co/DA2ea29yuE">pic.twitter.com/DA2ea29yuE</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1400253770697613313?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Deeper Deep Reinforcement Learning<br>pdf: <a href="https://t.co/xJHscJG62J">https://t.co/xJHscJG62J</a><br>abs: <a href="https://t.co/zwERkwjxjC">https://t.co/zwERkwjxjC</a> <a href="https://t.co/d2M0PY6hM0">pic.twitter.com/d2M0PY6hM0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400251926529335298?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Implicit Representations of Meaning in Neural Language Models

Belinda Z. Li, Maxwell Nye, Jacob Andreas

- retweets: 516, favorites: 164 (06/04/2021 06:43:25)

- links: [abs](https://arxiv.org/abs/2106.00737) | [pdf](https://arxiv.org/pdf/2106.00737)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Does the effectiveness of neural language models derive entirely from accurate modeling of surface word co-occurrence statistics, or do these models represent and reason about the world they describe? In BART and T5 transformer language models, we identify contextual word representations that function as models of entities and situations as they evolve throughout a discourse. These neural representations have functional similarities to linguistic models of dynamic semantics: they support a linear readout of each entity's current properties and relations, and can be manipulated with predictable effects on language generation. Our results indicate that prediction in pretrained neural language models is supported, at least in part, by dynamic representations of meaning and implicit simulation of entity state, and that this behavior can be learned with only text as training data. Code and data are available at https://github.com/belindal/state-probes .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do neural language models (trained on text alone!) construct representations of meaning? In a new <a href="https://twitter.com/hashtag/ACL2021NLP?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021NLP</a> paper, we find that LM representations implicitly model *entities and situations* as they evolve through a discourse. 1/<a href="https://t.co/Nz4C5m0i1F">https://t.co/Nz4C5m0i1F</a> <a href="https://t.co/QmmCK6uuus">pic.twitter.com/QmmCK6uuus</a></p>&mdash; Belinda Li (@belindazli) <a href="https://twitter.com/belindazli/status/1400516024437411843?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Implicit Representations of Meaning in Neural Language Models<br>pdf: <a href="https://t.co/6p3jgmE4aE">https://t.co/6p3jgmE4aE</a><br>abs: <a href="https://t.co/irfpGiOJnJ">https://t.co/irfpGiOJnJ</a><br>github: <a href="https://t.co/M6aC9T0Vf3">https://t.co/M6aC9T0Vf3</a> <a href="https://t.co/KuR5AJkFaQ">pic.twitter.com/KuR5AJkFaQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400267071318003714?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Efficient Passage Retrieval with Hashing for Open-domain Question  Answering

Ikuya Yamada, Akari Asai, Hannaneh Hajishirzi

- retweets: 468, favorites: 111 (06/04/2021 06:43:25)

- links: [abs](https://arxiv.org/abs/2106.00882) | [pdf](https://arxiv.org/pdf/2106.00882)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Most state-of-the-art open-domain question answering systems use a neural retrieval model to encode passages into continuous vectors and extract them from a knowledge source. However, such retrieval models often require large memory to run because of the massive size of their passage index. In this paper, we introduce Binary Passage Retriever (BPR), a memory-efficient neural retrieval model that integrates a learning-to-hash technique into the state-of-the-art Dense Passage Retriever (DPR) to represent the passage index using compact binary codes rather than continuous vectors. BPR is trained with a multi-task objective over two tasks: efficient candidate generation based on binary codes and accurate reranking based on continuous vectors. Compared with DPR, BPR substantially reduces the memory cost from 65GB to 2GB without a loss of accuracy on two standard open-domain question answering benchmarks: Natural Questions and TriviaQA. Our code and trained models are available at https://github.com/studio-ousia/bpr.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚀Neural passage retrieval with substantially reduced memory size🚀<br><br>BPR presented in our <a href="https://twitter.com/hashtag/acl2021nlp?src=hash&amp;ref_src=twsrc%5Etfw">#acl2021nlp</a> paper drastically reduces the memory size of the SOTA retriever (DPR) without a loss of QA accuracy<br><br>Paper: <a href="https://t.co/PjUZNmVGMV">https://t.co/PjUZNmVGMV</a><br>Code/Model: <a href="https://t.co/JqPOzQlapk">https://t.co/JqPOzQlapk</a><br><br>👇Threads <a href="https://t.co/GYwJgw5V3V">pic.twitter.com/GYwJgw5V3V</a></p>&mdash; Ikuya Yamada (@ikuyamada) <a href="https://twitter.com/ikuyamada/status/1400337133106130945?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Generalizable Approach to Learning Optimizers

Diogo Almeida, Clemens Winter, Jie Tang, Wojciech Zaremba

- retweets: 340, favorites: 165 (06/04/2021 06:43:25)

- links: [abs](https://arxiv.org/abs/2106.00958) | [pdf](https://arxiv.org/pdf/2106.00958)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A core issue with learning to optimize neural networks has been the lack of generalization to real world problems. To address this, we describe a system designed from a generalization-first perspective, learning to update optimizer hyperparameters instead of model parameters directly using novel features, actions, and a reward function. This system outperforms Adam at all neural network tasks including on modalities not seen during training. We achieve 2x speedups on ImageNet, and a 2.5x speedup on a language modeling task using over 5 orders of magnitude more compute than the training tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generalizable Approach to Learning Optimizers<br><br>Proposes a system that learns to update optimizer<br>hparams directly using novel features, etc. <br><br>2x speedups on ImageNet, 2.5x on a LM task w/ &gt; 5 orders of magnitude more compute than the training tasks<a href="https://t.co/NLU3jWG3Jt">https://t.co/NLU3jWG3Jt</a> <a href="https://t.co/4iH32IeQaM">pic.twitter.com/4iH32IeQaM</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1400255916902010883?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Generalizable Approach to Learning Optimizers<br>pdf: <a href="https://t.co/mkkARTAypW">https://t.co/mkkARTAypW</a><br>abs: <a href="https://t.co/nsoAtnPijs">https://t.co/nsoAtnPijs</a><br><br>outperforms Adam at all nn tasks including on modalities not seen during training. 2x speedups on ImageNet, and a 2.5x speedup on a language modeling task <a href="https://t.co/mPsRyX88Dw">pic.twitter.com/mPsRyX88Dw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400257157325594626?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Fourier Space Losses for Efficient Perceptual Image Super-Resolution

Dario Fuoli, Luc Van Gool, Radu Timofte

- retweets: 323, favorites: 104 (06/04/2021 06:43:25)

- links: [abs](https://arxiv.org/abs/2106.00783) | [pdf](https://arxiv.org/pdf/2106.00783)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Many super-resolution (SR) models are optimized for high performance only and therefore lack efficiency due to large model complexity. As large models are often not practical in real-world applications, we investigate and propose novel loss functions, to enable SR with high perceptual quality from much more efficient models. The representative power for a given low-complexity generator network can only be fully leveraged by strong guidance towards the optimal set of parameters. We show that it is possible to improve the performance of a recently introduced efficient generator architecture solely with the application of our proposed loss functions. In particular, we use a Fourier space supervision loss for improved restoration of missing high-frequency (HF) content from the ground truth image and design a discriminator architecture working directly in the Fourier domain to better match the target HF distribution. We show that our losses' direct emphasis on the frequencies in Fourier-space significantly boosts the perceptual image quality, while at the same time retaining high restoration quality in comparison to previously proposed loss functions for this task. The performance is further improved by utilizing a combination of spatial and frequency domain losses, as both representations provide complementary information during training. On top of that, the trained generator achieves comparable results with and is 2.4x and 48x faster than state-of-the-art perceptual SR methods RankSRGAN and SRFlow respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fourier Space Losses for Efficient Perceptual Image Super-Resolution<br>pdf: <a href="https://t.co/80ofHmxtzP">https://t.co/80ofHmxtzP</a><br>abs: <a href="https://t.co/avRJqpXENF">https://t.co/avRJqpXENF</a><br><br>two Fourier domain losses – a supervision and a GAN loss – to strengthen the training signal for the task of perceptual image SR <a href="https://t.co/RGcvFDZwX9">pic.twitter.com/RGcvFDZwX9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400272421396090882?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. What Ingredients Make for an Effective Crowdsourcing Protocol for  Difficult NLU Data Collection Tasks?

Nikita Nangia, Saku Sugawara, Harsh Trivedi, Alex Warstadt, Clara Vania, Samuel R. Bowman

- retweets: 346, favorites: 73 (06/04/2021 06:43:25)

- links: [abs](https://arxiv.org/abs/2106.00794) | [pdf](https://arxiv.org/pdf/2106.00794)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

Crowdsourcing is widely used to create data for common natural language understanding tasks. Despite the importance of these datasets for measuring and refining model understanding of language, there has been little focus on the crowdsourcing methods used for collecting the datasets. In this paper, we compare the efficacy of interventions that have been proposed in prior work as ways of improving data quality. We use multiple-choice question answering as a testbed and run a randomized trial by assigning crowdworkers to write questions under one of four different data collection protocols. We find that asking workers to write explanations for their examples is an ineffective stand-alone strategy for boosting NLU example difficulty. However, we find that training crowdworkers, and then using an iterative process of collecting data, sending feedback, and qualifying workers based on expert judgments is an effective means of collecting challenging data. But using crowdsourced, instead of expert judgments, to qualify workers and send feedback does not prove to be effective. We observe that the data from the iterative protocol with expert assessments is more challenging by several measures. Notably, the human--model gap on the unanimous agreement portion of this data is, on average, twice as large as the gap for the baseline protocol data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">👋 Hello <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> friends! Ever wondered What Ingredients Make for an Effective Crowdsourcing Protocol for Difficult NLU Data Collection Tasks? 🧂🧄🥕<br><br>Well, we’ve got an <a href="https://twitter.com/hashtag/ACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021</a> paper just for you: <a href="https://t.co/RJPBVUqqZM">https://t.co/RJPBVUqqZM</a>  1/7 <a href="https://t.co/ypeO65sFDW">pic.twitter.com/ypeO65sFDW</a></p>&mdash; Nikita Nangia (@meloncholist) <a href="https://twitter.com/meloncholist/status/1400481062644506626?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. On the Efficacy of Adversarial Data Collection for Question Answering:  Results from a Large-Scale Randomized Study

Divyansh Kaushik, Douwe Kiela, Zachary C. Lipton, Wen-tau Yih

- retweets: 324, favorites: 89 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.00872) | [pdf](https://arxiv.org/pdf/2106.00872)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In adversarial data collection (ADC), a human workforce interacts with a model in real time, attempting to produce examples that elicit incorrect predictions. Researchers hope that models trained on these more challenging datasets will rely less on superficial patterns, and thus be less brittle. However, despite ADC's intuitive appeal, it remains unclear when training on adversarial datasets produces more robust models. In this paper, we conduct a large-scale controlled study focused on question answering, assigning workers at random to compose questions either (i) adversarially (with a model in the loop); or (ii) in the standard fashion (without a model). Across a variety of models and datasets, we find that models trained on adversarial data usually perform better on other adversarial datasets but worse on a diverse collection of out-of-domain evaluation sets. Finally, we provide a qualitative analysis of adversarial (vs standard) data, identifying key differences and offering guidance for future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our latest work, <a href="https://twitter.com/douwekiela?ref_src=twsrc%5Etfw">@douwekiela</a>, <a href="https://twitter.com/zacharylipton?ref_src=twsrc%5Etfw">@zacharylipton</a>, <a href="https://twitter.com/scottyih?ref_src=twsrc%5Etfw">@scottyih</a> and I conduct a randomized controlled study to examine the efficacy of adversarial data collection (to appear at ACL-IJCNLP 2021 <a href="https://twitter.com/aclmeeting?ref_src=twsrc%5Etfw">@aclmeeting</a>). <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a>  Thread 🧵 1/11<a href="https://t.co/9WBDcsGS2d">https://t.co/9WBDcsGS2d</a> <a href="https://t.co/23Lm4s5Uss">pic.twitter.com/23Lm4s5Uss</a></p>&mdash; Divyansh Kaushik (@dkaushik96) <a href="https://twitter.com/dkaushik96/status/1400261950634971138?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Neural message passing for joint paratope-epitope prediction

Alice Del Vecchio, Andreea Deac, Pietro Liò, Petar Veličković

- retweets: 256, favorites: 101 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.00757) | [pdf](https://arxiv.org/pdf/2106.00757)
- [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Antibodies are proteins in the immune system which bind to antigens to detect and neutralise them. The binding sites in an antibody-antigen interaction are known as the paratope and epitope, respectively, and the prediction of these regions is key to vaccine and synthetic antibody development. Contrary to prior art, we argue that paratope and epitope predictors require asymmetric treatment, and propose distinct neural message passing architectures that are geared towards the specific aspects of paratope and epitope prediction, respectively. We obtain significant improvements on both tasks, setting the new state-of-the-art and recovering favourable qualitative predictions on antigens of relevance to COVID-19.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out EPMP, our latest GNN-based predictor of antibody/antigen binding interfaces:<a href="https://t.co/42DDmmsTbU">https://t.co/42DDmmsTbU</a><br><br>We use the proteins&#39; individual geometries but *not* their relative positions, setting new state-of-the-art. More below! 🧵<br><br>w/ Alice, <a href="https://twitter.com/andreeadeac22?ref_src=twsrc%5Etfw">@andreeadeac22</a>, <a href="https://twitter.com/pl219_Cambridge?ref_src=twsrc%5Etfw">@pl219_Cambridge</a> <a href="https://t.co/Vb4CoG7x8i">pic.twitter.com/Vb4CoG7x8i</a></p>&mdash; Petar Veličković (@PetarV_93) <a href="https://twitter.com/PetarV_93/status/1400253580016295938?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Diffusion Schrödinger Bridge with Applications to Score-Based  Generative Modeling

Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet

- retweets: 236, favorites: 112 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.01357) | [pdf](https://arxiv.org/pdf/2106.01357)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.PR](https://arxiv.org/list/math.PR/recent)

Progressively applying Gaussian noise transforms complex data distributions to approximately Gaussian. Reversing this dynamic defines a generative model. When the forward noising process is given by a Stochastic Differential Equation (SDE), Song et al. (2021) demonstrate how the time inhomogeneous drift of the associated reverse-time SDE may be estimated using score-matching. A limitation of this approach is that the forward-time SDE must be run for a sufficiently long time for the final distribution to be approximately Gaussian. In contrast, solving the Schr\"odinger Bridge problem (SB), i.e. an entropy-regularized optimal transport problem on path spaces, yields diffusions which generate samples from the data distribution in finite time. We present Diffusion SB (DSB), an original approximation of the Iterative Proportional Fitting (IPF) procedure to solve the SB problem, and provide theoretical analysis along with generative modeling experiments. The first DSB iteration recovers the methodology proposed by Song et al. (2021), with the flexibility of using shorter time intervals, as subsequent DSB iterations reduce the discrepancy between the final-time marginal of the forward (resp. backward) SDE with respect to the prior (resp. data) distribution. Beyond generative modeling, DSB offers a widely applicable computational optimal transport tool as the continuous state-space analogue of the popular Sinkhorn algorithm (Cuturi, 2013).

<blockquote class="twitter-tweet"><p lang="ca" dir="ltr">Diffusion Schrödinger Bridge, <a href="https://t.co/fbll6e6Dfa">https://t.co/fbll6e6Dfa</a>  with <a href="https://twitter.com/ValentinDeBort1?ref_src=twsrc%5Etfw">@ValentinDeBort1</a>,<a href="https://twitter.com/JamesTThorn?ref_src=twsrc%5Etfw">@JamesTThorn</a>,<a href="https://twitter.com/jeremyhengjm?ref_src=twsrc%5Etfw">@jeremyhengjm</a>, a  Sinkhorn variant using iterative diffusions to solve the Schrödinger Bridge problem <br>* Accelerates score-based generative models<br>* Facilitates optimal transport in high dim <a href="https://t.co/dkpZJKLoH5">pic.twitter.com/dkpZJKLoH5</a></p>&mdash; Arnaud Doucet (@ArnaudDoucet1) <a href="https://twitter.com/ArnaudDoucet1/status/1400494921006698501?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">very cool:<a href="https://t.co/fKw4WEB0mJ">https://t.co/fKw4WEB0mJ</a><br>`Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling&#39;<br>- Valentin De Bortoli, James Thornton, Jeremy Heng, Arnaud Doucet</p>&mdash; Sam Power (@sam_power_825) <a href="https://twitter.com/sam_power_825/status/1400359530723225600?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. A Differentiable Point Process with Its Application to Spiking Neural  Networks

Hiroshi Kajino

- retweets: 168, favorites: 97 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.00901) | [pdf](https://arxiv.org/pdf/2106.00901)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

This paper is concerned about a learning algorithm for a probabilistic model of spiking neural networks (SNNs). Jimenez Rezende & Gerstner (2014) proposed a stochastic variational inference algorithm to train SNNs with hidden neurons. The algorithm updates the variational distribution using the score function gradient estimator, whose high variance often impedes the whole learning algorithm. This paper presents an alternative gradient estimator for SNNs based on the path-wise gradient estimator. The main technical difficulty is a lack of a general method to differentiate a realization of an arbitrary point process, which is necessary to derive the path-wise gradient estimator. We develop a differentiable point process, which is the technical highlight of this paper, and apply it to derive the path-wise gradient estimator for SNNs. We investigate the effectiveness of our gradient estimator through numerical simulation.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">ICML-21 に採択された論文のプレプリントとコードを公開しました。<br>1. 微分可能な点過程を作った<br>2. ニューロン同士がスパイクで通信する spiking neural network の学習に応用した<br>3. 2/3くらいのデータで学習可能<br>4. コマンド一発で実験再現可能<a href="https://t.co/qUScRcKbKm">https://t.co/qUScRcKbKm</a> <a href="https://t.co/gtK6gu4j0o">https://t.co/gtK6gu4j0o</a></p>&mdash; ガッキーのプライベートアカウント (@azaazarashi) <a href="https://twitter.com/azaazarashi/status/1400318199078391809?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Connections and Equivalences between the Nyström Method and Sparse  Variational Gaussian Processes

Veit Wild, Motonobu Kanagawa, Dino Sejdinovic

- retweets: 169, favorites: 93 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.01121) | [pdf](https://arxiv.org/pdf/2106.01121)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.ST](https://arxiv.org/list/math.ST/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent)

We investigate the connections between sparse approximation methods for making kernel methods and Gaussian processes (GPs) scalable to massive data, focusing on the Nystr\"om method and the Sparse Variational Gaussian Processes (SVGP). While sparse approximation methods for GPs and kernel methods share some algebraic similarities, the literature lacks a deep understanding of how and why they are related. This is a possible obstacle for the communications between the GP and kernel communities, making it difficult to transfer results from one side to the other. Our motivation is to remove this possible obstacle, by clarifying the connections between the sparse approximations for GPs and kernel methods. In this work, we study the two popular approaches, the Nystr\"om and SVGP approximations, in the context of a regression problem, and establish various connections and equivalences between them. In particular, we provide an RKHS interpretation of the SVGP approximation, and show that the Evidence Lower Bound of the SVGP contains the objective function of the Nystr\"om approximation, revealing the origin of the algebraic equivalence between the two approaches. We also study recently established convergence results for the SVGP and how they are related to the approximation quality of the Nystr\"om method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RKHS methods and Gaussian Processes both have large-scale approximations based on &quot;inducing points&quot;: Nystrom method and sparse variational GPs. New preprint <a href="https://t.co/7DXdFTXyBG">https://t.co/7DXdFTXyBG</a> shows the connections between the two go very deep and bring new insights about both frameworks.</p>&mdash; Dino Sejdinovic (@sejDino) <a href="https://twitter.com/sejDino/status/1400409571437469697?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Babel Fees via Limited Liabilities

Manuel M. T. Chakravarty, Nikos Karayannidis, Aggelos Kiayias, Michael Peyton Jones, Polina Vinogradova

- retweets: 168, favorites: 73 (06/04/2021 06:43:26)

- links: [abs](https://arxiv.org/abs/2106.01161) | [pdf](https://arxiv.org/pdf/2106.01161)
- [cs.CR](https://arxiv.org/list/cs.CR/recent)

Custom currencies (ERC-20) on Ethereum are wildly popular, but they are second class to the primary currency Ether. Custom currencies are more complex and more expensive to handle than the primary currency as their accounting is not natively performed by the underlying ledger, but instead in user-defined contract code. Furthermore, and quite importantly, transaction fees can only be paid in Ether.   In this paper, we focus on being able to pay transaction fees in custom currencies. We achieve this by way of a mechanism permitting short term liabilities to pay transaction fees in conjunction with offers of custom currencies to compensate for those liabilities. This enables block producers to accept custom currencies in exchange for settling liabilities of transactions that they process.   We present formal ledger rules to handle liabilities together with the concept of babel fees to pay transaction fees in custom currencies. We also discuss how clients can determine what fees they have to pay, and we present a solution to the knapsack problem variant that block producers have to solve in the presence of babel fees to optimise their profits.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">With all the excitement around custom currencies (such as ERC-20 tokens), you still have to pay transaction fees in the blockchain’s primary currency: Our latest research paper proposes a mechanism to change that: “Babel Fees via Limited Liabilities”: <a href="https://t.co/O68cxGfTlc">https://t.co/O68cxGfTlc</a></p>&mdash; Manuel Chakravarty (@TacticalGrace) <a href="https://twitter.com/TacticalGrace/status/1400400559346180098?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Bottom-Up and Top-Down Neural Processing Systems Design: Neuromorphic  Intelligence as the Convergence of Natural and Artificial Intelligence

Charlotte Frenkel, David Bol, Giacomo Indiveri

- retweets: 196, favorites: 42 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.01288) | [pdf](https://arxiv.org/pdf/2106.01288)
- [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent)

While Moore's law has driven exponential computing power expectations, its nearing end calls for new avenues for improving the overall system performance. One of these avenues is the exploration of new alternative brain-inspired computing architectures that promise to achieve the flexibility and computational efficiency of biological neural processing systems. Within this context, neuromorphic intelligence represents a paradigm shift in computing based on the implementation of spiking neural network architectures tightly co-locating processing and memory. In this paper, we provide a comprehensive overview of the field, highlighting the different levels of granularity present in existing silicon implementations, comparing approaches that aim at replicating natural intelligence (bottom-up) versus those that aim at solving practical artificial intelligence applications (top-down), and assessing the benefits of the different circuit design styles used to achieve these goals. First, we present the analog, mixed-signal and digital circuit design styles, identifying the boundary between processing and memory through time multiplexing, in-memory computation and novel devices. Next, we highlight the key tradeoffs for each of the bottom-up and top-down approaches, survey their silicon implementations, and carry out detailed comparative analyses to extract design guidelines. Finally, we identify both necessary synergies and missing elements required to achieve a competitive advantage for neuromorphic edge computing over conventional machine-learning accelerators, and outline the key elements for a framework toward neuromorphic intelligence.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The preprint of our big <a href="https://twitter.com/hashtag/neuromorphic?src=hash&amp;ref_src=twsrc%5Etfw">#neuromorphic</a> review paper is out! 🧐<br>Analog or digital? Bottom-up or top-down? Comparison with ANN accelerators? Promising use cases, future trends?<br>We survey these key questions with <a href="https://twitter.com/giacomoi?ref_src=twsrc%5Etfw">@giacomoi</a> and David Bol:<br>👉 <a href="https://t.co/CF8YOICRQE">https://t.co/CF8YOICRQE</a><br>Feedback welcome! <a href="https://t.co/HOS03N4EKy">pic.twitter.com/HOS03N4EKy</a></p>&mdash; Charlotte Frenkel (@C_Frenkel) <a href="https://twitter.com/C_Frenkel/status/1400367411019358210?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Hi-Transformer: Hierarchical Interactive Transformer for Efficient and  Effective Long Document Modeling

Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang

- retweets: 100, favorites: 56 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.01040) | [pdf](https://arxiv.org/pdf/2106.01040)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformer is important for text modeling. However, it has difficulty in handling long documents due to the quadratic complexity with input text length. In order to handle this problem, we propose a hierarchical interactive Transformer (Hi-Transformer) for efficient and effective long document modeling. Hi-Transformer models documents in a hierarchical way, i.e., first learns sentence representations and then learns document representations. It can effectively reduce the complexity and meanwhile capture global document context in the modeling of each sentence. More specifically, we first use a sentence Transformer to learn the representations of each sentence. Then we use a document Transformer to model the global document context from these sentence representations. Next, we use another sentence Transformer to enhance sentence modeling using the global document context. Finally, we use hierarchical pooling method to obtain document embedding. Extensive experiments on three benchmark datasets validate the efficiency and effectiveness of Hi-Transformer in long document modeling.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hi-Transformer: Hierarchical Interactive Transformer for Efficient and Effective Long Document Modeling<br>pdf: <a href="https://t.co/jX6SCkKcOU">https://t.co/jX6SCkKcOU</a><br>abs: <a href="https://t.co/kHqNfHxXle">https://t.co/kHqNfHxXle</a><br><br>a hierarchical architecture that first learns sentence representations and then learns document representations <a href="https://t.co/wNzajpgQgy">pic.twitter.com/wNzajpgQgy</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400253322309865472?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Search Methods for Sufficient, Socially-Aligned Feature Importance  Explanations with In-Distribution Counterfactuals

Peter Hase, Harry Xie, Mohit Bansal

- retweets: 110, favorites: 35 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.00786) | [pdf](https://arxiv.org/pdf/2106.00786)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Feature importance (FI) estimates are a popular form of explanation, and they are commonly created and evaluated by computing the change in model confidence caused by removing certain input features at test time. For example, in the standard Sufficiency metric, only the top-k most important tokens are kept. In this paper, we study several under-explored dimensions of FI-based explanations, providing conceptual and empirical improvements for this form of explanation. First, we advance a new argument for why it can be problematic to remove features from an input when creating or evaluating explanations: the fact that these counterfactual inputs are out-of-distribution (OOD) to models implies that the resulting explanations are socially misaligned. The crux of the problem is that the model prior and random weight initialization influence the explanations (and explanation metrics) in unintended ways. To resolve this issue, we propose a simple alteration to the model training process, which results in more socially aligned explanations and metrics. Second, we compare among five approaches for removing features from model inputs. We find that some methods produce more OOD counterfactuals than others, and we make recommendations for selecting a feature-replacement function. Finally, we introduce four search-based methods for identifying FI explanations and compare them to strong baselines, including LIME, Integrated Gradients, and random search. On experiments with six diverse text classification datasets, we find that the only method that consistently outperforms random search is a Parallel Local Search that we introduce. Improvements over the second-best method are as large as 5.4 points for Sufficiency and 17 points for Comprehensiveness. All supporting code is publicly available at https://github.com/peterbhase/ExplanationSearch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper out! “Search Methods for Sufficient, Socially-Aligned Feature Importance Explanations with In-Distribution Counterfactuals”<br><br>Eval. issues raised+resolved and new expln. methods<br><br>w/ Harry Xie, <a href="https://twitter.com/mohitban47?ref_src=twsrc%5Etfw">@mohitban47</a><br><br>arxiv: <a href="https://t.co/jtHYYULPcj">https://t.co/jtHYYULPcj</a><br>demo: <a href="https://t.co/jpV5joAobu">https://t.co/jpV5joAobu</a><br><br>1/n <a href="https://t.co/3VyGS14TsS">pic.twitter.com/3VyGS14TsS</a></p>&mdash; Peter Hase (@peterbhase) <a href="https://twitter.com/peterbhase/status/1400261601802870784?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. Examining the Inductive Bias of Neural Language Models with Artificial  Languages

Jennifer C. White, Ryan Cotterell

- retweets: 64, favorites: 58 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.01044) | [pdf](https://arxiv.org/pdf/2106.01044)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Since language models are used to model a wide variety of languages, it is natural to ask whether the neural architectures used for the task have inductive biases towards modeling particular types of languages. Investigation of these biases has proved complicated due to the many variables that appear in the experimental setup. Languages vary in many typological dimensions, and it is difficult to single out one or two to investigate without the others acting as confounders. We propose a novel method for investigating the inductive biases of language models using artificial languages. These languages are constructed to allow us to create parallel corpora across languages that differ only in the typological feature being investigated, such as word order. We then use them to train and test language models. This constitutes a fully controlled causal framework, and demonstrates how grammar engineering can serve as a useful tool for analyzing neural models. Using this method, we find that commonly used neural architectures exhibit different inductive biases: LSTMs display little preference with respect to word ordering, while transformers display a clear preference for some orderings over others. Further, we find that neither the inductive bias of the LSTM nor that of the transformer appears to reflect any tendencies that we see in attested natural languages.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out my new <a href="https://twitter.com/hashtag/ACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021</a> paper with <a href="https://twitter.com/ryandcotterell?ref_src=twsrc%5Etfw">@ryandcotterell</a> where we use entirely artificial languages to investigate the inductive bias of language models! <a href="https://t.co/mRjrCozDxp">https://t.co/mRjrCozDxp</a> <a href="https://t.co/JFJS8jlyNj">pic.twitter.com/JFJS8jlyNj</a></p>&mdash; Jennifer White (@JenniferCWhite) <a href="https://twitter.com/JenniferCWhite/status/1400508810691817472?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. JUMBO: Scalable Multi-task Bayesian Optimization using Offline Data

Kourosh Hakhamaneshi, Pieter Abbeel, Vladimir Stojanovic, Aditya Grover

- retweets: 42, favorites: 32 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.00942) | [pdf](https://arxiv.org/pdf/2106.00942)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

The goal of Multi-task Bayesian Optimization (MBO) is to minimize the number of queries required to accurately optimize a target black-box function, given access to offline evaluations of other auxiliary functions. When offline datasets are large, the scalability of prior approaches comes at the expense of expressivity and inference quality. We propose JUMBO, an MBO algorithm that sidesteps these limitations by querying additional data based on a combination of acquisition signals derived from training two Gaussian Processes (GP): a cold-GP operating directly in the input domain and a warm-GP that operates in the feature space of a deep neural network pretrained using the offline data. Such a decomposition can dynamically control the reliability of information derived from the online and offline data and the use of pretrained neural networks permits scalability to large offline datasets. Theoretically, we derive regret bounds for JUMBO and show that it achieves no-regret under conditions analogous to GP-UCB (Srinivas et. al. 2010). Empirically, we demonstrate significant performance improvements over existing approaches on two real-world optimization problems: hyper-parameter optimization and automated circuit design.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">JUMBO: Scalable Multi-task Bayesian Optimization<br>using Offline Data<br>pdf: <a href="https://t.co/FCuxfJUd73">https://t.co/FCuxfJUd73</a><br>abs: <a href="https://t.co/MwojvwyOt1">https://t.co/MwojvwyOt1</a><br><br>employs a hybrid of nns and Gaussian Processes and a novel acquisition procedure for scalable and sample-efficient Multi-task Bayesian Optimization <a href="https://t.co/bGwExzAVaz">pic.twitter.com/bGwExzAVaz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1400255493566865410?ref_src=twsrc%5Etfw">June 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Warming-up recurrent neural networks to maximize reachable  multi-stability greatly improves learning

Nicolas Vecoven, Damien Ernst, Guillaume Drion

- retweets: 58, favorites: 14 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.01001) | [pdf](https://arxiv.org/pdf/2106.01001)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Training recurrent neural networks is known to be difficult when time dependencies become long. Consequently, training standard gated cells such as gated recurrent units and long-short term memory on benchmarks where long-term memory is required remains an arduous task. In this work, we propose a general way to initialize any recurrent network connectivity through a process called "warm-up" to improve its capability to learn arbitrarily long time dependencies. This initialization process is designed to maximize network reachable multi-stability, i.e. the number of attractors within the network that can be reached through relevant input trajectories. Warming-up is performed before training, using stochastic gradient descent on a specifically designed loss. We show that warming-up greatly improves recurrent neural network performance on long-term memory benchmarks for multiple recurrent cell types, but can sometimes impede precision. We therefore introduce a parallel recurrent network structure with partial warm-up that is shown to greatly improve learning on long time-series while maintaining high levels of precision. This approach provides a general framework for improving learning abilities of any recurrent cell type when long-term memory is required.




# 21. Lower Perplexity is Not Always Human-Like

Tatsuki Kuribayashi, Yohei Oseki, Takumi Ito, Ryo Yoshida, Masayuki Asahara, Kentaro Inui

- retweets: 42, favorites: 16 (06/04/2021 06:43:27)

- links: [abs](https://arxiv.org/abs/2106.01229) | [pdf](https://arxiv.org/pdf/2106.01229)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

In computational psycholinguistics, various language models have been evaluated against human reading behavior (e.g., eye movement) to build human-like computational models. However, most previous efforts have focused almost exclusively on English, despite the recent trend towards linguistic universal within the general community. In order to fill the gap, this paper investigates whether the established results in computational psycholinguistics can be generalized across languages. Specifically, we re-examine an established generalization -- the lower perplexity a language model has, the more human-like the language model is -- in Japanese with typologically different structures from English. Our experiments demonstrate that this established generalization exhibits a surprising lack of universality; namely, lower perplexity is not always human-like. Moreover, this discrepancy between English and Japanese is further explored from the perspective of (non-)uniform information density. Overall, our results suggest that a cross-lingual evaluation will be necessary to construct human-like computational models.



