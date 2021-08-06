---
title: Hot Papers 2021-08-05
date: 2021-08-06T22:50:21.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-05"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-05"
socialImage: "/media/flying-marine.jpg"

---

# 1. Lachesis: Scalable Asynchronous BFT on DAG Streams

Quan Nguyen, Andre Cronje, Michael Kong, Egor Lysenko, Alex Guzev

- retweets: 12268, favorites: 15 (08/06/2021 22:50:21)

- links: [abs](https://arxiv.org/abs/2108.01900) | [pdf](https://arxiv.org/pdf/2108.01900)
- [cs.DC](https://arxiv.org/list/cs.DC/recent)

This paper consolidates the core technologies and key concepts of our novel Lachesis consensus protocol and Fantom Opera platform, which is permissionless, leaderless and EVM compatible.   We introduce our new protocol, so-called Lachesis, for distributed networks achieving Byzantine fault tolerance (BFT)~\cite{lachesis01}. Each node in Lachesis protocol operates on a local block DAG, namely \emph{OPERA DAG}. Aiming for a low time to finality (TTF) for transactions, our general model considers DAG streams of high speed but asynchronous events. We integrate Proof-of-Stake (PoS) into a DAG model in Lachesis protocol to improve performance and security. Our general model of trustless system leverages participants' stake as their validating power~\cite{stakedag}. Lachesis's consensus algorithm uses Lamport timestamps, graph layering and concurrent common knowledge to guarantee a consistent total ordering of event blocks and transactions. In addition, Lachesis protocol allows dynamic participation of new nodes into Opera network. Lachesis optimizes DAG storage and processing time by splitting local history into checkpoints (so-called epochs). We also propose a model to improve stake decentralization, and network safety and liveness ~\cite{stairdag}.   Built on our novel Lachesis protocol, Fantom's Opera platform is a public, leaderless, asynchronous BFT, layer-1 blockchain, with guaranteed deterministic finality. Hence, Lachesis protocol is suitable for distributed ledgers by leveraging asynchronous partially ordered sets with logical time ordering instead of blockchains. We also present our proofs into a model that can be applied to abstract asynchronous distributed system.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">As much as I love defi, my real passion is consensus research, I still believe that localized decentralized adversarial consensus is a breakthrough that will change the world far beyond crypto.<br><br>Lachesis is my pride and joy<a href="https://t.co/awgnZCSA0p">https://t.co/awgnZCSA0p</a></p>&mdash; Andre Cronje (@AndreCronjeTech) <a href="https://twitter.com/AndreCronjeTech/status/1423241672364802048?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Vision Transformer with Progressive Sampling

Xiaoyu Yue, Shuyang Sun, Zhanghui Kuang, Meng Wei, Philip Torr, Wayne Zhang, Dahua Lin

- retweets: 1434, favorites: 184 (08/06/2021 22:50:22)

- links: [abs](https://arxiv.org/abs/2108.01684) | [pdf](https://arxiv.org/pdf/2108.01684)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Transformers with powerful global relation modeling abilities have been introduced to fundamental computer vision tasks recently. As a typical example, the Vision Transformer (ViT) directly applies a pure transformer architecture on image classification, by simply splitting images into tokens with a fixed length, and employing transformers to learn relations between these tokens. However, such naive tokenization could destruct object structures, assign grids to uninterested regions such as background, and introduce interference signals. To mitigate the above issues, in this paper, we propose an iterative and progressive sampling strategy to locate discriminative regions. At each iteration, embeddings of the current sampling step are fed into a transformer encoder layer, and a group of sampling offsets is predicted to update the sampling locations for the next step. The progressive sampling is differentiable. When combined with the Vision Transformer, the obtained PS-ViT network can adaptively learn where to look. The proposed PS-ViT is both effective and efficient. When trained from scratch on ImageNet, PS-ViT performs 3.8% higher than the vanilla ViT in terms of top-1 accuracy with about $4\times$ fewer parameters and $10\times$ fewer FLOPs. Code is available at https://github.com/yuexy/PS-ViT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vision Transformer with Progressive Sampling<br>pdf: <a href="https://t.co/UW4Q8YmWPi">https://t.co/UW4Q8YmWPi</a><br>abs: <a href="https://t.co/usaqUHuSkS">https://t.co/usaqUHuSkS</a><br><br>When trained from scratch on ImageNet, PS-ViT performs 3.8% higher than the vanilla ViT in terms of top-1 accuracy with about 4× fewer parameters and 10× fewer FLOPs <a href="https://t.co/ikxFIUuk9M">pic.twitter.com/ikxFIUuk9M</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423082450964688896?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. A Pragmatic Look at Deep Imitation Learning

Kai Arulkumaran, Dan Ogawa Lillrank

- retweets: 706, favorites: 111 (08/06/2021 22:50:22)

- links: [abs](https://arxiv.org/abs/2108.01867) | [pdf](https://arxiv.org/pdf/2108.01867)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The introduction of the generative adversarial imitation learning (GAIL) algorithm has spurred the development of scalable imitation learning approaches using deep neural networks. The GAIL objective can be thought of as 1) matching the expert policy's state distribution; 2) penalising the learned policy's state distribution; and 3) maximising entropy. While theoretically motivated, in practice GAIL can be difficult to apply, not least due to the instabilities of adversarial training. In this paper, we take a pragmatic look at GAIL and related imitation learning algorithms. We implement and automatically tune a range of algorithms in a unified experimental setup, presenting a fair evaluation between the competing methods. From our results, our primary recommendation is to consider non-adversarial methods. Furthermore, we discuss the common components of imitation learning objectives, and present promising avenues for future research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pleased to release &quot;A Pragmatic Look at Deep Imitation Learning&quot; w/ <a href="https://twitter.com/DanOL71607511?ref_src=twsrc%5Etfw">@DanOL71607511</a>: <a href="https://t.co/g8j4pJok6A">https://t.co/g8j4pJok6A</a><br><br>We look at, and try to benchmark lots of different IL algos.  The takeaway? No need to be so adversarial 😉 <a href="https://t.co/5EoBaXCT43">pic.twitter.com/5EoBaXCT43</a></p>&mdash; Kai Arulkumaran (@kaixhin) <a href="https://twitter.com/kaixhin/status/1423093727392763905?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Internal Video Inpainting by Implicit Long-range Propagation

Hao Ouyang, Tengfei Wang, Qifeng Chen

- retweets: 272, favorites: 76 (08/06/2021 22:50:22)

- links: [abs](https://arxiv.org/abs/2108.01912) | [pdf](https://arxiv.org/pdf/2108.01912)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel framework for video inpainting by adopting an internal learning strategy. Unlike previous methods that use optical flow for cross-frame context propagation to inpaint unknown regions, we show that this can be achieved implicitly by fitting a convolutional neural network to the known region. Moreover, to handle challenging sequences with ambiguous backgrounds or long-term occlusion, we design two regularization terms to preserve high-frequency details and long-term temporal consistency. Extensive experiments on the DAVIS dataset demonstrate that the proposed method achieves state-of-the-art inpainting quality quantitatively and qualitatively. We further extend the proposed method to another challenging task: learning to remove an object from a video giving a single object mask in only one frame in a 4K video.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Internal Video Inpainting by Implicit Long-range Propagation<br>pdf: <a href="https://t.co/yTau322GpO">https://t.co/yTau322GpO</a><br>abs: <a href="https://t.co/jShDPOlqVn">https://t.co/jShDPOlqVn</a><br>project page: <a href="https://t.co/zFOR6q4aik">https://t.co/zFOR6q4aik</a> <a href="https://t.co/qOIW3Nl2m6">pic.twitter.com/qOIW3Nl2m6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423085175639052289?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. PARADISE: Exploiting Parallel Data for Multilingual Sequence-to-Sequence  Pretraining

Machel Reid, Mikel Artetxe

- retweets: 76, favorites: 56 (08/06/2021 22:50:22)

- links: [abs](https://arxiv.org/abs/2108.01887) | [pdf](https://arxiv.org/pdf/2108.01887)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite the success of multilingual sequence-to-sequence pretraining, most existing approaches rely on monolingual corpora, and do not make use of the strong cross-lingual signal contained in parallel data. In this paper, we present PARADISE (PARAllel & Denoising Integration in SEquence-to-sequence models), which extends the conventional denoising objective used to train these models by (i) replacing words in the noised sequence according to a multilingual dictionary, and (ii) predicting the reference translation according to a parallel corpus instead of recovering the original sequence. Our experiments on machine translation and cross-lingual natural language inference show an average improvement of 2.0 BLEU points and 6.7 accuracy points from integrating parallel data into pretraining, respectively, obtaining results that are competitive with several popular models at a fraction of their computational cost.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Welcome to PARADISE! We propose a new, more efficient method for multilingual sequence to sequence pre-training by leveraging smaller corpora of word/sentence parallel data for improved cross-lingual Seq2Seq pre-training.<br><br>w/ <a href="https://twitter.com/artetxem?ref_src=twsrc%5Etfw">@artetxem</a> <br><br>📃 <a href="https://t.co/cix2BY9Gbn">https://t.co/cix2BY9Gbn</a><br>🧵 (1/) <a href="https://t.co/YANRqBL5lq">pic.twitter.com/YANRqBL5lq</a></p>&mdash; Machel Reid (@machelreid) <a href="https://twitter.com/machelreid/status/1423120273746763781?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. FedJAX: Federated learning simulation with JAX

Jae Hun Ro, Ananda Theertha Suresh, Ke Wu

- retweets: 49, favorites: 34 (08/06/2021 22:50:22)

- links: [abs](https://arxiv.org/abs/2108.02117) | [pdf](https://arxiv.org/pdf/2108.02117)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Federated learning is a machine learning technique that enables training across decentralized data. Recently, federated learning has become an active area of research due to the increased concerns over privacy and security. In light of this, a variety of open source federated learning libraries have been developed and released. We introduce FedJAX, a JAX-based open source library for federated learning simulations that emphasizes ease-of-use in research. With its simple primitives for implementing federated learning algorithms, prepackaged datasets, models and algorithms, and fast simulation speed, FedJAX aims to make developing and evaluating federated algorithms faster and easier for researchers. Our benchmark results show that FedJAX can be used to train models with federated averaging on the EMNIST dataset in a few minutes and the Stack Overflow dataset in roughly an hour with standard hyperparmeters using TPUs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FedJAX: Federated learning simulation with JAX<br>pdf: <a href="https://t.co/fD1QqOROKf">https://t.co/fD1QqOROKf</a><br>abs: <a href="https://t.co/NjNScV4KxB">https://t.co/NjNScV4KxB</a> <a href="https://t.co/PJyxneXtI5">pic.twitter.com/PJyxneXtI5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423129644300177408?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Policy Gradients Incorporating the Future

David Venuto, Elaine Lau, Doina Precup, Ofir Nachum

- retweets: 30, favorites: 39 (08/06/2021 22:50:23)

- links: [abs](https://arxiv.org/abs/2108.02096) | [pdf](https://arxiv.org/pdf/2108.02096)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Reasoning about the future -- understanding how decisions in the present time affect outcomes in the future -- is one of the central challenges for reinforcement learning (RL), especially in highly-stochastic or partially observable environments. While predicting the future directly is hard, in this work we introduce a method that allows an agent to "look into the future" without explicitly predicting it. Namely, we propose to allow an agent, during its training on past experience, to observe what \emph{actually} happened in the future at that time, while enforcing an information bottleneck to avoid the agent overly relying on this privileged information. This gives our agent the opportunity to utilize rich and useful information about the future trajectory dynamics in addition to the present. Our method, Policy Gradients Incorporating the Future (PGIF), is easy to implement and versatile, being applicable to virtually any policy gradient algorithm. We apply our proposed method to a number of off-the-shelf RL algorithms and show that PGIF is able to achieve higher reward faster in a variety of online and offline RL domains, as well as sparse-reward and partially observable environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Policy Gradients Incorporating the Future<br>pdf: <a href="https://t.co/RNx389aXQv">https://t.co/RNx389aXQv</a><br>abs: <a href="https://t.co/DakZTEluOC">https://t.co/DakZTEluOC</a><br><br>is able to achieve higher reward faster in a variety of online and offline RL domains, as well as sparse-reward and partially observable environments <a href="https://t.co/W2tU1UZqqj">pic.twitter.com/W2tU1UZqqj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1423129208763600905?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Q-Pain: A Question Answering Dataset to Measure Social Bias in Pain  Management

Cécile Logé, Emily Ross, David Yaw Amoah Dadey, Saahil Jain, Adriel Saporta, Andrew Y. Ng, Pranav Rajpurkar

- retweets: 20, favorites: 45 (08/06/2021 22:50:23)

- links: [abs](https://arxiv.org/abs/2108.01764) | [pdf](https://arxiv.org/pdf/2108.01764)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Recent advances in Natural Language Processing (NLP), and specifically automated Question Answering (QA) systems, have demonstrated both impressive linguistic fluency and a pernicious tendency to reflect social biases. In this study, we introduce Q-Pain, a dataset for assessing bias in medical QA in the context of pain management, one of the most challenging forms of clinical decision-making. Along with the dataset, we propose a new, rigorous framework, including a sample experimental design, to measure the potential biases present when making treatment decisions. We demonstrate its use by assessing two reference Question-Answering systems, GPT-2 and GPT-3, and find statistically significant differences in treatment between intersectional race-gender subgroups, thus reaffirming the risks posed by AI in medical settings, and the need for datasets like ours to ensure safety before medical AI applications are deployed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨 New in NeurIPS Datasets &amp; Benchmarks 🧵⤵️:<br><br>We asked GPT-3 to prescribe pain treatment for patients of different races and genders. Here&#39;s what we found:<a href="https://t.co/4sd3ky9Vaa">https://t.co/4sd3ky9Vaa</a><a href="https://twitter.com/CeciLoge?ref_src=twsrc%5Etfw">@ceciloge</a> <a href="https://twitter.com/mle_ross?ref_src=twsrc%5Etfw">@mle_ross</a> <a href="https://twitter.com/DYDadey?ref_src=twsrc%5Etfw">@DYDadey</a> <a href="https://twitter.com/saahil9jain?ref_src=twsrc%5Etfw">@saahil9jain</a>  <a href="https://twitter.com/ARSaporta?ref_src=twsrc%5Etfw">@ARSaporta</a> <a href="https://twitter.com/AndrewYNg?ref_src=twsrc%5Etfw">@AndrewYNg</a> <a href="https://twitter.com/StanfordAILab?ref_src=twsrc%5Etfw">@StanfordAILab</a> <a href="https://twitter.com/HarvardDBMI?ref_src=twsrc%5Etfw">@HarvardDBMI</a><br><br>1/10 <a href="https://t.co/iYGibFF3X3">pic.twitter.com/iYGibFF3X3</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1423253320961957890?ref_src=twsrc%5Etfw">August 5, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. High Performance Across Two Atari Paddle Games Using the Same Perceptual  Control Architecture Without Training

Tauseef Gulrez, Warren Mansell

- retweets: 42, favorites: 14 (08/06/2021 22:50:23)

- links: [abs](https://arxiv.org/abs/2108.01895) | [pdf](https://arxiv.org/pdf/2108.01895)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

Deep reinforcement learning (DRL) requires large samples and a long training time to operate optimally. Yet humans rarely require long periods training to perform well on novel tasks, such as computer games, once they are provided with an accurate program of instructions. We used perceptual control theory (PCT) to construct a simple closed-loop model which requires no training samples and training time within a video game study using the Arcade Learning Environment (ALE). The model was programmed to parse inputs from the environment into hierarchically organised perceptual signals, and it computed a dynamic error signal by subtracting the incoming signal for each perceptual variable from a reference signal to drive output signals to reduce this error. We tested the same model across two different Atari paddle games Breakout and Pong to achieve performance at least as high as DRL paradigms, and close to good human performance. Our study shows that perceptual control models, based on simple assumptions, can perform well without learning. We conclude by specifying a parsimonious role of learning that may be more similar to psychological functioning.



