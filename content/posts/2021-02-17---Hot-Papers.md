---
title: Hot Papers 2021-02-17
date: 2021-02-18T09:33:19.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-17"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-17"
socialImage: "/media/flying-marine.jpg"

---

# 1. COMBO: Conservative Offline Model-Based Policy Optimization

Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, Chelsea Finn

- retweets: 3320, favorites: 417 (02/18/2021 09:33:19)

- links: [abs](https://arxiv.org/abs/2102.08363) | [pdf](https://arxiv.org/pdf/2102.08363)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Model-based algorithms, which learn a dynamics model from logged experience and perform some sort of pessimistic planning under the learned model, have emerged as a promising paradigm for offline reinforcement learning (offline RL). However, practical variants of such model-based algorithms rely on explicit uncertainty quantification for incorporating pessimism. Uncertainty estimation with complex models, such as deep neural networks, can be difficult and unreliable. We overcome this limitation by developing a new model-based offline RL algorithm, COMBO, that regularizes the value function on out-of-support state-action tuples generated via rollouts under the learned model. This results in a conservative estimate of the value function for out-of-support state-action tuples, without requiring explicit uncertainty estimation. We theoretically show that our method optimizes a lower bound on the true policy value, that this bound is tighter than that of prior methods, and our approach satisfies a policy improvement guarantee in the offline setting. Through experiments, we find that COMBO consistently performs as well or better as compared to prior offline model-free and model-based methods on widely studied offline RL benchmarks, including image-based tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A new algo for RL from offline data:<br>COMBO pushes down the est. value of states &amp; actions that are likely to be out-of-support.<br><br>This leads to strong performance &amp; has theoretical guarantees!<a href="https://t.co/aZLgoueL8k">https://t.co/aZLgoueL8k</a><br><br>w/ <a href="https://twitter.com/TianheYu?ref_src=twsrc%5Etfw">@TianheYu</a> <a href="https://twitter.com/aviral_kumar2?ref_src=twsrc%5Etfw">@aviral_kumar2</a> <a href="https://twitter.com/rmrafailov?ref_src=twsrc%5Etfw">@rmrafailov</a> <a href="https://twitter.com/aravindr93?ref_src=twsrc%5Etfw">@aravindr93</a> <a href="https://twitter.com/svlevine?ref_src=twsrc%5Etfw">@svlevine</a> <a href="https://t.co/qZvUXyMUP8">pic.twitter.com/qZvUXyMUP8</a></p>&mdash; Chelsea Finn (@chelseabfinn) <a href="https://twitter.com/chelseabfinn/status/1361920873628770305?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A principled offline model-based RL method: COMBO combines conservative Q-learning (CQL) with model-based learning, providing state-of-the-art offline RL results and formal guarantees!<a href="https://t.co/qi3u9erxOV">https://t.co/qi3u9erxOV</a><br>w/ <a href="https://twitter.com/TianheYu?ref_src=twsrc%5Etfw">@TianheYu</a>, Aviral Kumar, R. Rafailov, <a href="https://twitter.com/aravindr93?ref_src=twsrc%5Etfw">@aravindr93</a>, <a href="https://twitter.com/chelseabfinn?ref_src=twsrc%5Etfw">@chelseabfinn</a> <a href="https://t.co/IFGbStG2Oc">pic.twitter.com/IFGbStG2Oc</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1361908304423751684?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale  Language Models

Zhuohan Li, Siyuan Zhuang, Shiyuan Guo, Danyang Zhuo, Hao Zhang, Dawn Song, Ion Stoica

- retweets: 670, favorites: 156 (02/18/2021 09:33:19)

- links: [abs](https://arxiv.org/abs/2102.07988) | [pdf](https://arxiv.org/pdf/2102.07988)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Model parallelism has become a necessity for training modern large-scale deep language models. In this work, we identify a new and orthogonal dimension from existing model parallel approaches: it is possible to perform pipeline parallelism within a single training sequence for Transformer-based language models thanks to its autoregressive property. This enables a more fine-grained pipeline compared with previous work. With this key idea, we design TeraPipe, a high-performance token-level pipeline parallel algorithm for synchronous model-parallel training of Transformer-based language models. We develop a novel dynamic programming-based algorithm to calculate the optimal pipelining execution scheme given a specific model and cluster configuration. We show that TeraPipe can speed up the training by 5.0x for the largest GPT-3 model with 175 billion parameters on an AWS cluster with 48 p3.16xlarge instances compared with state-of-the-art model-parallel methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models<br><br>Speeds up the training of GPT-3 by 5.0x on an AWS cluster with 48 p3.16xlarge instances over the SotA model-parallel methods by token-level pipelining.<a href="https://t.co/NOunZQ5bYs">https://t.co/NOunZQ5bYs</a> <a href="https://t.co/BKVXzI0Btg">pic.twitter.com/BKVXzI0Btg</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1361855820783214593?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. GradInit: Learning to Initialize Neural Networks for Stable and  Efficient Training

Chen Zhu, Renkun Ni, Zheng Xu, Kezhi Kong, W. Ronny Huang, Tom Goldstein

- retweets: 615, favorites: 161 (02/18/2021 09:33:20)

- links: [abs](https://arxiv.org/abs/2102.08098) | [pdf](https://arxiv.org/pdf/2102.08098)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Changes in neural architectures have fostered significant breakthroughs in language modeling and computer vision. Unfortunately, novel architectures often require re-thinking the choice of hyperparameters (e.g., learning rate, warmup schedule, and momentum coefficients) to maintain stability of the optimizer. This optimizer instability is often the result of poor parameter initialization, and can be avoided by architecture-specific initialization schemes. In this paper, we present GradInit, an automated and architecture agnostic method for initializing neural networks. GradInit is based on a simple heuristic; the variance of each network layer is adjusted so that a single step of SGD or Adam results in the smallest possible loss value. This adjustment is done by introducing a scalar multiplier variable in front of each parameter block, and then optimizing these variables using a simple numerical scheme. GradInit accelerates the convergence and test performance of many convolutional architectures, both with or without skip connections, and even without normalization layers. It also enables training the original Post-LN Transformer for machine translation without learning rate warmup under a wide range of learning rates and momentum coefficients. Code is available at https://github.com/zhuchen03/gradinit.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training<br>pdf: <a href="https://t.co/XY85zNdYdw">https://t.co/XY85zNdYdw</a><br>abs: <a href="https://t.co/ArRUtJfHeO">https://t.co/ArRUtJfHeO</a> <a href="https://t.co/Bibpe7MwZi">pic.twitter.com/Bibpe7MwZi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361856814833401857?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training<a href="https://twitter.com/Eiri1114?ref_src=twsrc%5Etfw">@Eiri1114</a>, Renkun Ni, Zheng Xu, Kezhi Kong, W. Ronny Huang, Tom Goldstein<br><br>Idea: &quot;the variance is adjusted so that a single step of SGD or Adam -&gt; smallest possible loss&quot;<a href="https://t.co/4ROUV8BPI7">https://t.co/4ROUV8BPI7</a> <a href="https://t.co/chvJkLF7gI">pic.twitter.com/chvJkLF7gI</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1361959642314797058?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Training Larger Networks for Deep Reinforcement Learning

Kei Ota, Devesh K. Jha, Asako Kanezaki

- retweets: 440, favorites: 149 (02/18/2021 09:33:20)

- links: [abs](https://arxiv.org/abs/2102.07920) | [pdf](https://arxiv.org/pdf/2102.07920)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

The success of deep learning in the computer vision and natural language processing communities can be attributed to training of very deep neural networks with millions or billions of parameters which can then be trained with massive amounts of data. However, similar trend has largely eluded training of deep reinforcement learning (RL) algorithms where larger networks do not lead to performance improvement. Previous work has shown that this is mostly due to instability during training of deep RL agents when using larger networks. In this paper, we make an attempt to understand and address training of larger networks for deep RL. We first show that naively increasing network capacity does not improve performance. Then, we propose a novel method that consists of 1) wider networks with DenseNet connection, 2) decoupling representation learning from training of RL, 3) a distributed training method to mitigate overfitting problems. Using this three-fold technique, we show that we can train very large networks that result in significant performance gains. We present several ablation studies to demonstrate the efficacy of the proposed method and some intuitive understanding of the reasons for performance gain. We show that our proposed method outperforms other baseline algorithms on several challenging locomotion tasks.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">新しい論文を公開しました！<br>RLにおいてもCVやNLPのように巨大なネットワークを使って性能向上できないか？というモチベーションのもと、新しいアーキテクチャを提案し性能を大きく向上できることを確認しました。<a href="https://twitter.com/kanejaki?ref_src=twsrc%5Etfw">@kanejaki</a> 先生、MERLとの共同研究結果です。<a href="https://t.co/89kNV6eCqO">https://t.co/89kNV6eCqO</a> <a href="https://t.co/yO05f3VDWk">pic.twitter.com/yO05f3VDWk</a></p>&mdash; Kei Ohta (@ohtake_i) <a href="https://twitter.com/ohtake_i/status/1361947993247600642?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Capturing the learning curves of generic features maps for realistic  data sets with a teacher-student model

Bruno Loureiro, Cédric Gerbelot, Hugo Cui, Sebastian Goldt, Florent Krzakala, Marc Mézard, Lenka Zdeborová

- retweets: 260, favorites: 110 (02/18/2021 09:33:20)

- links: [abs](https://arxiv.org/abs/2102.08127) | [pdf](https://arxiv.org/pdf/2102.08127)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.PR](https://arxiv.org/list/math.PR/recent) | [math.ST](https://arxiv.org/list/math.ST/recent)

Teacher-student models provide a powerful framework in which the typical case performance of high-dimensional supervised learning tasks can be studied in closed form. In this setting, labels are assigned to data - often taken to be Gaussian i.i.d. - by a teacher model, and the goal is to characterise the typical performance of the student model in recovering the parameters that generated the labels. In this manuscript we discuss a generalisation of this setting where the teacher and student can act on different spaces, generated with fixed, but generic feature maps. This is achieved via the rigorous study of a high-dimensional Gaussian covariate model. Our contribution is two-fold: First, we prove a rigorous formula for the asymptotic training loss and generalisation error achieved by empirical risk minimization for this model. Second, we present a number of situations where the learning curve of the model captures the one of a \emph{realistic data set} learned with kernel regression and classification, with out-of-the-box feature maps such as random projections or scattering transforms, or with pre-learned ones - such as the features learned by training multi-layer neural networks. We discuss both the power and the limitations of the Gaussian teacher-student framework as a typical case analysis capturing learning curves as encountered in practice on real data sets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sometimes in research *magic happens*! The value of the test error as a function of the number of samples can be captured by a simple teacher-student model in a range of realistic settings. I would have never believed this a year ago, see for yourself: <a href="https://t.co/47FKnvzqQD">https://t.co/47FKnvzqQD</a> <a href="https://t.co/wmT0oiXyrn">pic.twitter.com/wmT0oiXyrn</a></p>&mdash; Lenka Zdeborova (@zdeborova) <a href="https://twitter.com/zdeborova/status/1361956101395218432?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">IdePHICS proudly annonce the first paper of the team now in EPFL: <a href="https://t.co/5Uak2GrEMH">https://t.co/5Uak2GrEMH</a><br><br>Between many stunning results (inc. a 20 pages proof of a replica formula), we show how idealistic Gaussian models can actually reproduce what happens in reality with many features maps. <a href="https://t.co/GUvHOJBMU6">pic.twitter.com/GUvHOJBMU6</a></p>&mdash; IdePHICS lab (@idephics) <a href="https://twitter.com/idephics/status/1361960503170502659?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Meta Back-translation

Hieu Pham, Xinyi Wang, Yiming Yang, Graham Neubig

- retweets: 219, favorites: 133 (02/18/2021 09:33:20)

- links: [abs](https://arxiv.org/abs/2102.07847) | [pdf](https://arxiv.org/pdf/2102.07847)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Back-translation is an effective strategy to improve the performance of Neural Machine Translation~(NMT) by generating pseudo-parallel data. However, several recent works have found that better translation quality of the pseudo-parallel data does not necessarily lead to better final translation models, while lower-quality but more diverse data often yields stronger results. In this paper, we propose a novel method to generate pseudo-parallel data from a pre-trained back-translation model. Our method is a meta-learning algorithm which adapts a pre-trained back-translation model so that the pseudo-parallel data it generates would train a forward-translation model to do well on a validation set. In our evaluations in both the standard datasets WMT En-De'14 and WMT En-Fr'14, as well as a multilingual translation setting, our method leads to significant improvements over strong baselines. Our code will be made available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[1/n] Back-Translation is great. Meta Back-Translation is meta-great.<br><br>Happy to announce that our paper was accepted to <a href="https://twitter.com/hashtag/ICLR2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICLR2021</a><br><br>Paper: <a href="https://t.co/cZezBq9c8d">https://t.co/cZezBq9c8d</a> <a href="https://t.co/IgEr515qRn">pic.twitter.com/IgEr515qRn</a></p>&mdash; Hieu Pham (@hieupham789) <a href="https://twitter.com/hieupham789/status/1361886796930424838?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Meta Back-translation<br>pdf: <a href="https://t.co/x3Jf7vHGVL">https://t.co/x3Jf7vHGVL</a><br>abs: <a href="https://t.co/uKTstM9xQb">https://t.co/uKTstM9xQb</a> <a href="https://t.co/X9GORon6P2">pic.twitter.com/X9GORon6P2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361881852517171201?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Non-Autoregressive Text Generation with Pre-trained Language Models

Yixuan Su, Deng Cai, Yan Wang, David Vandyke, Simon Baker, Piji Li, Nigel Collier

- retweets: 197, favorites: 54 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.08220) | [pdf](https://arxiv.org/pdf/2102.08220)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Non-autoregressive generation (NAG) has recently attracted great attention due to its fast inference speed. However, the generation quality of existing NAG models still lags behind their autoregressive counterparts. In this work, we show that BERT can be employed as the backbone of a NAG model to greatly improve performance. Additionally, we devise mechanisms to alleviate the two common problems of vanilla NAG models: the inflexibility of prefixed output length and the conditional independence of individual token predictions. Lastly, to further increase the speed advantage of the proposed model, we propose a new decoding strategy, ratio-first, for applications where the output lengths can be approximately estimated beforehand. For a comprehensive evaluation, we test the proposed model on three text generation tasks, including text summarization, sentence compression and machine translation. Experimental results show that our model significantly outperforms existing non-autoregressive baselines and achieves competitive performance with many strong autoregressive models. In addition, we also conduct extensive analysis experiments to reveal the effect of each proposed component.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Non-Autoregressive Text Generation with Pre-trained Language Models<br>pdf: <a href="https://t.co/VkWfTu29yR">https://t.co/VkWfTu29yR</a><br>abs: <a href="https://t.co/npZCCkohfP">https://t.co/npZCCkohfP</a> <a href="https://t.co/koaC0p21DD">pic.twitter.com/koaC0p21DD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361859528527122437?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Online hyperparameter optimization by real-time recurrent learning

Daniel Jiwoong Im, Cristina Savin, Kyunghyun Cho

- retweets: 98, favorites: 117 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.07813) | [pdf](https://arxiv.org/pdf/2102.07813)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Conventional hyperparameter optimization methods are computationally intensive and hard to generalize to scenarios that require dynamically adapting hyperparameters, such as life-long learning. Here, we propose an online hyperparameter optimization algorithm that is asymptotically exact and computationally tractable, both theoretically and practically. Our framework takes advantage of the analogy between hyperparameter optimization and parameter learning in recurrent neural networks (RNNs). It adapts a well-studied family of online learning algorithms for RNNs to tune hyperparameters and network parameters simultaneously, without repeatedly rolling out iterative optimization. This procedure yields systematically better generalization performance compared to standard methods, at a fraction of wallclock time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">here goes the preprint: <a href="https://t.co/BknYEkU3yk">https://t.co/BknYEkU3yk</a> <a href="https://t.co/pusruWhOEt">https://t.co/pusruWhOEt</a></p>&mdash; Kyunghyun Cho (@kchonyc) <a href="https://twitter.com/kchonyc/status/1361912669285060610?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">**Online hyperparameter optimization by real-time recurrent learning**<br><br>We tune hyperparameters and network parameters simultaneously, without repeatedly rolling out iterative optimization. <a href="https://t.co/ebjRZnp7Lw">https://t.co/ebjRZnp7Lw</a> <a href="https://t.co/Avexwdt8fZ">https://t.co/Avexwdt8fZ</a><br>Joint work  w. Cristina Savin <a href="https://twitter.com/kchonyc?ref_src=twsrc%5Etfw">@kchonyc</a> <a href="https://t.co/i9JYv1Q56y">pic.twitter.com/i9JYv1Q56y</a></p>&mdash; Daniel Jiwoong Im (@Daniel_J_Im) <a href="https://twitter.com/Daniel_J_Im/status/1362067276200087556?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Controlling False Discovery Rates Using Null Bootstrapping

Junpei Komiyama, Masaya Abe, Kei Nakagawa, Kenichiro McAlinn

- retweets: 88, favorites: 120 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.07826) | [pdf](https://arxiv.org/pdf/2102.07826)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We consider controlling the false discovery rate for many tests with unknown correlation structure. Given a large number of hypotheses, false and missing discoveries can plague an analysis. While many procedures have been proposed to control false discovery, they either assume independent hypotheses or lack statistical power. We propose a novel method for false discovery control using null bootstrapping. By bootstrapping from the correlated null, we achieve superior statistical power to existing methods and prove that the false discovery rate is controlled. Simulated examples illustrate the efficacy of our method over existing methods. We apply our proposed methodology to financial asset pricing, where the goal is to determine which "factors" lead to excess returns out of a large number of potential factors.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">多重検定の論文を書きました (w/ 野村AMの阿部さん、中川さん、Temple Uのマクリンさん）。帰無仮説からサンプルできる場合、Storey法よりfalse discoveryに強い手法です。  <a href="https://t.co/o5W1JTLfZp">https://t.co/o5W1JTLfZp</a></p>&mdash; Junpei Komiyama (@jkomiyama_) <a href="https://twitter.com/jkomiyama_/status/1361880955124932608?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. ReGraphX: NoC-enabled 3D Heterogeneous ReRAM Architecture for Training  Graph Neural Networks

Aqeeb Iqbal Arka, Biresh Kumar Joardar, Janardhan Rao Doppa, Partha Pratim Pande, Krishnendu Chakrabarty

- retweets: 156, favorites: 35 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.07959) | [pdf](https://arxiv.org/pdf/2102.07959)
- [cs.AR](https://arxiv.org/list/cs.AR/recent) | [cs.ET](https://arxiv.org/list/cs.ET/recent)

Graph Neural Network (GNN) is a variant of Deep Neural Networks (DNNs) operating on graphs. However, GNNs are more complex compared to traditional DNNs as they simultaneously exhibit features of both DNN and graph applications. As a result, architectures specifically optimized for either DNNs or graph applications are not suited for GNN training. In this work, we propose a 3D heterogeneous manycore architecture for on-chip GNN training to address this problem. The proposed architecture, ReGraphX, involves heterogeneous ReRAM crossbars to fulfill the disparate requirements of both DNN and graph computations simultaneously. The ReRAM-based architecture is complemented with a multicast-enabled 3D NoC to improve the overall achievable performance. We demonstrate that ReGraphX outperforms conventional GPUs by up to 3.5X (on an average 3X) in terms of execution time, while reducing energy consumption by as much as 11X.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper is proposed ReGraphX, a heterogeneous ReRAM-based manycore architecture enabled by 3D NoC for training GNNs, which outperforms conventional GPUs by up to 3.5x in terms of execution time, while reducing energy consumption by as much as 11x.<a href="https://t.co/XNAfELFxCb">https://t.co/XNAfELFxCb</a> <a href="https://t.co/ftkznXitaB">pic.twitter.com/ftkznXitaB</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1361910791499300864?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. AlphaNet: Improved Training of Supernet with Alpha-Divergence

Dilin Wang, Chengyue Gong, Meng Li, Qiang Liu, Vikas Chandra

- retweets: 102, favorites: 50 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.07954) | [pdf](https://arxiv.org/pdf/2102.07954)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Weight-sharing neural architecture search (NAS) is an effective technique for automating efficient neural architecture design. Weight-sharing NAS builds a supernet that assembles all the architectures as its sub-networks and jointly trains the supernet with the sub-networks. The success of weight-sharing NAS heavily relies on distilling the knowledge of the supernet to the sub-networks. However, we find that the widely used distillation divergence, i.e., KL divergence, may lead to student sub-networks that over-estimate or under-estimate the uncertainty of the teacher supernet, leading to inferior performance of the sub-networks. In this work, we propose to improve the supernet training with a more generalized alpha-divergence. By adaptively selecting the alpha-divergence, we simultaneously prevent the over-estimation or under-estimation of the uncertainty of the teacher model. We apply the proposed alpha-divergence based supernet training to both slimmable neural networks and weight-sharing NAS, and demonstrate significant improvements. Specifically, our discovered model family, AlphaNet, outperforms prior-art models on a wide range of FLOPs regimes, including BigNAS, Once-for-All networks, FBNetV3, and AttentiveNAS. We achieve ImageNet top-1 accuracy of 80.0% with only 444 MFLOPs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AlphaNet: Improved Training of Supernet with Alpha-Divergence<br>pdf: <a href="https://t.co/myT0mn2AAV">https://t.co/myT0mn2AAV</a><br>abs: <a href="https://t.co/J2o7u2DM1l">https://t.co/J2o7u2DM1l</a> <a href="https://t.co/3HonH3Avdv">pic.twitter.com/3HonH3Avdv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361894540156801025?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Fast Validated Byzantine Broadcast

Ittai Abraham, Kartik Nayak, Ling Ren, Zhuolun Xiang

- retweets: 74, favorites: 34 (02/18/2021 09:33:21)

- links: [abs](https://arxiv.org/abs/2102.07932) | [pdf](https://arxiv.org/pdf/2102.07932)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent)

Byzantine fault-tolerant (BFT) state machine replication (SMR) has been studied for over 30 years. Recently it has received more attention due to its application in permissioned blockchain systems. A sequence of research efforts focuses on improving the commit latency of the SMR protocol in the common good case, including PBFT with $3$-round latency and $n\geq 3f+1$ and FaB with $2$-round latency and $n\geq 5f+1$. In this paper, we abstract a single-shot BFT SMR with a new broadcast formulation named partially synchronous validated Byzantine broadcast (psync-VBB), and propose a $2$-round psync-VBB protocol under the optimal resilience $n\geq 5f-1$ with a matching lower bound. Our protocol solves $2$-round BFT SMR with only $n\geq 5f-1$ replicas, which refutes the optimal resiliency claim made in FaB for needing $n \geq 5f+1$ for $2$-round PBFT-style BFT protocols. For the special case when $f=1$, our protocol needs only $4$ replicas, and strictly improves PBFT by reducing the latency by one round (even when one backup is faulty).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Companion work with lead author Zhuolun Xiang along with <a href="https://twitter.com/kartik1507?ref_src=twsrc%5Etfw">@kartik1507</a> and Ling Ren<br><br>Fast Validated Byzantine Broadcast <a href="https://t.co/F3uCBNANog">https://t.co/F3uCBNANog</a><br><br>TLDR; with n=4 we strictly improve PBFT from 3 phases to 2 phases, and show this is tight!</p>&mdash; Ittai Abraham (@ittaia) <a href="https://twitter.com/ittaia/status/1361996561199362050?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Quantifying environment and population diversity in multi-agent  reinforcement learning

Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, Richard Everett

- retweets: 64, favorites: 30 (02/18/2021 09:33:22)

- links: [abs](https://arxiv.org/abs/2102.08370) | [pdf](https://arxiv.org/pdf/2102.08370)
- [cs.MA](https://arxiv.org/list/cs.MA/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Generalization is a major challenge for multi-agent reinforcement learning. How well does an agent perform when placed in novel environments and in interactions with new co-players? In this paper, we investigate and quantify the relationship between generalization and diversity in the multi-agent domain. Across the range of multi-agent environments considered here, procedurally generating training levels significantly improves agent performance on held-out levels. However, agent performance on the specific levels used in training sometimes declines as a result. To better understand the effects of co-player variation, our experiments introduce a new environment-agnostic measure of behavioral diversity. Results demonstrate that population size and intrinsic motivation are both effective methods of generating greater population diversity. In turn, training with a diverse set of co-players strengthens agent performance in some (but not all) cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Quantifying Environment and Population Diversity in Multi-Agent Reinforcement Learning<br>pdf: <a href="https://t.co/N7VGqBPcAs">https://t.co/N7VGqBPcAs</a><br>abs: <a href="https://t.co/AVgbf7Z8Fm">https://t.co/AVgbf7Z8Fm</a> <a href="https://t.co/zvFffKScm9">pic.twitter.com/zvFffKScm9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361902591525986305?ref_src=twsrc%5Etfw">February 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. What Do We Want From Explainable Artificial Intelligence (XAI)? -- A  Stakeholder Perspective on XAI and a Conceptual Model Guiding  Interdisciplinary XAI Research

Markus Langer, Daniel Oster, Timo Speith, Holger Hermanns, Lena Kästner, Eva Schmidt, Andreas Sesing, Kevin Baum

- retweets: 49, favorites: 17 (02/18/2021 09:33:22)

- links: [abs](https://arxiv.org/abs/2102.07817) | [pdf](https://arxiv.org/pdf/2102.07817)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

Previous research in Explainable Artificial Intelligence (XAI) suggests that a main aim of explainability approaches is to satisfy specific interests, goals, expectations, needs, and demands regarding artificial systems (we call these stakeholders' desiderata) in a variety of contexts. However, the literature on XAI is vast, spreads out across multiple largely disconnected disciplines, and it often remains unclear how explainability approaches are supposed to achieve the goal of satisfying stakeholders' desiderata. This paper discusses the main classes of stakeholders calling for explainability of artificial systems and reviews their desiderata. We provide a model that explicitly spells out the main concepts and relations necessary to consider and investigate when evaluating, adjusting, choosing, and developing explainability approaches that aim to satisfy stakeholders' desiderata. This model can serve researchers from the variety of different disciplines involved in XAI as a common ground. It emphasizes where there is interdisciplinary potential in the evaluation and the development of explainability approaches.



