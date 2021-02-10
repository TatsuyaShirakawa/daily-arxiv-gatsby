---
title: Hot Papers 2021-02-09
date: 2021-02-10T10:23:46.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-09"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-09"
socialImage: "/media/flying-marine.jpg"

---

# 1. Symbolic Behaviour in Artificial Intelligence

Adam Santoro, Andrew Lampinen, Kory Mathewson, Timothy Lillicrap, David Raposo

- retweets: 4164, favorites: 482 (02/10/2021 10:23:46)

- links: [abs](https://arxiv.org/abs/2102.03406) | [pdf](https://arxiv.org/pdf/2102.03406)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The ability to use symbols is the pinnacle of human intelligence, but has yet to be fully replicated in machines. Here we argue that the path towards symbolically fluent artificial intelligence (AI) begins with a reinterpretation of what symbols are, how they come to exist, and how a system behaves when it uses them. We begin by offering an interpretation of symbols as entities whose meaning is established by convention. But crucially, something is a symbol only for those who demonstrably and actively participate in this convention. We then outline how this interpretation thematically unifies the behavioural traits humans exhibit when they use symbols. This motivates our proposal that the field place a greater emphasis on symbolic behaviour rather than particular computational mechanisms inspired by more restrictive interpretations of symbols. Finally, we suggest that AI research explore social and cultural engagement as a tool to develop the cognitive machinery necessary for symbolic behaviour to emerge. This approach will allow for AI to interpret something as symbolic on its own rather than simply manipulate things that are only symbols to human onlookers, and thus will ultimately lead to AI with more human-like symbolic fluency.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What are symbols? Where do symbols come from? What behaviors demonstrate the ability to engage with symbols? How do the answers to these questions impact AI research? We argue for a new perspective on these issues  our preprint: <a href="https://t.co/BW3yf6tWkD">https://t.co/BW3yf6tWkD</a> Summary in thread: 1/6</p>&mdash; Andrew Lampinen (@AndrewLampinen) <a href="https://twitter.com/AndrewLampinen/status/1358968214185598977?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Symbolic Behaviour in Artificial Intelligence<br><br>“We suggest that AI research explore social and cultural engagement as a tool to develop the cognitive machinery necessary for symbolic behaviour to emerge.”<a href="https://t.co/hZZYQzdQwT">https://t.co/hZZYQzdQwT</a> <a href="https://t.co/63lWqkxEuw">https://t.co/63lWqkxEuw</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1358982900310417408?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Nyströmformer: A Nyström-Based Algorithm for Approximating  Self-Attention

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh

- retweets: 2732, favorites: 377 (02/10/2021 10:23:46)

- links: [abs](https://arxiv.org/abs/2102.03902) | [pdf](https://arxiv.org/pdf/2102.03902)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers have emerged as a powerful tool for a broad range of natural language processing tasks. A key component that drives the impressive performance of Transformers is the self-attention mechanism that encodes the influence or dependence of other tokens on each specific token. While beneficial, the quadratic complexity of self-attention on the input sequence length has limited its application to longer sequences -- a topic being actively studied in the community. To address this limitation, we propose Nystr\"omformer -- a model that exhibits favorable scalability as a function of sequence length. Our idea is based on adapting the Nystr\"om method to approximate standard self-attention with $O(n)$ complexity. The scalability of Nystr\"omformer enables application to longer sequences with thousands of tokens. We perform evaluations on multiple downstream tasks on the GLUE benchmark and IMDB reviews with standard sequence length, and find that our Nystr\"omformer performs comparably, or in a few cases, even slightly better, than standard Transformer. Our code is at https://github.com/mlpen/Nystromformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention<br><br>By adapting the Nyström method to approximate standard self-attention with O(n) complexity, they can train Transformers on longer sequences with 1000s of tokens with small compute.<a href="https://t.co/ecLcuweNnU">https://t.co/ecLcuweNnU</a> <a href="https://t.co/I28cKAkVaD">pic.twitter.com/I28cKAkVaD</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1359143438831079425?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="da" dir="ltr">Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention<br>pdf: <a href="https://t.co/BG4tUJ4MBJ">https://t.co/BG4tUJ4MBJ</a><br>abs: <a href="https://t.co/1KJxItlY39">https://t.co/1KJxItlY39</a><br>github: <a href="https://t.co/YtHwXlpkfY">https://t.co/YtHwXlpkfY</a> <a href="https://t.co/5lcDmdnbsB">pic.twitter.com/5lcDmdnbsB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358965767816040450?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">自己注意機構で系列長Nの時、計算量がO(N)である手法としてLinformer, Longformer, BIGBIRDが提案されている。Nystromformerは行列近似のNystrom法を適用。セグメント平均をランドマークに使い近似。既存線形時間手法より高性能で元のTransformerに匹敵する精度を達成 <a href="https://t.co/uME5LW10G0">https://t.co/uME5LW10G0</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1359273598582542337?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Fairness for Unobserved Characteristics: Insights from Technological  Impacts on Queer Communities

Nenad Tomasev, Kevin R. McKee, Jackie Kay, Shakir Mohamed

- retweets: 2083, favorites: 285 (02/10/2021 10:23:47)

- links: [abs](https://arxiv.org/abs/2102.04257) | [pdf](https://arxiv.org/pdf/2102.04257)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Advances in algorithmic fairness have largely omitted sexual orientation and gender identity. We explore queer concerns in privacy, censorship, language, online safety, health, and employment to study the positive and negative effects of artificial intelligence on queer communities. These issues underscore the need for new directions in fairness research that take into account a multiplicity of considerations, from privacy preservation, context sensitivity and process fairness, to an awareness of sociotechnical impact and the increasingly important role of inclusive and participatory research processes. Most current approaches for algorithmic fairness assume that the target characteristics for fairness--frequently, race and legal gender--can be observed or recorded. Sexual orientation and gender identity are prototypical instances of unobserved characteristics, which are frequently missing, unknown or fundamentally unmeasurable. This paper highlights the importance of developing new approaches for algorithmic fairness that break away from the prevailing assumption of observed characteristics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In a new paper, <a href="https://twitter.com/weballergy?ref_src=twsrc%5Etfw">@weballergy</a>, <a href="https://twitter.com/jackayline?ref_src=twsrc%5Etfw">@jackayline</a>, <a href="https://twitter.com/empiricallykev?ref_src=twsrc%5Etfw">@empiricallykev</a> &amp; <a href="https://twitter.com/shakir_za?ref_src=twsrc%5Etfw">@shakir_za</a> highlight the importance of developing new approaches for algorithmic fairness, exploring queer concerns like privacy and language to understand the effects of AI on queer communities: <a href="https://t.co/CLsdVbyEHM">https://t.co/CLsdVbyEHM</a></p>&mdash; DeepMind (@DeepMind) <a href="https://twitter.com/DeepMind/status/1359112968881995782?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fairness for Unobserved Characteristics: Insights from Technological Impacts on Queer Communities <a href="https://t.co/jG4ogA4oDx">https://t.co/jG4ogA4oDx</a> : highlighting some of the key challenges in ensuring algorithmic fairness for queer communities <a href="https://twitter.com/jackayline?ref_src=twsrc%5Etfw">@jackayline</a> <a href="https://twitter.com/empiricallykev?ref_src=twsrc%5Etfw">@empiricallykev</a> <a href="https://twitter.com/shakir_za?ref_src=twsrc%5Etfw">@Shakir_za</a></p>&mdash; Nenad Tomasev (@weballergy) <a href="https://twitter.com/weballergy/status/1359061877494599682?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Queer experience brings a unique insight into  considerations of algorithmic fairness. We were so happy to be able to build on work in this space in this paper 🎉🌟And hopefully we&#39;ll see much more research on queer fairness 🏳️‍🌈. <a href="https://t.co/VOz5eyh6Gv">https://t.co/VOz5eyh6Gv</a> Great 🧵 below <a href="https://t.co/i81n6uC5DD">https://t.co/i81n6uC5DD</a></p>&mdash; Shakir Mohamed (@shakir_za) <a href="https://twitter.com/shakir_za/status/1359065454912929795?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Colorization Transformer

Manoj Kumar, Dirk Weissenborn, Nal Kalchbrenner

- retweets: 576, favorites: 141 (02/10/2021 10:23:47)

- links: [abs](https://arxiv.org/abs/2102.04432) | [pdf](https://arxiv.org/pdf/2102.04432)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present the Colorization Transformer, a novel approach for diverse high fidelity image colorization based on self-attention. Given a grayscale image, the colorization proceeds in three steps. We first use a conditional autoregressive transformer to produce a low resolution coarse coloring of the grayscale image. Our architecture adopts conditional transformer layers to effectively condition grayscale input. Two subsequent fully parallel networks upsample the coarse colored low resolution image into a finely colored high resolution image. Sampling from the Colorization Transformer produces diverse colorings whose fidelity outperforms the previous state-of-the-art on colorising ImageNet based on FID results and based on a human evaluation in a Mechanical Turk test. Remarkably, in more than 60% of cases human evaluators prefer the highest rated among three generated colorings over the ground truth. The code and pre-trained checkpoints for Colorization Transformer are publicly available at https://github.com/google-research/google-research/tree/master/coltran

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Colorization Transformer<br>pdf: <a href="https://t.co/QrvdW2sZxJ">https://t.co/QrvdW2sZxJ</a><br>abs: <a href="https://t.co/D63vl4Sl7E">https://t.co/D63vl4Sl7E</a><br>github: <a href="https://t.co/HaEyCaQbxH">https://t.co/HaEyCaQbxH</a> <a href="https://t.co/QLa118WnDv">pic.twitter.com/QLa118WnDv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358971791876775936?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out Colorization Transformer: Our ICLR 2021 paper that explores conditional transformers for diverse image colorization.<br><br>Paper: <a href="https://t.co/4Jva6vBI5M">https://t.co/4Jva6vBI5M</a><br>Code: <a href="https://t.co/uKLVPaVogq">https://t.co/uKLVPaVogq</a><br>Openreview: <a href="https://t.co/K7IckMQvIR">https://t.co/K7IckMQvIR</a><br><br>Work w/ <a href="https://twitter.com/NalKalchbrenner?ref_src=twsrc%5Etfw">@NalKalchbrenner</a> &amp; Dirk Weissenborn <a href="https://t.co/5sNeW6QDly">pic.twitter.com/5sNeW6QDly</a></p>&mdash; mechcoder (@mechcoder) <a href="https://twitter.com/mechcoder/status/1359145505666830342?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. HumanACGAN: conditional generative adversarial network with human-based  auxiliary classifier and its evaluation in phoneme perception

Yota Ueda, Kazuki Fujii, Yuki Saito, Shinnosuke Takamichi, Yukino Baba, Hiroshi Saruwatari

- retweets: 462, favorites: 44 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.04051) | [pdf](https://arxiv.org/pdf/2102.04051)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose a conditional generative adversarial network (GAN) incorporating humans' perceptual evaluations. A deep neural network (DNN)-based generator of a GAN can represent a real-data distribution accurately but can never represent a human-acceptable distribution, which are ranges of data in which humans accept the naturalness regardless of whether the data are real or not. A HumanGAN was proposed to model the human-acceptable distribution. A DNN-based generator is trained using a human-based discriminator, i.e., humans' perceptual evaluations, instead of the GAN's DNN-based discriminator. However, the HumanGAN cannot represent conditional distributions. This paper proposes the HumanACGAN, a theoretical extension of the HumanGAN, to deal with conditional human-acceptable distributions. Our HumanACGAN trains a DNN-based conditional generator by regarding humans as not only a discriminator but also an auxiliary classifier. The generator is trained by deceiving the human-based discriminator that scores the unconditioned naturalness and the human-based classifier that scores the class-conditioned perceptual acceptability. The training can be executed using the backpropagation algorithm involving humans' perceptual evaluations. Our experimental results in phoneme perception demonstrate that our HumanACGAN can successfully train this conditional generator.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper is out! Human is acting as discriminator and classifier functions in GAN!<br><br>HumanACGAN: conditional generative adversarial network with human-based auxiliary classifier and its evaluation in phoneme perception<a href="https://t.co/fO0JNyKttI">https://t.co/fO0JNyKttI</a> <a href="https://t.co/hB4DgXm0Vd">pic.twitter.com/hB4DgXm0Vd</a></p>&mdash; Shinnosuke Takamichi (高道 慎之介) (@forthshinji) <a href="https://twitter.com/forthshinji/status/1358970018231320578?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Solid Texture Synthesis using Generative Adversarial Networks

Xin Zhao, Lin Wang, Jifeng Guo, Bo Yang, Junteng Zheng, Fanqi Li

- retweets: 288, favorites: 87 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.03973) | [pdf](https://arxiv.org/pdf/2102.03973)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Solid texture synthesis, as an effective way to extend 2D texture to 3D solid texture, exhibits advantages in numerous application domains. However, existing methods generally suffer from synthesis distortion due to the underutilization of texture information. In this paper, we proposed a novel neural network-based approach for the solid texture synthesis based on generative adversarial networks, namely STS-GAN, in which the generator composed of multi-scale modules learns the internal distribution of 2D exemplar and further extends it to a 3D solid texture. In addition, the discriminator evaluates the similarity between 2D exemplar and slices, promoting the generator to synthesize realistic solid texture. Experiment results demonstrate that the proposed method can synthesize high-quality 3D solid texture with similar visual characteristics to the exemplar.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Solid Texture Synthesis using Generative Adversarial Networks<br>pdf: <a href="https://t.co/5R8r2GW3IE">https://t.co/5R8r2GW3IE</a><br>abs: <a href="https://t.co/VsYqzSthOh">https://t.co/VsYqzSthOh</a> <a href="https://t.co/lY80K7O7YG">pic.twitter.com/lY80K7O7YG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358975772078276610?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Grid-to-Graph: Flexible Spatial Relational Inductive Biases for  Reinforcement Learning

Zhengyao Jiang, Pasquale Minervini, Minqi Jiang, Tim Rocktaschel

- retweets: 196, favorites: 59 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.04220) | [pdf](https://arxiv.org/pdf/2102.04220)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Although reinforcement learning has been successfully applied in many domains in recent years, we still lack agents that can systematically generalize. While relational inductive biases that fit a task can improve generalization of RL agents, these biases are commonly hard-coded directly in the agent's neural architecture. In this work, we show that we can incorporate relational inductive biases, encoded in the form of relational graphs, into agents. Based on this insight, we propose Grid-to-Graph (GTG), a mapping from grid structures to relational graphs that carry useful spatial relational inductive biases when processed through a Relational Graph Convolution Network (R-GCN). We show that, with GTG, R-GCNs generalize better both in terms of in-distribution and out-of-distribution compared to baselines based on Convolutional Neural Networks and Neural Logic Machines on challenging procedurally generated environments and MinAtar. Furthermore, we show that GTG produces agents that can jointly reason over observations and environment dynamics encoded in knowledge bases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper &quot;Grid-to-Graph: Flexible Spatial Relational Inductive Biases for Reinforcement Learning&quot; (AAMAS 2021) is now available on Arxiv: <a href="https://t.co/od5OsgEWK1">https://t.co/od5OsgEWK1</a><br>Many thanks to amazing collaborators and my supervisor! <a href="https://twitter.com/PMinervini?ref_src=twsrc%5Etfw">@PMinervini</a> <a href="https://twitter.com/MinqiJiang?ref_src=twsrc%5Etfw">@MinqiJiang</a> <a href="https://twitter.com/_rockt?ref_src=twsrc%5Etfw">@_rockt</a></p>&mdash; Zhengyao Jiang (@zhengyaojiang) <a href="https://twitter.com/zhengyaojiang/status/1359069944789532673?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Learning Curve Theory

Marcus Hutter

- retweets: 101, favorites: 107 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.04074) | [pdf](https://arxiv.org/pdf/2102.04074)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recently a number of empirical "universal" scaling law papers have been published, most notably by OpenAI. `Scaling laws' refers to power-law decreases of training or test error w.r.t. more data, larger neural networks, and/or more compute. In this work we focus on scaling w.r.t. data size $n$. Theoretical understanding of this phenomenon is largely lacking, except in finite-dimensional models for which error typically decreases with $n^{-1/2}$ or $n^{-1}$, where $n$ is the sample size. We develop and theoretically analyse the simplest possible (toy) model that can exhibit $n^{-\beta}$ learning curves for arbitrary power $\beta>0$, and determine whether power laws are universal or depend on the data distribution.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">what a nice exposition by Marcus Hutter on the learning curve theory: <a href="https://t.co/Vq37XJJ0Aw">https://t.co/Vq37XJJ0Aw</a>! very nice read.</p>&mdash; Kyunghyun Cho (@kchonyc) <a href="https://twitter.com/kchonyc/status/1358978252073795586?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Curve Theory<br>pdf: <a href="https://t.co/Fdh8NlMkts">https://t.co/Fdh8NlMkts</a><br>abs: <a href="https://t.co/lU3YOdhDjZ">https://t.co/lU3YOdhDjZ</a> <a href="https://t.co/rLElXsrjCD">pic.twitter.com/rLElXsrjCD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358962145673826311?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch

Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, Hongsheng Li

- retweets: 169, favorites: 24 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.04010) | [pdf](https://arxiv.org/pdf/2102.04010)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent)

Sparsity in Deep Neural Networks (DNNs) has been widely studied to compress and accelerate the models on resource-constrained environments. It can be generally categorized into unstructured fine-grained sparsity that zeroes out multiple individual weights distributed across the neural network, and structured coarse-grained sparsity which prunes blocks of sub-networks of a neural network. Fine-grained sparsity can achieve a high compression ratio but is not hardware friendly and hence receives limited speed gains. On the other hand, coarse-grained sparsity cannot concurrently achieve both apparent acceleration on modern GPUs and decent performance. In this paper, we are the first to study training from scratch an N:M fine-grained structured sparse network, which can maintain the advantages of both unstructured fine-grained sparsity and structured coarse-grained sparsity simultaneously on specifically designed GPUs. Specifically, a 2:4 sparse network could achieve 2x speed-up without performance drop on Nvidia A100 GPUs. Furthermore, we propose a novel and effective ingredient, sparse-refined straight-through estimator (SR-STE), to alleviate the negative influence of the approximated gradients computed by vanilla STE during optimization. We also define a metric, Sparse Architecture Divergence (SAD), to measure the sparse network's topology change during the training process. Finally, We justify SR-STE's advantages with SAD and demonstrate the effectiveness of SR-STE by performing comprehensive experiments on various tasks. Source codes and models are available at https://github.com/NM-sparsity/NM-sparsity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch<br>pdf: <a href="https://t.co/JqHMU8nzn9">https://t.co/JqHMU8nzn9</a><br>abs: <a href="https://t.co/xJR4FGEzt9">https://t.co/xJR4FGEzt9</a><br>github: <a href="https://t.co/dyDetrSuwj">https://t.co/dyDetrSuwj</a> <a href="https://t.co/4jnTXfAg6x">pic.twitter.com/4jnTXfAg6x</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1359017914406477825?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Neural SDEs as Infinite-Dimensional GANs

Patrick Kidger, James Foster, Xuechen Li, Harald Oberhauser, Terry Lyons

- retweets: 100, favorites: 62 (02/10/2021 10:23:48)

- links: [abs](https://arxiv.org/abs/2102.03657) | [pdf](https://arxiv.org/pdf/2102.03657)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Stochastic differential equations (SDEs) are a staple of mathematical modelling of temporal dynamics. However, a fundamental limitation has been that such models have typically been relatively inflexible, which recent work introducing Neural SDEs has sought to solve. Here, we show that the current classical approach to fitting SDEs may be approached as a special case of (Wasserstein) GANs, and in doing so the neural and classical regimes may be brought together. The input noise is Brownian motion, the output samples are time-evolving paths produced by a numerical solver, and by parameterising a discriminator as a Neural Controlled Differential Equation (CDE), we obtain Neural SDEs as (in modern machine learning parlance) continuous-time generative time series models. Unlike previous work on this problem, this is a direct extension of the classical approach without reference to either prespecified statistics or density functions. Arbitrary drift and diffusions are admissible, so as the Wasserstein loss has a unique global minima, in the infinite data limit \textit{any} SDE may be learnt.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Neural SDEs as Infinite-Dimensional GANs<br>pdf: <a href="https://t.co/9hXFQWMNur">https://t.co/9hXFQWMNur</a><br>abs: <a href="https://t.co/XFTHVtNzOW">https://t.co/XFTHVtNzOW</a> <a href="https://t.co/eE1Y4urWBB">pic.twitter.com/eE1Y4urWBB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358973923942146051?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. End-to-End Multi-Channel Transformer for Speech Recognition

Feng-Ju Chang, Martin Radfar, Athanasios Mouchtaris, Brian King, Siegfried Kunzmann

- retweets: 81, favorites: 42 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.03951) | [pdf](https://arxiv.org/pdf/2102.03951)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Transformers are powerful neural architectures that allow integrating different modalities using attention mechanisms. In this paper, we leverage the neural transformer architectures for multi-channel speech recognition systems, where the spectral and spatial information collected from different microphones are integrated using attention layers. Our multi-channel transformer network mainly consists of three parts: channel-wise self attention layers (CSA), cross-channel attention layers (CCA), and multi-channel encoder-decoder attention layers (EDA). The CSA and CCA layers encode the contextual relationship within and between channels and across time, respectively. The channel-attended outputs from CSA and CCA are then fed into the EDA layers to help decode the next token given the preceding ones. The experiments show that in a far-field in-house dataset, our method outperforms the baseline single-channel transformer, as well as the super-directive and neural beamformers cascaded with the transformers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-End Multi-Channel Transformer for Speech Recognition<br>pdf: <a href="https://t.co/FxSiuu7jYD">https://t.co/FxSiuu7jYD</a><br>abs: <a href="https://t.co/NFhD7675HC">https://t.co/NFhD7675HC</a> <a href="https://t.co/NCLISRLToM">pic.twitter.com/NCLISRLToM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358972623934394368?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Template-Free Try-on Image Synthesis via Semantic-guided Optimization

Chien-Lung Chou, Chieh-Yun Chen, Chia-Wei Hsieh, Hong-Han Shuai, Jiaying Liu, Wen-Huang Cheng

- retweets: 49, favorites: 45 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.03503) | [pdf](https://arxiv.org/pdf/2102.03503)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The virtual try-on task is so attractive that it has drawn considerable attention in the field of computer vision. However, presenting the three-dimensional (3D) physical characteristic (e.g., pleat and shadow) based on a 2D image is very challenging. Although there have been several previous studies on 2D-based virtual try-on work, most 1) required user-specified target poses that are not user-friendly and may not be the best for the target clothing, and 2) failed to address some problematic cases, including facial details, clothing wrinkles and body occlusions. To address these two challenges, in this paper, we propose an innovative template-free try-on image synthesis (TF-TIS) network. The TF-TIS first synthesizes the target pose according to the user-specified in-shop clothing. Afterward, given an in-shop clothing image, a user image, and a synthesized pose, we propose a novel model for synthesizing a human try-on image with the target clothing in the best fitting pose. The qualitative and quantitative experiments both indicate that the proposed TF-TIS outperforms the state-of-the-art methods, especially for difficult cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Template-Free Try-on Image Synthesis via Semantic-guided Optimization<br>pdf: <a href="https://t.co/bAAZ45Xv6Y">https://t.co/bAAZ45Xv6Y</a><br>abs: <a href="https://t.co/m2skd6hDh3">https://t.co/m2skd6hDh3</a> <a href="https://t.co/Yed3o5L0kX">pic.twitter.com/Yed3o5L0kX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358979776170647554?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. LightSpeech: Lightweight and Fast Text to Speech with Neural  Architecture Search

Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Jinzhu Li, Sheng Zhao, Enhong Chen, Tie-Yan Liu

- retweets: 56, favorites: 36 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.04040) | [pdf](https://arxiv.org/pdf/2102.04040)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Text to speech (TTS) has been broadly used to synthesize natural and intelligible speech in different scenarios. Deploying TTS in various end devices such as mobile phones or embedded devices requires extremely small memory usage and inference latency. While non-autoregressive TTS models such as FastSpeech have achieved significantly faster inference speed than autoregressive models, their model size and inference latency are still large for the deployment in resource constrained devices. In this paper, we propose LightSpeech, which leverages neural architecture search~(NAS) to automatically design more lightweight and efficient models based on FastSpeech. We first profile the components of current FastSpeech model and carefully design a novel search space containing various lightweight and potentially effective architectures. Then NAS is utilized to automatically discover well performing architectures within the search space. Experiments show that the model discovered by our method achieves 15x model compression ratio and 6.5x inference speedup on CPU with on par voice quality. Audio demos are provided at https://speechresearch.github.io/lightspeech.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search<br>pdf: <a href="https://t.co/oFmqF8NcJl">https://t.co/oFmqF8NcJl</a><br>abs: <a href="https://t.co/YzymNRhw76">https://t.co/YzymNRhw76</a><br>project page: <a href="https://t.co/qs8LnT2TeM">https://t.co/qs8LnT2TeM</a> <a href="https://t.co/EUJRXpQeNz">pic.twitter.com/EUJRXpQeNz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1358983000747343875?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Towards a mathematical framework to inform Neural Network modelling via  Polynomial Regression

Pablo Morala, Jenny Alexandra Cifuentes, Rosa E. Lillo, Iñaki Ucar

- retweets: 47, favorites: 22 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.03865) | [pdf](https://arxiv.org/pdf/2102.03865)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Even when neural networks are widely used in a large number of applications, they are still considered as black boxes and present some difficulties for dimensioning or evaluating their prediction error. This has led to an increasing interest in the overlapping area between neural networks and more traditional statistical methods, which can help overcome those problems. In this article, a mathematical framework relating neural networks and polynomial regression is explored by building an explicit expression for the coefficients of a polynomial regression from the weights of a given neural network, using a Taylor expansion approach. This is achieved for single hidden layer neural networks in regression problems. The validity of the proposed method depends on different factors like the distribution of the synaptic potentials or the chosen activation function. The performance of this method is empirically tested via simulation of synthetic data generated from polynomials to train neural networks with different structures and hyperparameters, showing that almost identical predictions can be obtained when certain conditions are met. Lastly, when learning from polynomial generated data, the proposed method produces polynomials that approximate correctly the data locally.




# 15. An Efficient Framework for Zero-Shot Sketch-Based Image Retrieval

Osman Tursun, Simon Denman, Sridha Sridharan, Ethan Goan, Clinton Fookes

- retweets: 42, favorites: 24 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.04016) | [pdf](https://arxiv.org/pdf/2102.04016)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, Zero-shot Sketch-based Image Retrieval (ZS-SBIR) has attracted the attention of the computer vision community due to it's real-world applications, and the more realistic and challenging setting than found in SBIR. ZS-SBIR inherits the main challenges of multiple computer vision problems including content-based Image Retrieval (CBIR), zero-shot learning and domain adaptation. The majority of previous studies using deep neural networks have achieved improved results through either projecting sketch and images into a common low-dimensional space or transferring knowledge from seen to unseen classes. However, those approaches are trained with complex frameworks composed of multiple deep convolutional neural networks (CNNs) and are dependent on category-level word labels. This increases the requirements on training resources and datasets. In comparison, we propose a simple and efficient framework that does not require high computational training resources, and can be trained on datasets without semantic categorical labels. Furthermore, at training and inference stages our method only uses a single CNN. In this work, a pre-trained ImageNet CNN (e.g., ResNet50) is fine-tuned with three proposed learning objects: domain-aware quadruplet loss, semantic classification loss, and semantic knowledge preservation loss. The domain-aware quadruplet and semantic classification losses are introduced to learn discriminative, semantic and domain invariant features through considering ZS-SBIR as object detection and verification problem. ...

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Efficient Framework for Zero-Shot Sketch-Based Image Retrieval<br>pdf: <a href="https://t.co/gXJJK7X578">https://t.co/gXJJK7X578</a><br>abs: <a href="https://t.co/WDluI3JJ0S">https://t.co/WDluI3JJ0S</a> <a href="https://t.co/vZ9EA95KHW">pic.twitter.com/vZ9EA95KHW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1359018946326900738?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Unlocking Pixels for Reinforcement Learning via Implicit Attention

Krzysztof Choromanski, Deepali Jain, Jack Parker-Holder, Xingyou Song, Valerii Likhosherstov, Anirban Santara, Aldo Pacchiano, Yunhao Tang, Adrian Weller

- retweets: 49, favorites: 17 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.04353) | [pdf](https://arxiv.org/pdf/2102.04353)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

There has recently been significant interest in training reinforcement learning (RL) agents in vision-based environments. This poses many challenges, such as high dimensionality and potential for observational overfitting through spurious correlations. A promising approach to solve both of these problems is a self-attention bottleneck, which provides a simple and effective framework for learning high performing policies, even in the presence of distractions. However, due to poor scalability of attention architectures, these methods do not scale beyond low resolution visual inputs, using large patches (thus small attention matrices). In this paper we make use of new efficient attention algorithms, recently shown to be highly effective for Transformers, and demonstrate that these new techniques can be applied in the RL setting. This allows our attention-based controllers to scale to larger visual inputs, and facilitate the use of smaller patches, even individual pixels, improving generalization. In addition, we propose a new efficient algorithm approximating softmax attention with what we call hybrid random features, leveraging the theory of angular kernels. We show theoretically and empirically that hybrid random features is a promising approach when using attention for vision-based RL.




# 17. The Implicit Biases of Stochastic Gradient Descent on Deep Neural  Networks with Batch Normalization

Ziquan Liu, Yufei Cui, Jia Wan, Yu Mao, Antoni B. Chan

- retweets: 36, favorites: 22 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.03497) | [pdf](https://arxiv.org/pdf/2102.03497)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Deep neural networks with batch normalization (BN-DNNs) are invariant to weight rescaling due to their normalization operations. However, using weight decay (WD) benefits these weight-scale-invariant networks, which is often attributed to an increase of the effective learning rate when the weight norms are decreased. In this paper, we demonstrate the insufficiency of the previous explanation and investigate the implicit biases of stochastic gradient descent (SGD) on BN-DNNs to provide a theoretical explanation for the efficacy of weight decay. We identity two implicit biases of SGD on BN-DNNs: 1) the weight norms in SGD training remain constant in the continuous-time domain and keep increasing in the discrete-time domain; 2) SGD optimizes weight vectors in fully-connected networks or convolution kernels in convolution neural networks by updating components lying in the input feature span, while leaving those components orthogonal to the input feature span unchanged. Thus, SGD without WD accumulates weight noise orthogonal to the input feature span, and cannot eliminate such noise. Our empirical studies corroborate the hypothesis that weight decay suppresses weight noise that is left untouched by SGD. Furthermore, we propose to use weight rescaling (WRS) instead of weight decay to achieve the same regularization effect, while avoiding performance degradation of WD on some momentum-based optimizers. Our empirical results on image recognition show that regardless of optimization methods and network architectures, training BN-DNNs using WRS achieves similar or better performance compared with using WD. We also show that training with WRS generalizes better compared to WD, on other computer vision tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Implicit Biases of Stochastic Gradient Descent on Deep Neural Networks with Batch Normalization. (arXiv:2102.03497v1 [cs.LG]) <a href="https://t.co/GuJDd0UmUw">https://t.co/GuJDd0UmUw</a></p>&mdash; Stat.ML Papers (@StatMLPapers) <a href="https://twitter.com/StatMLPapers/status/1359061330624471041?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. From Toxicity in Online Comments to Incivility in American News: Proceed  with Caution

Anushree Hede, Oshin Agarwal, Linda Lu, Diana C. Mutz, Ani Nenkova

- retweets: 12, favorites: 43 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.03671) | [pdf](https://arxiv.org/pdf/2102.03671)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The ability to quantify incivility online, in news and in congressional debates, is of great interest to political scientists. Computational tools for detecting online incivility for English are now fairly accessible and potentially could be applied more broadly. We test the Jigsaw Perspective API for its ability to detect the degree of incivility on a corpus that we developed, consisting of manual annotations of civility in American news. We demonstrate that toxicity models, as exemplified by Perspective, are inadequate for the analysis of incivility in news. We carry out error analysis that points to the need to develop methods to remove spurious correlations between words often mentioned in the news, especially identity descriptors and incivility. Without such improvements, applying Perspective or similar models on news is likely to lead to wrong conclusions, that are not aligned with the human perception of incivility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our EACL’21 paper is on arXiv<br><br>We ask if models for predicting toxicity in online comments, like Perspective, can be used to quantify incivility in American news.<br><br>The answer is NO.<br><br>⁦with <a href="https://twitter.com/AnushreeHede?ref_src=twsrc%5Etfw">@AnushreeHede</a>⁩  ⁦<a href="https://twitter.com/agarwal_oshin?ref_src=twsrc%5Etfw">@agarwal_oshin</a>⁩ Linda Lu, Diana Mutz <a href="https://t.co/JoA2e5PLIR">https://t.co/JoA2e5PLIR</a></p>&mdash; Ani Nenkova (@ani_nenkova) <a href="https://twitter.com/ani_nenkova/status/1359154922508472323?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Functional Space Analysis of Local GAN Convergence

Valentin Khrulkov, Artem Babenko, Ivan Oseledets

- retweets: 30, favorites: 23 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.04448) | [pdf](https://arxiv.org/pdf/2102.04448)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent work demonstrated the benefits of studying continuous-time dynamics governing the GAN training. However, this dynamics is analyzed in the model parameter space, which results in finite-dimensional dynamical systems. We propose a novel perspective where we study the local dynamics of adversarial training in the general functional space and show how it can be represented as a system of partial differential equations. Thus, the convergence properties can be inferred from the eigenvalues of the resulting differential operator. We show that these eigenvalues can be efficiently estimated from the target dataset before training. Our perspective reveals several insights on the practical tricks commonly used to stabilize GANs, such as gradient penalty, data augmentation, and advanced integration schemes. As an immediate practical benefit, we demonstrate how one can a priori select an optimal data augmentation strategy for a particular generation task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/n Analyzing GAN convergence is hard. We derived exact rates and oscillations providing complete analysis from Poincare constant and weighted Laplacian. Check our new paper <a href="https://t.co/gxrowco5WQ">https://t.co/gxrowco5WQ</a> w <a href="https://twitter.com/vforvalya1?ref_src=twsrc%5Etfw">@vforvalya1</a> and Artem Babenko.</p>&mdash; Ivan Oseledets (@oseledetsivan) <a href="https://twitter.com/oseledetsivan/status/1359004813942525953?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. Improving Artificial Teachers by Considering How People Learn and Forget

Aurélien Nioche, Pierre-Alexandre Murena, Carlos de la Torre-Ortiz, Antti Oulasvirta

- retweets: 20, favorites: 31 (02/10/2021 10:23:49)

- links: [abs](https://arxiv.org/abs/2102.04174) | [pdf](https://arxiv.org/pdf/2102.04174)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

The paper presents a novel model-based method for intelligent tutoring, with particular emphasis on the problem of selecting teaching interventions in interaction with humans. Whereas previous work has focused on either personalization of teaching or optimization of teaching intervention sequences, the proposed individualized model-based planning approach represents convergence of these two lines of research. Model-based planning picks the best interventions via interactive learning of a user memory model's parameters. The approach is novel in its use of a cognitive model that can account for several key individual- and material-specific characteristics related to recall/forgetting, along with a planning technique that considers users' practice schedules. Taking a rule-based approach as a baseline, the authors evaluated the method's benefits in a controlled study of artificial teaching in second-language vocabulary learning (N=53).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our upcoming <a href="https://twitter.com/hashtag/iui2021?src=hash&amp;ref_src=twsrc%5Etfw">#iui2021</a> paper is out: &quot;Improving Artificial Teachers by Considering how People Learn and Forget&quot; <a href="https://t.co/HS67TlxSRf">https://t.co/HS67TlxSRf</a> Work at <a href="https://twitter.com/FCAI_fi?ref_src=twsrc%5Etfw">@FCAI_fi</a>  and led by Aurelien Nioche <a href="https://t.co/1dkvGm4Lgb">pic.twitter.com/1dkvGm4Lgb</a></p>&mdash; Antti Oulasvirta (@oulasvirta) <a href="https://twitter.com/oulasvirta/status/1359042327055134720?ref_src=twsrc%5Etfw">February 9, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



