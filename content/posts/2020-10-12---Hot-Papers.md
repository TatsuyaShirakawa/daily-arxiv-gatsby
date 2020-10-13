---
title: Hot Papers 2020-10-12
date: 2020-10-13T10:03:15.Z
template: "post"
draft: false
slug: "hot-papers-2020-10-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-10-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. No MCMC for me: Amortized sampling for fast and stable training of  energy-based models

Will Grathwohl, Jacob Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud

- retweets: 2000, favorites: 241 (10/13/2020 10:03:15)

- links: [abs](https://arxiv.org/abs/2010.04230) | [pdf](https://arxiv.org/pdf/2010.04230)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Energy-Based Models (EBMs) present a flexible and appealing way to representuncertainty. Despite recent advances, training EBMs on high-dimensional dataremains a challenging problem as the state-of-the-art approaches are costly, unstable, and require considerable tuning and domain expertise to apply successfully. In this work we present a simple method for training EBMs at scale which uses an entropy-regularized generator to amortize the MCMC sampling typically usedin EBM training. We improve upon prior MCMC-based entropy regularization methods with a fast variational approximation. We demonstrate the effectiveness of our approach by using it to train tractable likelihood models. Next, we apply our estimator to the recently proposed Joint Energy Model (JEM), where we matchthe original performance with faster and stable training. This allows us to extend JEM models to semi-supervised classification on tabular data from a variety of continuous domains.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pleased to share my latest work! <a href="https://t.co/9Qe7afh2HM">https://t.co/9Qe7afh2HM</a> <br><br>We present VERA, a new method for EBM training. Our approach uses a generator to amortize away the MCMC sampling typically used to train EBMs. This training procedure solves many known issues with MCMC-based EBM training. <a href="https://t.co/o9JtBa8TDv">pic.twitter.com/o9JtBa8TDv</a></p>&mdash; will grathwohl (@wgrathwohl) <a href="https://twitter.com/wgrathwohl/status/1315481445507502080?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. GRF: Learning a General Radiance Field for 3D Scene Representation and  Rendering

Alex Trevithick, Bo Yang

- retweets: 1406, favorites: 143 (10/13/2020 10:03:15)

- links: [abs](https://arxiv.org/abs/2010.04595) | [pdf](https://arxiv.org/pdf/2010.04595)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present a simple yet powerful implicit neural function that can represent and render arbitrarily complex 3D scenes in a single network only from 2D observations. The function models 3D scenes as a general radiance field, which takes a set of posed 2D images as input, constructs an internal representation for each 3D point of the scene, and renders the corresponding appearance and geometry of any 3D point viewing from an arbitrary angle. The key to our approach is to explicitly integrate the principle of multi-view geometry to obtain the internal representations from observed 2D views, guaranteeing the learned implicit representations meaningful and multi-view consistent. In addition, we introduce an effective neural module to learn general features for each pixel in 2D images, allowing the constructed internal 3D representations to be remarkably general as well. Extensive experiments demonstrate the superiority of our approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering<br>pdf: <a href="https://t.co/9U8Phmdk4U">https://t.co/9U8Phmdk4U</a><br>abs: <a href="https://t.co/Vq4gsTUt5R">https://t.co/Vq4gsTUt5R</a> <a href="https://t.co/FL8aTkdB9x">pic.twitter.com/FL8aTkdB9x</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1315458557593124869?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. LSTMs Compose (and Learn) Bottom-Up

Naomi Saphra, Adam Lopez

- retweets: 1225, favorites: 225 (10/13/2020 10:03:15)

- links: [abs](https://arxiv.org/abs/2010.04650) | [pdf](https://arxiv.org/pdf/2010.04650)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent work in NLP shows that LSTM language models capture hierarchical structure in language data. In contrast to existing work, we consider the \textit{learning} process that leads to their compositional behavior. For a closer look at how an LSTM's sequential representations are composed hierarchically, we present a related measure of Decompositional Interdependence (DI) between word meanings in an LSTM, based on their gate interactions. We connect this measure to syntax with experiments on English language data, where DI is higher on pairs of words with lower syntactic distance. To explore the inductive biases that cause these compositional representations to arise during training, we conduct simple experiments on synthetic data. These synthetic experiments support a specific hypothesis about how hierarchical structures are discovered over the course of training: that LSTM constituent representations are learned bottom-up, relying on effective representations of their shorter children, rather than learning the longer-range relations independently from children.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My weird obsession with catching an LSTM “in the act” of building syntax—explaining how the training process has an inductive bias towards hierarchy—has culminated in an <a href="https://twitter.com/hashtag/emnlp2020?src=hash&amp;ref_src=twsrc%5Etfw">#emnlp2020</a> Findings paper! <a href="https://t.co/ldNR8wt6m8">https://t.co/ldNR8wt6m8</a> <a href="https://t.co/3GmMMTlvQ2">pic.twitter.com/3GmMMTlvQ2</a></p>&mdash; Naomi&#39;sAFRAID (@nsaphra) <a href="https://twitter.com/nsaphra/status/1315624647841452032?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and  Transfer Learning

Ossama Ahmed, Frederik Träuble, Anirudh Goyal, Alexander Neitz, Manuel Wüthrich, Yoshua Bengio, Bernhard Schölkopf, Stefan Bauer

- retweets: 240, favorites: 67 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04296) | [pdf](https://arxiv.org/pdf/2010.04296)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Despite recent successes of reinforcement learning (RL), it remains a challenge for agents to transfer learned skills to related environments. To facilitate research addressing this problem, we propose CausalWorld, a benchmark for causal structure and transfer learning in a robotic manipulation environment. The environment is a simulation of an open-source robotic platform, hence offering the possibility of sim-to-real transfer. Tasks consist of constructing 3D shapes from a given set of blocks - inspired by how children learn to build complex structures. The key strength of CausalWorld is that it provides a combinatorial family of such tasks with common causal structure and underlying factors (including, e.g., robot and object masses, colors, sizes). The user (or the agent) may intervene on all causal variables, which allows for fine-grained control over how similar different tasks (or task distributions) are. One can thus easily define training and evaluation distributions of a desired difficulty level, targeting a specific form of generalization (e.g., only changes in appearance or object mass). Further, this common parametrization facilitates defining curricula by interpolating between an initial and a target task. While users may define their own task distributions, we present eight meaningful distributions as concrete benchmarks, ranging from simple to very challenging, all of which require long-horizon planning as well as precise low-level motor control. Finally, we provide baseline results for a subset of these tasks on distinct training curricula and corresponding evaluation protocols, verifying the feasibility of the tasks in this benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Causalworld:A Robotic  Manipulation Benchmark For Causal Structure And Transfer Learning<a href="https://t.co/3NbXFuXjY6">https://t.co/3NbXFuXjY6</a><a href="https://t.co/kCS41Pqtm3">https://t.co/kCS41Pqtm3</a> <a href="https://t.co/MUdWQlA1V5">pic.twitter.com/MUdWQlA1V5</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1315503542300360705?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. A Survey of Non-Volatile Main Memory Technologies: State-of-the-Arts,  Practices, and Future Directions

Haikun Liu, Di Chen, Hai Jin, Xiaofei Liao, Bingsheng He, Kan Hu, Yu Zhang

- retweets: 212, favorites: 55 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04406) | [pdf](https://arxiv.org/pdf/2010.04406)
- [cs.DC](https://arxiv.org/list/cs.DC/recent)

Non-Volatile Main Memories (NVMMs) have recently emerged as promising technologies for future memory systems. Generally, NVMMs have many desirable properties such as high density, byte-addressability, non-volatility, low cost, and energy efficiency, at the expense of high write latency, high write power consumption and limited write endurance. NVMMs have become a competitive alternative of Dynamic Random Access Memory (DRAM), and will fundamentally change the landscape of memory systems. They bring many research opportunities as well as challenges on system architectural designs, memory management in operating systems (OSes), and programming models for hybrid memory systems. In this article, we first revisit the landscape of emerging NVMM technologies, and then survey the state-of-the-art studies of NVMM technologies. We classify those studies with a taxonomy according to different dimensions such as memory architectures, data persistence, performance improvement, energy saving, and wear leveling. Second, to demonstrate the best practices in building NVMM systems, we introduce our recent work of hybrid memory system designs from the dimensions of architectures, systems, and applications. At last, we present our vision of future research directions of NVMMs and shed some light on design challenges and opportunities.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this paper is presented a great survey on Non-Volatile Main Memory technologies, showing a taxonomy of the different memory architectures, design challenges and future research directions.<a href="https://t.co/28xE8sUyND">https://t.co/28xE8sUyND</a> <a href="https://t.co/Gsp5ZJ4Mt1">pic.twitter.com/Gsp5ZJ4Mt1</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1315577589109587968?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Fast Fourier Transformation for Optimizing Convolutional Neural Networks  in Object Recognition

Varsha Nair, Moitrayee Chatterjee, Neda Tavakoli, Akbar Siami Namin, Craig Snoeyink

- retweets: 72, favorites: 35 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04257) | [pdf](https://arxiv.org/pdf/2010.04257)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

This paper proposes to use Fast Fourier Transformation-based U-Net (a refined fully convolutional networks) and perform image convolution in neural networks. Leveraging the Fast Fourier Transformation, it reduces the image convolution costs involved in the Convolutional Neural Networks (CNNs) and thus reduces the overall computational costs. The proposed model identifies the object information from the images. We apply the Fast Fourier transform algorithm on an image data set to obtain more accessible information about the image data, before segmenting them through the U-Net architecture. More specifically, we implement the FFT-based convolutional neural network to improve the training time of the network. The proposed approach was applied to publicly available Broad Bioimage Benchmark Collection (BBBC) dataset. Our model demonstrated improvement in training time during convolution from $600-700$ ms/step to $400-500$ ms/step. We evaluated the accuracy of our model using Intersection over Union (IoU) metric showing significant improvements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fast Fourier Transformation for Optimizing Convolutional Neural Networks in Object Recogn... <a href="https://t.co/jcpPlPpH45">https://t.co/jcpPlPpH45</a> <a href="https://t.co/yXColahAUd">pic.twitter.com/yXColahAUd</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1315675633821704192?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis  Including Unsupervised Duration Modeling

Jonathan Shen, Ye Jia, Mike Chrzanowski, Yu Zhang, Isaac Elias, Heiga Zen, Yonghui Wu

- retweets: 49, favorites: 42 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04301) | [pdf](https://arxiv.org/pdf/2010.04301)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

This paper presents Non-Attentive Tacotron based on the Tacotron 2 text-to-speech model, replacing the attention mechanism with an explicit duration predictor. This improves robustness significantly as measured by unaligned duration ratio and word deletion rate, two metrics introduced in this paper for large-scale robustness evaluation using a pre-trained speech recognition model. With the use of Gaussian upsampling, Non-Attentive Tacotron achieves a 5-scale mean opinion score for naturalness of 4.41, slightly outperforming Tacotron 2. The duration predictor enables both utterance-wide and per-phoneme control of duration at inference time. When accurate target durations are scarce or unavailable in the training data, we propose a method using a fine-grained variational auto-encoder to train the duration predictor in a semi-supervised or unsupervised manner, with results almost as good as supervised training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling<br>pdf: <a href="https://t.co/UNlkaBdWIa">https://t.co/UNlkaBdWIa</a><br>abs: <a href="https://t.co/9P2hJ2NINZ">https://t.co/9P2hJ2NINZ</a><br>samples: <a href="https://t.co/LAAYQCwBOH">https://t.co/LAAYQCwBOH</a> <a href="https://t.co/9pcFKXw0Nc">pic.twitter.com/9pcFKXw0Nc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1315463038062002176?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Physical invariance in neural networks for subgrid-scale scalar flux  modeling

Hugo Frezat, Guillaume Balarac, Julien Le Sommer, Ronan Fablet, Redouane Lguensat

- retweets: 56, favorites: 27 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04663) | [pdf](https://arxiv.org/pdf/2010.04663)
- [physics.flu-dyn](https://arxiv.org/list/physics.flu-dyn/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper we present a new strategy to model the subgrid-scale scalar flux in a three-dimensional turbulent incompressible flow using physics-informed neural networks (NNs). When trained from direct numerical simulation (DNS) data, state-of-the-art neural networks, such as convolutional neural networks, may not preserve well known physical priors, which may in turn question their application to real case-studies. To address this issue, we investigate hard and soft constraints into the model based on classical invariances and symmetries derived from physical laws. From simulation-based experiments, we show that the proposed physically-invariant NN model outperforms both purely data-driven ones as well as parametric state-of-the-art subgrid-scale model. The considered invariances are regarded as regularizers on physical metrics during the a priori evaluation and constrain the distribution tails of the predicted subgrid-scale term to be closer to the DNS. They also increase the stability and performance of the model when used as a surrogate during a large-eddy simulation. Moreover, the physically-invariant NN is shown to generalize to configurations that have not been seen during the training phase.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Good news for physicists : physical constrains improve performance and generalisation of NN-based subgrid closures. &quot;Physical invariance in neural networks for subgrid-scale scalar flux modeling&quot; by Frezat et al. paper : <a href="https://t.co/CKLbObRbb3">https://t.co/CKLbObRbb3</a> ; codes : <a href="https://t.co/fhoGLkSkuD">https://t.co/fhoGLkSkuD</a></p>&mdash; Julien Le Sommer (@jlesommer) <a href="https://twitter.com/jlesommer/status/1315667323454455808?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Recurrent babbling: evaluating the acquisition of grammar from limited  input data

Ludovica Pannitto, Aurélie Herbelot

- retweets: 32, favorites: 25 (10/13/2020 10:03:16)

- links: [abs](https://arxiv.org/abs/2010.04637) | [pdf](https://arxiv.org/pdf/2010.04637)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Recurrent Neural Networks (RNNs) have been shown to capture various aspects of syntax from raw linguistic input. In most previous experiments, however, learning happens over unrealistic corpora, which do not reflect the type and amount of data a child would be exposed to. This paper remedies this state of affairs by training a Long Short-Term Memory network (LSTM) over a realistically sized subset of child-directed input. The behaviour of the network is analysed over time using a novel methodology which consists in quantifying the level of grammatical abstraction in the model's generated output (its "babbling"), compared to the language it has been exposed to. We show that the LSTM indeed abstracts new structuresas learning proceeds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/conll2020?src=hash&amp;ref_src=twsrc%5Etfw">#conll2020</a> paper (&quot;Recurrent babbling: evaluating the acquisition of grammar from limited input data&quot;) is now online!<a href="https://t.co/zfXwzOVSQN">https://t.co/zfXwzOVSQN</a><a href="https://twitter.com/ah__cl?ref_src=twsrc%5Etfw">@ah__cl</a> <a href="https://t.co/p71Fvtp4ZZ">https://t.co/p71Fvtp4ZZ</a></p>&mdash; Ludovica Pannitto (@ellepannitto) <a href="https://twitter.com/ellepannitto/status/1315561877011869701?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. comp-syn: Perceptually Grounded Word Embeddings with Color

Bhargav Srinivasa Desikan, Tasker Hull, Ethan O. Nadler, Douglas Guilbeault, Aabir Abubaker Kar, Mark Chu, Donald Ruggiero Lo Sardo

- retweets: 36, favorites: 19 (10/13/2020 10:03:17)

- links: [abs](https://arxiv.org/abs/2010.04292) | [pdf](https://arxiv.org/pdf/2010.04292)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Popular approaches to natural language processing create word embeddings based on textual co-occurrence patterns, but often ignore embodied, sensory aspects of language. Here, we introduce the Python package comp-syn, which provides grounded word embeddings based on the perceptually uniform color distributions of Google Image search results. We demonstrate that comp-syn significantly enriches models of distributional semantics. In particular, we show that (1) comp-syn predicts human judgments of word concreteness with greater accuracy and in a more interpretable fashion than word2vec using low-dimensional word-color embeddings, and (2) comp-syn performs comparably to word2vec on a metaphorical vs. literal word-pair classification task. comp-syn is open-source on PyPi and is compatible with mainstream machine-learning Python packages. Our package release includes word-color embeddings for over 40,000 English words, each associated with crowd-sourced word concreteness judgments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨1/ Our newest compsyn pub. w. Int’l Conference on Computational Linguistics <a href="https://twitter.com/hashtag/coling2020?src=hash&amp;ref_src=twsrc%5Etfw">#coling2020</a>: “Perceptually Grounded Word Embeddings with Color” <a href="https://t.co/cLgOCPgQMl">https://t.co/cLgOCPgQMl</a> Incl. color vectors for 40k most popular words in English, each w. crowdsourced ratings of concept concreteness</p>&mdash; Douglas Guilbeault (@DzGuilbeault) <a href="https://twitter.com/DzGuilbeault/status/1315478368045858818?ref_src=twsrc%5Etfw">October 12, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Deep Learning for Procedural Content Generation

Jialin Liu, Sam Snodgrass, Ahmed Khalifa, Sebastian Risi, Georgios N. Yannakakis, Julian Togelius

- retweets: 24, favorites: 26 (10/13/2020 10:03:17)

- links: [abs](https://arxiv.org/abs/2010.04548) | [pdf](https://arxiv.org/pdf/2010.04548)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Procedural content generation in video games has a long history. Existing procedural content generation methods, such as search-based, solver-based, rule-based and grammar-based methods have been applied to various content types such as levels, maps, character models, and textures. A research field centered on content generation in games has existed for more than a decade. More recently, deep learning has powered a remarkable range of inventions in content production, which are applicable to games. While some cutting-edge deep learning methods are applied on their own, others are applied in combination with more traditional methods, or in an interactive setting. This article surveys the various deep learning methods that have been applied to generate game content directly or indirectly, discusses deep learning methods that could be used for content generation purposes but are rarely used today, and envisages some limitations and potential future directions of deep learning for procedural content generation.



