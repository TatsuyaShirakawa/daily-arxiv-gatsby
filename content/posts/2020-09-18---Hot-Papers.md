---
title: Hot Papers 2020-09-18
date: 2020-09-20T13:13:43.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Decoupling Representation Learning from Reinforcement Learning

Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin

- retweets: 8160, favorites: 0 (09/20/2020 13:13:43)

- links: [abs](https://arxiv.org/abs/2009.08319) | [pdf](https://arxiv.org/pdf/2009.08319)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In an effort to overcome limitations of reward-driven feature learning in deep reinforcement learning (RL) from images, we propose decoupling representation learning from policy learning. To this end, we introduce a new unsupervised learning (UL) task, called Augmented Temporal Contrast (ATC), which trains a convolutional encoder to associate pairs of observations separated by a short time difference, under image augmentations and using a contrastive loss. In online RL experiments, we show that training the encoder exclusively using ATC matches or outperforms end-to-end RL in most environments. Additionally, we benchmark several leading UL algorithms by pre-training encoders on expert demonstrations and using them, with weights frozen, in RL agents; we find that agents using ATC-trained encoders outperform all others. We also train multi-task encoders on data from multiple environments and show generalization to different downstream RL tasks. Finally, we ablate components of ATC, and introduce a new data augmentation to enable replay of (compressed) latent images from pre-trained encoders when RL requires augmentation. Our experiments span visually diverse RL benchmarks in DeepMind Control, DeepMind Lab, and Atari, and our complete code is available at https://github.com/astooke/rlpyt/rlpyt/ul.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper led by <a href="https://twitter.com/astooke?ref_src=twsrc%5Etfw">@astooke</a> w/ <a href="https://twitter.com/kimin_le2?ref_src=twsrc%5Etfw">@kimin_le2</a>  &amp; <a href="https://twitter.com/pabbeel?ref_src=twsrc%5Etfw">@pabbeel</a> - Decoupling Representation Learning from RL. First time RL trained on unsupervised features matches (or beats) end-to-end RL!<br>Paper: <a href="https://t.co/OKxnb2Bt5L">https://t.co/OKxnb2Bt5L</a><br>Code: <a href="https://t.co/XAjhQRYC7Y">https://t.co/XAjhQRYC7Y</a><br>Site: <a href="https://t.co/L7anQtQnh3">https://t.co/L7anQtQnh3</a><br>[1/N] <a href="https://t.co/jmE7QJdpWS">pic.twitter.com/jmE7QJdpWS</a></p>&mdash; Michael (Misha) Laskin (@MishaLaskin) <a href="https://twitter.com/MishaLaskin/status/1306779645287763970?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Neural CDEs for Long Time Series via the Log-ODE Method

James Morrill, Patrick Kidger, Cristopher Salvi, James Foster, Terry Lyons

- retweets: 1448, favorites: 195 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08295) | [pdf](https://arxiv.org/pdf/2009.08295)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [math.DS](https://arxiv.org/list/math.DS/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Neural Controlled Differential Equations (Neural CDEs) are the continuous-time analogue of an RNN, just as Neural ODEs are analogous to ResNets. However just like RNNs, training Neural CDEs can be difficult for long time series. Here, we propose to apply a technique drawn from stochastic analysis, namely the log-ODE method. Instead of using the original input sequence, our procedure summarises the information over local time intervals via the log-signature map, and uses the resulting shorter stream of log-signatures as the new input. This represents a length/channel trade-off. In doing so we demonstrate efficacy on problems of length up to 17k observations and observe significant training speed-ups, improvements in model performance, and reduced memory requirements compared to the existing algorithm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper:<br>&quot;Neural CDEs for Long Time-Series via the Log-ODE Method&quot;<br>GitHub: <a href="https://t.co/MmaYYIKgfk">https://t.co/MmaYYIKgfk</a><br>arXiv: <a href="https://t.co/S2L6YPzL3T">https://t.co/S2L6YPzL3T</a><br>Reddit: <a href="https://t.co/yRVw29LuCt">https://t.co/yRVw29LuCt</a><br><br>We process very long time series of length up to 17k!<br><br>1/ <a href="https://t.co/n08NGsHdmC">pic.twitter.com/n08NGsHdmC</a></p>&mdash; Patrick Kidger (@PatrickKidger) <a href="https://twitter.com/PatrickKidger/status/1306880525026635781?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper:<br>&quot;Neural CDEs for Long Time-Series via the Log-ODE Method&quot;<br>GitHub: <a href="https://t.co/MmaYYIKgfk">https://t.co/MmaYYIKgfk</a><br>arXiv: <a href="https://t.co/S2L6YPzL3T">https://t.co/S2L6YPzL3T</a><br>Reddit: <a href="https://t.co/yRVw29LuCt">https://t.co/yRVw29LuCt</a><br><br>We process very long time series of length up to 17k!<br><br>1/ <a href="https://t.co/n08NGsHdmC">pic.twitter.com/n08NGsHdmC</a></p>&mdash; Patrick Kidger (@PatrickKidger) <a href="https://twitter.com/PatrickKidger/status/1306880525026635781?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. ShapeAssembly: Learning to Generate Programs for 3D Shape Structure  Synthesis

R. Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J. Mitra, Daniel Ritchie

- retweets: 992, favorites: 128 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08026) | [pdf](https://arxiv.org/pdf/2009.08026)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Manually authoring 3D shapes is difficult and time consuming; generative models of 3D shapes offer compelling alternatives. Procedural representations are one such possibility: they offer high-quality and editable results but are difficult to author and often produce outputs with limited diversity. On the other extreme are deep generative models: given enough data, they can learn to generate any class of shape but their outputs have artifacts and the representation is not editable. In this paper, we take a step towards achieving the best of both worlds for novel 3D shape synthesis. We propose ShapeAssembly, a domain-specific "assembly-language" for 3D shape structures. ShapeAssembly programs construct shapes by declaring cuboid part proxies and attaching them to one another, in a hierarchical and symmetrical fashion. Its functions are parameterized with free variables, so that one program structure is able to capture a family of related shapes. We show how to extract ShapeAssembly programs from existing shape structures in the PartNet dataset. Then we train a deep generative model, a hierarchical sequence VAE, that learns to write novel ShapeAssembly programs. The program captures the subset of variability that is interpretable and editable. The deep model captures correlations across shape collections that are hard to express procedurally. We evaluate our approach by comparing shapes output by our generated programs to those from other recent shape structure synthesis models. We find that our generated shapes are more plausible and physically-valid than those of other methods. Additionally, we assess the latent spaces of these models, and find that ours is better structured and produces smoother interpolations. As an application, we use our generative model and differentiable program interpreter to infer and fit shape programs to unstructured geometry, such as point clouds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis<br>pdf: <a href="https://t.co/TI7d8WoEkA">https://t.co/TI7d8WoEkA</a><br>abs: <a href="https://t.co/NIusuS1ObG">https://t.co/NIusuS1ObG</a><br>project page: <a href="https://t.co/uDisteGMlQ">https://t.co/uDisteGMlQ</a><br>github: <a href="https://t.co/wKMVjdhtG1">https://t.co/wKMVjdhtG1</a> <a href="https://t.co/GyRvK5Ryox">pic.twitter.com/GyRvK5Ryox</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306754547319271424?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis<br>pdf: <a href="https://t.co/TI7d8WoEkA">https://t.co/TI7d8WoEkA</a><br>abs: <a href="https://t.co/NIusuS1ObG">https://t.co/NIusuS1ObG</a><br>project page: <a href="https://t.co/uDisteGMlQ">https://t.co/uDisteGMlQ</a><br>github: <a href="https://t.co/wKMVjdhtG1">https://t.co/wKMVjdhtG1</a> <a href="https://t.co/GyRvK5Ryox">pic.twitter.com/GyRvK5Ryox</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306754547319271424?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Distributional Generalization: A New Kind of Generalization

Preetum Nakkiran, Yamini Bansal

- retweets: 252, favorites: 171 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08092) | [pdf](https://arxiv.org/pdf/2009.08092)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [math.ST](https://arxiv.org/list/math.ST/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We introduce a new notion of generalization-- Distributional Generalization-- which roughly states that outputs of a classifier at train and test time are close *as distributions*, as opposed to close in just their average error. For example, if we mislabel 30% of dogs as cats in the train set of CIFAR-10, then a ResNet trained to interpolation will in fact mislabel roughly 30% of dogs as cats on the *test set* as well, while leaving other classes unaffected. This behavior is not captured by classical generalization, which would only consider the average error and not the distribution of errors over the input domain. This example is a specific instance of our much more general conjectures which apply even on distributions where the Bayes risk is zero. Our conjectures characterize the form of distributional generalization that can be expected, in terms of problem parameters (model architecture, training procedure, number of samples, data distribution). We verify the quantitative predictions of these conjectures across a variety of domains in machine learning, including neural networks, kernel machines, and decision trees. These empirical observations are independently interesting, and form a more fine-grained characterization of interpolating classifiers beyond just their test error.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper with <a href="https://twitter.com/whybansal?ref_src=twsrc%5Etfw">@whybansal</a>:<br>&quot;Distributional Generalization: A New Kind of Generalization&quot;<a href="https://t.co/bjA3cX8Eym">https://t.co/bjA3cX8Eym</a><br><br>Thread 1/n<br><br>Here are some quizzes that motivate our results (vote in thread!)<br>QUIZ 1: <a href="https://t.co/ewlJ0kSrcu">pic.twitter.com/ewlJ0kSrcu</a></p>&mdash; Preetum Nakkiran (@PreetumNakkiran) <a href="https://twitter.com/PreetumNakkiran/status/1306790160152121344?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. S2SD: Simultaneous Similarity-based Self-Distillation for Deep Metric  Learning

Karsten Roth, Timo Milbich, Björn Ommer, Joseph Paul Cohen, Marzyeh Ghassemi

- retweets: 182, favorites: 65 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08348) | [pdf](https://arxiv.org/pdf/2009.08348)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep Metric Learning (DML) provides a crucial tool for visual similarity and zero-shot retrieval applications by learning generalizing embedding spaces, although recent work in DML has shown strong performance saturation across training objectives. However, generalization capacity is known to scale with the embedding space dimensionality. Unfortunately, high dimensional embeddings also create higher retrieval cost for downstream applications. To remedy this, we propose S2SD - Simultaneous Similarity-based Self-distillation. S2SD extends DML with knowledge distillation from auxiliary, high-dimensional embedding and feature spaces to leverage complementary context during training while retaining test-time cost and with negligible changes to the training time. Experiments and ablations across different objectives and standard benchmarks show S2SD offering notable improvements of up to 7% in Recall@1, while also setting a new state-of-the-art. Code available at https://github.com/MLforHealth/S2SD.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work out on massively improving generalization in Deep Metric Learning via<br><br>S2SD: Simultaneous Similarity-based Self-Distillation <br><br>Link|Code: <a href="https://t.co/Y5LpCOn6Ra">https://t.co/Y5LpCOn6Ra</a> | <a href="https://t.co/xjJSE9ZAtt">https://t.co/xjJSE9ZAtt</a><br><br>J/w with <a href="https://twitter.com/timoMil?ref_src=twsrc%5Etfw">@timoMil</a>, Björn Ommer, <a href="https://twitter.com/josephpaulcohen?ref_src=twsrc%5Etfw">@josephpaulcohen</a> &amp; <a href="https://twitter.com/MarzyehGhassemi?ref_src=twsrc%5Etfw">@MarzyehGhassemi</a>! <a href="https://t.co/fC6Gjpco0M">pic.twitter.com/fC6Gjpco0M</a></p>&mdash; Karsten Roth (@confusezius) <a href="https://twitter.com/confusezius/status/1306954950329675778?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. The Limits of Pan Privacy and Shuffle Privacy for Learning and  Estimation

Albert Cheu, Jonathan Ullman

- retweets: 148, favorites: 51 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08000) | [pdf](https://arxiv.org/pdf/2009.08000)
- [cs.DS](https://arxiv.org/list/cs.DS/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

There has been a recent wave of interest in intermediate trust models for differential privacy that eliminate the need for a fully trusted central data collector, but overcome the limitations of local differential privacy. This interest has led to the introduction of the shuffle model (Cheu et al., EUROCRYPT 2019; Erlingsson et al., SODA 2019) and revisiting the pan-private model (Dwork et al., ITCS 2010). The message of this line of work is that, for a variety of low-dimensional problems---such as counts, means, and histograms---these intermediate models offer nearly as much power as central differential privacy. However, there has been considerably less success using these models for high-dimensional learning and estimation problems.   In this work, we show that, for a variety of high-dimensional learning and estimation problems, both the shuffle model and the pan-private model inherently incur an exponential price in sample complexity relative to the central model. For example, we show that, private agnostic learning of parity functions over $d$ bits requires $\Omega(2^{d/2})$ samples in these models, and privately selecting the most common attribute from a set of $d$ choices requires $\Omega(d^{1/2})$ samples, both of which are exponential separations from the central model. Our work gives the first non-trivial lower bounds for these problems for both the pan-private model and the general multi-message shuffle model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Really pleased with this new paper with my PhD student Albert Cheu.  We prove strong lower bounds for two &quot;intermediate models&quot; of differential privacy: the shuffle model and the pan-private model. 1/3<a href="https://t.co/iuievPndzc">https://t.co/iuievPndzc</a></p>&mdash; Jonathan Ullman (@thejonullman) <a href="https://twitter.com/thejonullman/status/1306934928765079554?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Moving with the Times: Investigating the Alt-Right Network Gab with  Temporal Interaction Graphs

Naomi A. Arnold, Benjamin A. Steer, Imane Hafnaoui, Hugo A. Parada G., Raul J. Mondragon, Felix Cuadrado, Richard G. Clegg

- retweets: 125, favorites: 54 (09/20/2020 13:13:44)

- links: [abs](https://arxiv.org/abs/2009.08322) | [pdf](https://arxiv.org/pdf/2009.08322)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

Gab is an online social network often associated with the alt-right political movement and users barred from other networks. It presents an interesting opportunity for research because near-complete data is available from day one of the network's creation. In this paper, we investigate the evolution of the user interaction graph, that is the graph where a link represents a user interacting with another user at a given time. We view this graph both at different times and at different timescales. The latter is achieved by using sliding windows on the graph which gives a novel perspective on social network data. The Gab network is relatively slowly growing over the period of months but subject to large bursts of arrivals over hours and days. We identify plausible events that are of interest to the Gab community associated with the most obvious such bursts. The network is characterised by interactions between `strangers' rather than by reinforcing links between `friends'. Gab usage follows the diurnal cycle of the predominantly US and Europe based users. At off-peak hours the Gab interaction network fragments into sub-networks with absolutely no interaction between them. A small group of users are highly influential across larger timescales, but a substantial number of users gain influence for short periods of time. Temporal analysis at different timescales gives new insights above and beyond what could be found on static graphs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Out now on the ArXiv 🎉 our work using temporal interaction graphs to study the online social network Gab. <a href="https://twitter.com/miratepuffin?ref_src=twsrc%5Etfw">@miratepuffin</a> <a href="https://twitter.com/imanehafnus?ref_src=twsrc%5Etfw">@imanehafnus</a> <a href="https://twitter.com/felixcuadrado?ref_src=twsrc%5Etfw">@felixcuadrado</a> <a href="https://twitter.com/richardclegg?ref_src=twsrc%5Etfw">@richardclegg</a> <a href="https://t.co/HDc7mf2amG">https://t.co/HDc7mf2amG</a> <a href="https://t.co/gbEHO7jWDE">pic.twitter.com/gbEHO7jWDE</a></p>&mdash; Naomi Arnold (@narnolddd) <a href="https://twitter.com/narnolddd/status/1306925997682458624?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Impact and dynamics of hate and counter speech online

Joshua Garland, Keyan Ghazi-Zahedi, Jean-Gabriel Young, Laurent Hébert-Dufresne, Mirta Galesic

- retweets: 138, favorites: 20 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08392) | [pdf](https://arxiv.org/pdf/2009.08392)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Citizen-generated counter speech is a promising way to fight hate speech and promote peaceful, non-polarized discourse. However, there is a lack of large-scale longitudinal studies of its effectiveness for reducing hate speech. We investigate the effectiveness of counter speech using several different macro- and micro-level measures of over 180,000 political conversations that took place on German Twitter over four years. We report on the dynamic interactions of hate and counter speech over time and provide insights into whether, as in `classic' bullying situations, organized efforts are more effective than independent individuals in steering online discourse. Taken together, our results build a multifaceted picture of the dynamics of hate and counter speech online. They suggest that organized hate speech produced changes in the public discourse. Counter speech, especially when organized, could help in curbing hate speech in online discussions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Impact and dynamics of hate and counter speech online”<br><br>New preprint led by <a href="https://twitter.com/JoshuaGarland?ref_src=twsrc%5Etfw">@JoshuaGarland</a> w/faculty members <a href="https://twitter.com/LHDnets?ref_src=twsrc%5Etfw">@LHDnets</a> &amp; <a href="https://twitter.com/_jgyou?ref_src=twsrc%5Etfw">@_jgyou</a> plus Keyan Ghazi-Zahedi &amp; Mirta Galesic<a href="https://t.co/OuumkbbsD6">https://t.co/OuumkbbsD6</a> <a href="https://t.co/0F45EFcMPL">pic.twitter.com/0F45EFcMPL</a></p>&mdash; Vermont Complex Systems Center @ UVM (@uvmcomplexity) <a href="https://twitter.com/uvmcomplexity/status/1306952401748930561?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. EventProp: Backpropagation for Exact Gradients in Spiking Neural  Networks

Timo C. Wunderlich, Christian Pehle

- retweets: 130, favorites: 21 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08378) | [pdf](https://arxiv.org/pdf/2009.08378)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

We derive the backpropagation algorithm for spiking neural networks composed of leaky integrate-and-fire neurons operating in continuous time. This algorithm, EventProp, computes the exact gradient of an arbitrary loss function of spike times and membrane potentials by backpropagating errors in time. For the first time, by leveraging methods from optimal control theory, we are able to backpropagate errors through spike discontinuities and avoid approximations or smoothing operations. EventProp can be applied to spiking networks with arbitrary connectivity, including recurrent, convolutional and deep feed-forward architectures. While we consider the leaky integrate-and-fire neuron model in this work, our methodology to derive the gradient can be applied to other spiking neuron models. As errors are backpropagated in an event-based manner (at spike times), EventProp requires the storage of state variables only at these times, providing favorable memory requirements. We demonstrate learning using gradients computed via EventProp in a deep spiking network using an event-based simulator and a non-linearly separable dataset encoded using spike time latencies. Our work supports the rigorous study of gradient-based methods to train spiking neural networks while providing insights toward the development of learning algorithms in neuromorphic hardware.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Backpropagation for Spiking Neural Networks has been solved! <br>A brilliant friend of mine just published this preprint on using the adjoint method to compute exact gradients for discontinuous spiking dynamics. Pretty cool! <a href="https://twitter.com/hashtag/ml?src=hash&amp;ref_src=twsrc%5Etfw">#ml</a> <a href="https://twitter.com/hashtag/ai?src=hash&amp;ref_src=twsrc%5Etfw">#ai</a> <a href="https://twitter.com/hashtag/snn?src=hash&amp;ref_src=twsrc%5Etfw">#snn</a> <a href="https://twitter.com/hashtag/compneuro?src=hash&amp;ref_src=twsrc%5Etfw">#compneuro</a> <a href="https://twitter.com/hashtag/phdlife?src=hash&amp;ref_src=twsrc%5Etfw">#phdlife</a><a href="https://t.co/cDW7HwJ2Tn">https://t.co/cDW7HwJ2Tn</a></p>&mdash; Jens Egholm (@jensegholm) <a href="https://twitter.com/jensegholm/status/1306894879629467655?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. MoPro: Webly Supervised Learning with Momentum Prototypes

Junnan Li, Caiming Xiong, Steven C.H. Hoi

- retweets: 110, favorites: 40 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.07995) | [pdf](https://arxiv.org/pdf/2009.07995)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a webly-supervised representation learning method that does not suffer from the annotation unscalability of supervised learning, nor the computation unscalability of self-supervised learning. Most existing works on webly-supervised representation learning adopt a vanilla supervised learning method without accounting for the prevalent noise in the training data, whereas most prior methods in learning with label noise are less effective for real-world large-scale noisy data. We propose momentum prototypes (MoPro), a simple contrastive learning method that achieves online label noise correction, out-of-distribution sample removal, and representation learning. MoPro achieves state-of-the-art performance on WebVision, a weakly-labeled noisy dataset. MoPro also shows superior performance when the pretrained model is transferred to down-stream image classification and detection tasks. It outperforms the ImageNet supervised pretrained model by +10.5 on 1-shot classification on VOC, and outperforms the best self-supervised pretrained model by +17.3 when finetuned on 1\% of ImageNet labeled samples. Furthermore, MoPro is more robust to distribution shifts. Code and pretrained models are available at https://github.com/salesforce/MoPro.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to introduce “MoPro”: a Webly Supervised Learning paradigm with Momentum Prototypes for learning representation beyond Self-Supervised Learning. <br><br>Paper: <a href="https://t.co/8wsN09xJZq">https://t.co/8wsN09xJZq</a><br>Blog: <a href="https://t.co/cyMUyHR7X1">https://t.co/cyMUyHR7X1</a><br>Code: <a href="https://t.co/UptcvF8Sxe">https://t.co/UptcvF8Sxe</a><br><br>w/ <a href="https://twitter.com/LiJunnan0409?ref_src=twsrc%5Etfw">@LiJunnan0409</a> &amp; <a href="https://twitter.com/CaimingXiong?ref_src=twsrc%5Etfw">@CaimingXiong</a></p>&mdash; Steven Hoi (@stevenhoi) <a href="https://twitter.com/stevenhoi/status/1306856080304082944?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Computational models in Electroencephalography

Katharina Glomb, Joana Cabral, Anna Cattani, Alberto Mazzoni, Ashish Raj, Benedetta Franceschiello

- retweets: 114, favorites: 30 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08385) | [pdf](https://arxiv.org/pdf/2009.08385)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.CE](https://arxiv.org/list/cs.CE/recent)

Computational models lie at the intersection of basic neuroscience and healthcare applications because they allow researchers to test hypotheses \textit{in silico} and predict the outcome of experiments and interactions that are very hard to test in reality. Yet, what is meant by "computational model" is understood in many different ways by researchers in different fields of neuroscience and psychology, hindering communication and collaboration. In this review, we point out the state of the art of computational modeling in Electroencephalography (EEG) and outline how these models can be used to integrate findings from electrophysiology, network-level models, and behavior. On the one hand, computational models serve to investigate the mechanisms that generate brain activity, for example measured with EEG, such as the transient emergence of oscillations at different frequency bands and/or with different spatial topographies. On the other hand, computational models serve to design experiments and test hypotheses \emph{in silico}. The final purpose of computational models of EEG is to obtain a comprehensive understanding of the mechanisms that underlie the EEG signal. This is crucial for an accurate interpretation of EEG measurements that may ultimately serve in the development of novel clinical applications.




# 12. DLBCL-Morph: Morphological features computed using deep learning for an  annotated digital DLBCL image set

Damir Vrabac, Akshay Smit, Rebecca Rojansky, Yasodha Natkunam, Ranjana H. Advani, Andrew Y. Ng, Sebastian Fernandez-Pol, Pranav Rajpurkar

- retweets: 90, favorites: 50 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08123) | [pdf](https://arxiv.org/pdf/2009.08123)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Diffuse Large B-Cell Lymphoma (DLBCL) is the most common non-Hodgkin lymphoma. Though histologically DLBCL shows varying morphologies, no morphologic features have been consistently demonstrated to correlate with prognosis. We present a morphologic analysis of histology sections from 209 DLBCL cases with associated clinical and cytogenetic data. Duplicate tissue core sections were arranged in tissue microarrays (TMAs), and replicate sections were stained with H&E and immunohistochemical stains for CD10, BCL6, MUM1, BCL2, and MYC. The TMAs are accompanied by pathologist-annotated regions-of-interest (ROIs) that identify areas of tissue representative of DLBCL. We used a deep learning model to segment all tumor nuclei in the ROIs, and computed several geometric features for each segmented nucleus. We fit a Cox proportional hazards model to demonstrate the utility of these geometric features in predicting survival outcome, and found that it achieved a C-index (95% CI) of 0.635 (0.574,0.691). Our finding suggests that geometric features computed from tumor nuclei are of prognostic importance, and should be validated in prospective studies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our new work (paper+dataset) on predicting survival with cancer (DLBCL) using histology sections 🔬💭<br><br>DLBCL-Morph <a href="https://t.co/gBqJLFlwmV">https://t.co/gBqJLFlwmV</a><br><br>With <a href="https://twitter.com/dvrabac?ref_src=twsrc%5Etfw">@dvrabac</a> <a href="https://twitter.com/AkshaySmit?ref_src=twsrc%5Etfw">@AkshaySmit</a>, Sebastian Fernandez-Pol<br><br>Rebecca Rojansky, <a href="https://twitter.com/yaso_natkunam?ref_src=twsrc%5Etfw">@yaso_natkunam</a>, Ranjana Advani, <a href="https://twitter.com/AndrewYNg?ref_src=twsrc%5Etfw">@AndrewYNg</a> <br><br>1/n <a href="https://t.co/b1090DPji0">pic.twitter.com/b1090DPji0</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1306777007527616512?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. DanceIt: Music-inspired Dancing Video Synthesis

Xin Guo, Jia Li, Yifan Zhao

- retweets: 65, favorites: 41 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08027) | [pdf](https://arxiv.org/pdf/2009.08027)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Close your eyes and listen to music, one can easily imagine an actor dancing rhythmically along with the music. These dance movements are usually made up of dance movements you have seen before. In this paper, we propose to reproduce such an inherent capability of the human-being within a computer vision system. The proposed system consists of three modules. To explore the relationship between music and dance movements, we propose a cross-modal alignment module that focuses on dancing video clips, accompanied on pre-designed music, to learn a system that can judge the consistency between the visual features of pose sequences and the acoustic features of music. The learned model is then used in the imagination module to select a pose sequence for the given music. Such pose sequence selected from the music, however, is usually discontinuous. To solve this problem, in the spatial-temporal alignment module we develop a spatial alignment algorithm based on the tendency and periodicity of dance movements to predict dance movements between discontinuous fragments. In addition, the selected pose sequence is often misaligned with the music beat. To solve this problem, we further develop a temporal alignment algorithm to align the rhythm of music and dance. Finally, the processed pose sequence is used to synthesize realistic dancing videos in the imagination module. The generated dancing videos match the content and rhythm of the music. Experimental results and subjective evaluations show that the proposed approach can perform the function of generating promising dancing videos by inputting music.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DanceIt: Music-inspired Dancing Video Synthesis<br>pdf: <a href="https://t.co/ERo9TysMCD">https://t.co/ERo9TysMCD</a><br>abs: <a href="https://t.co/V0NNje3v57">https://t.co/V0NNje3v57</a> <a href="https://t.co/wV3RnHcRT6">pic.twitter.com/wV3RnHcRT6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306768840634839043?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Temporally Guided Music-to-Body-Movement Generation

Hsuan-Kai Kao, Li Su

- retweets: 62, favorites: 41 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08015) | [pdf](https://arxiv.org/pdf/2009.08015)
- [cs.MM](https://arxiv.org/list/cs.MM/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

This paper presents a neural network model to generate virtual violinist's 3-D skeleton movements from music audio. Improved from the conventional recurrent neural network models for generating 2-D skeleton data in previous works, the proposed model incorporates an encoder-decoder architecture, as well as the self-attention mechanism to model the complicated dynamics in body movement sequences. To facilitate the optimization of self-attention model, beat tracking is applied to determine effective sizes and boundaries of the training examples. The decoder is accompanied with a refining network and a bowing attack inference mechanism to emphasize the right-hand behavior and bowing attack timing. Both objective and subjective evaluations reveal that the proposed model outperforms the state-of-the-art methods. To the best of our knowledge, this work represents the first attempt to generate 3-D violinists' body movements considering key features in musical body movement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Temporally Guided Music-to-Body-Movement Generation<br>pdf: <a href="https://t.co/KquHBWs10Q">https://t.co/KquHBWs10Q</a><br>abs: <a href="https://t.co/H9tbn1kATj">https://t.co/H9tbn1kATj</a><br>github: <a href="https://t.co/nKrSwGDUIf">https://t.co/nKrSwGDUIf</a> <a href="https://t.co/ep5xLkTLUh">pic.twitter.com/ep5xLkTLUh</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306758436340760576?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Fast and robust quantum state tomography from few basis measurements

Fernando G.S.L. Brandão, Richard Kueng, Daniel Stilck França

- retweets: 35, favorites: 52 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08216) | [pdf](https://arxiv.org/pdf/2009.08216)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

Quantum state tomography is a powerful, but resource-intensive, general solution for numerous quantum information processing tasks. This motivates the design of robust tomography procedures that use relevant resources as sparingly as possible. Important cost factors include the number of state copies and measurement settings, as well as classical postprocessing time and memory. In this work, we present and analyze an online tomography algorithm that is designed to optimize all the aforementioned resources at the cost of a worse dependence on accuracy. The protocol is the first to give optimal performance in terms of rank and dimension for state copies, measurement settings and memory. Classical runtime is also reduced substantially. Further improvements are possible by executing the algorithm on a quantum computer, giving a quantum speedup for quantum state tomography.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper out with Richard Küng and Fernando Brandão from <a href="https://twitter.com/IQIM_Caltech?ref_src=twsrc%5Etfw">@IQIM_Caltech</a>:<br>Fast and robust quantum state tomography from few basis measurements (<a href="https://t.co/vnlZ1UGnhP">https://t.co/vnlZ1UGnhP</a>). In this paper we analyze an algorithm for low-rank quantum tomography with several nice features:</p>&mdash; Daniel Stilck França (@dsfranca) <a href="https://twitter.com/dsfranca/status/1306831732814098432?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Analysis of Generalizability of Deep Neural Networks Based on the  Complexity of Decision Boundary

Shuyue Guan, Murray Loew

- retweets: 64, favorites: 16 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.07974) | [pdf](https://arxiv.org/pdf/2009.07974)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

For supervised learning models, the analysis of generalization ability (generalizability) is vital because the generalizability expresses how well a model will perform on unseen data. Traditional generalization methods, such as the VC dimension, do not apply to deep neural network (DNN) models. Thus, new theories to explain the generalizability of DNNs are required. In this study, we hypothesize that the DNN with a simpler decision boundary has better generalizability by the law of parsimony (Occam's Razor). We create the decision boundary complexity (DBC) score to define and measure the complexity of decision boundary of DNNs. The idea of the DBC score is to generate data points (called adversarial examples) on or near the decision boundary. Our new approach then measures the complexity of the boundary using the entropy of eigenvalues of these data. The method works equally well for high-dimensional data. We use training data and the trained model to compute the DBC score. And, the ground truth for model's generalizability is its test accuracy. Experiments based on the DBC score have verified our hypothesis. The DBC is shown to provide an effective method to measure the complexity of a decision boundary and gives a quantitative measure of the generalizability of DNNs.




# 17. Captum: A unified and generic model interpretability library for PyTorch

Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, Orion Reblitz-Richardson

- retweets: 58, favorites: 21 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.07896) | [pdf](https://arxiv.org/pdf/2009.07896)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In this paper we introduce a novel, unified, open-source model interpretability library for PyTorch [12]. The library contains generic implementations of a number of gradient and perturbation-based attribution algorithms, also known as feature, neuron and layer importance algorithms, as well as a set of evaluation metrics for these algorithms. It can be used for both classification and non-classification models including graph-structured models built on Neural Networks (NN). In this paper we give a high-level overview of supported attribution algorithms and show how to perform memory-efficient and scalable computations. We emphasize that the three main characteristics of the library are multimodality, extensibility and ease of use. Multimodality supports different modality of inputs such as image, text, audio or video. Extensibility allows adding new algorithms and features. The library is also designed for easy understanding and use. Besides, we also introduce an interactive visualization tool called Captum Insights that is built on top of Captum library and allows sample-based model debugging and visualization using feature importance metrics.




# 18. Crossing You in Style: Cross-modal Style Transfer from Music to Visual  Arts

Cheng-Che Lee, Wan-Yi Lin, Yen-Ting Shih, Pei-Yi Patricia Kuo, Li Su

- retweets: 36, favorites: 40 (09/20/2020 13:13:45)

- links: [abs](https://arxiv.org/abs/2009.08083) | [pdf](https://arxiv.org/pdf/2009.08083)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Music-to-visual style transfer is a challenging yet important cross-modal learning problem in the practice of creativity. Its major difference from the traditional image style transfer problem is that the style information is provided by music rather than images. Assuming that musical features can be properly mapped to visual contents through semantic links between the two domains, we solve the music-to-visual style transfer problem in two steps: music visualization and style transfer. The music visualization network utilizes an encoder-generator architecture with a conditional generative adversarial network to generate image-based music representations from music data. This network is integrated with an image style transfer method to accomplish the style transfer process. Experiments are conducted on WikiArt-IMSLP, a newly compiled dataset including Western music recordings and paintings listed by decades. By utilizing such a label to learn the semantic connection between paintings and music, we demonstrate that the proposed framework can generate diverse image style representations from a music piece, and these representations can unveil certain art forms of the same era. Subjective testing results also emphasize the role of the era label in improving the perceptual quality on the compatibility between music and visual content.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Crossing You in Style: Cross-modal Style Transfer from Music to Visual Arts<br>pdf: <a href="https://t.co/my9Pp08zD5">https://t.co/my9Pp08zD5</a><br>abs: <a href="https://t.co/cg6iLMoxnt">https://t.co/cg6iLMoxnt</a><br>project page: <a href="https://t.co/5J82dJ47Lq">https://t.co/5J82dJ47Lq</a> <a href="https://t.co/9v1FNYIycY">pic.twitter.com/9v1FNYIycY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1306765462735409152?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Radar-Camera Sensor Fusion for Joint Object Detection and Distance  Estimation in Autonomous Vehicles

Ramin Nabati, Hairong Qi

- retweets: 58, favorites: 18 (09/20/2020 13:13:46)

- links: [abs](https://arxiv.org/abs/2009.08428) | [pdf](https://arxiv.org/pdf/2009.08428)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper we present a novel radar-camera sensor fusion framework for accurate object detection and distance estimation in autonomous driving scenarios. The proposed architecture uses a middle-fusion approach to fuse the radar point clouds and RGB images. Our radar object proposal network uses radar point clouds to generate 3D proposals from a set of 3D prior boxes. These proposals are mapped to the image and fed into a Radar Proposal Refinement (RPR) network for objectness score prediction and box refinement. The RPR network utilizes both radar information and image feature maps to generate accurate object proposals and distance estimations. The radar-based proposals are combined with image-based proposals generated by a modified Region Proposal Network (RPN). The RPN has a distance regression layer for estimating distance for every generated proposal. The radar-based and image-based proposals are merged and used in the next stage for object classification. Experiments on the challenging nuScenes dataset show our method outperforms other existing radar-camera fusion methods in the 2D object detection task while at the same time accurately estimates objects' distances.




# 20. A Glimpse of the First Eight Months of the COVID-19 Literature on  Microsoft Academic Graph: Themes, Citation Contexts, and Uncertainties

Chaomei Chen

- retweets: 56, favorites: 15 (09/20/2020 13:13:46)

- links: [abs](https://arxiv.org/abs/2009.08374) | [pdf](https://arxiv.org/pdf/2009.08374)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

As scientists worldwide search for answers to the overwhelmingly unknown behind the deadly pandemic, the literature concerning COVID-19 has been growing exponentially. Keeping abreast of the body of literature at such a rapidly advancing pace poses significant challenges not only to active researchers but also to the society as a whole. Although numerous data resources have been made openly available, the analytic and synthetic process that is essential in effectively navigating through the vast amount of information with heightened levels of uncertainty remains a significant bottleneck. We introduce a generic method that facilitates the data collection and sense-making process when dealing with a rapidly growing landscape of a research domain such as COVID-19 at multiple levels of granularity. The method integrates the analysis of structural and temporal patterns in scholarly publications with the delineation of thematic concentrations and the types of uncertainties that may offer additional insights into the complexity of the unknown. We demonstrate the application of the method in a study of the COVID-19 literature.




# 21. What if we had no Wikipedia? Domain-independent Term Extraction from a  Large News Corpus

Yonatan Bilu, Shai Gretz, Edo Cohen, Noam Slonim

- retweets: 26, favorites: 37 (09/20/2020 13:13:46)

- links: [abs](https://arxiv.org/abs/2009.08240) | [pdf](https://arxiv.org/pdf/2009.08240)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

One of the most impressive human endeavors of the past two decades is the collection and categorization of human knowledge in the free and accessible format that is Wikipedia. In this work we ask what makes a term worthy of entering this edifice of knowledge, and having a page of its own in Wikipedia? To what extent is this a natural product of on-going human discourse and discussion rather than an idiosyncratic choice of Wikipedia editors? Specifically, we aim to identify such "wiki-worthy" terms in a massive news corpus, and see if this can be done with no, or minimal, dependency on actual Wikipedia entries. We suggest a five-step pipeline for doing so, providing baseline results for all five, and the relevant datasets for benchmarking them. Our work sheds new light on the domain-specific Automatic Term Extraction problem, with the problem at hand being a domain-independent variant of it.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;What if we had no Wikipedia? Domain-independent Term Extraction from a Large News Corpus&quot;   identifying   “wiki-worthy” terms in a massive news corpus, with minimal dependency on actual Wikipedia entries.<br><br>(Bilu et al, 2020)<a href="https://t.co/wEts0vt9tl">https://t.co/wEts0vt9tl</a> <a href="https://t.co/7Zv3AnPZXt">pic.twitter.com/7Zv3AnPZXt</a></p>&mdash; WikiResearch (@WikiResearch) <a href="https://twitter.com/WikiResearch/status/1306913641153863680?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. GraphCodeBERT: Pre-training Code Representations with Data Flow

Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Jian Yin, Daxin Jiang, Ming Zhou

- retweets: 32, favorites: 28 (09/20/2020 13:13:46)

- links: [abs](https://arxiv.org/abs/2009.08366) | [pdf](https://arxiv.org/pdf/2009.08366)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Pre-trained models for programming language have achieved dramatic empirical improvements on a variety of code-related tasks such as code search, code completion, code summarization, etc. However, existing pre-trained models regard a code snippet as a sequence of tokens, while ignoring the inherent structure of code, which provides crucial code semantics and would enhance the code understanding process. We present GraphCodeBERT, a pre-trained model for programming language that considers the inherent structure of code. Instead of taking syntactic-level structure of code like abstract syntax tree (AST), we use data flow in the pre-training stage, which is a semantic-level structure of code that encodes the relation of "where-the-value-comes-from" between variables. Such a semantic-level structure is neat and does not bring an unnecessarily deep hierarchy of AST, the property of which makes the model more efficient. We develop GraphCodeBERT based on Transformer. In addition to using the task of masked language modeling, we introduce two structure-aware pre-training tasks. One is to predict code structure edges, and the other is to align representations between source code and code structure. We implement the model in an efficient way with a graph-guided masked attention function to incorporate the code structure. We evaluate our model on four tasks, including code search, clone detection, code translation, and code refinement. Results show that code structure and newly introduced pre-training tasks can improve GraphCodeBERT and achieves state-of-the-art performance on the four downstream tasks. We further show that the model prefers structure-level attentions over token-level attentions in the task of code search.




# 23. A Computational Approach to Understanding Empathy Expressed in  Text-Based Mental Health Support

Ashish Sharma, Adam S. Miner, David C. Atkins, Tim Althoff

- retweets: 30, favorites: 26 (09/20/2020 13:13:46)

- links: [abs](https://arxiv.org/abs/2009.08441) | [pdf](https://arxiv.org/pdf/2009.08441)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Empathy is critical to successful mental health support. Empathy measurement has predominantly occurred in synchronous, face-to-face settings, and may not translate to asynchronous, text-based contexts. Because millions of people use text-based platforms for mental health support, understanding empathy in these contexts is crucial. In this work, we present a computational approach to understanding how empathy is expressed in online mental health platforms. We develop a novel unifying theoretically-grounded framework for characterizing the communication of empathy in text-based conversations. We collect and share a corpus of 10k (post, response) pairs annotated using this empathy framework with supporting evidence for annotations (rationales). We develop a multi-task RoBERTa-based bi-encoder model for identifying empathy in conversations and extracting rationales underlying its predictions. Experiments demonstrate that our approach can effectively identify empathic conversations. We further apply this model to analyze 235k mental health interactions and show that users do not self-learn empathy over time, revealing opportunities for empathy training and feedback.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our <a href="https://twitter.com/hashtag/EMNLP2020?src=hash&amp;ref_src=twsrc%5Etfw">#EMNLP2020</a> paper presents a new computational approach to understanding how empathy is expressed in text-based mental health support! <br><br>Joint work with Adam Miner, Dave Atkins, and <a href="https://twitter.com/timalthoff?ref_src=twsrc%5Etfw">@timalthoff</a><br><br>Preprint: <a href="https://t.co/HS7HT9v8IV">https://t.co/HS7HT9v8IV</a><br><br>Summary 👇 1/7</p>&mdash; Ashish Sharma (@sharma_ashish_2) <a href="https://twitter.com/sharma_ashish_2/status/1306956675786571778?ref_src=twsrc%5Etfw">September 18, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



