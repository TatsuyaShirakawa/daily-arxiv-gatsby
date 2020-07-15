---
title: Hot Papers 2020-07-14
date: 2020-07-15T09:58:12.Z
template: "post"
draft: false
slug: "hot-papers-2020-07-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-07-14"
socialImage: "/media/42-line-bible.jpg"

---

# 1. CheXphoto: 10,000+ Smartphone Photos and Synthetic Photographic  Transformations of Chest X-rays for Benchmarking Deep Learning Robustness

Nick A. Phillips, Pranav Rajpurkar, Mark Sabini, Rayan Krishnan, Sharon Zhou, Anuj Pareek, Nguyet Minh Phu, Chris Wang, Andrew Y. Ng, Matthew P. Lungren

- retweets: 61, favorites: 240 (07/15/2020 09:58:12)

- links: [abs](https://arxiv.org/abs/2007.06199) | [pdf](https://arxiv.org/pdf/2007.06199)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Clinical deployment of deep learning algorithms for chest x-ray interpretation requires a solution that can integrate into the vast spectrum of clinical workflows across the world. An appealing solution to scaled deployment is to leverage the existing ubiquity of smartphones: in several parts of the world, clinicians and radiologists capture photos of chest x-rays to share with other experts or clinicians via smartphone using messaging services like WhatsApp. However, the application of chest x-ray algorithms to photos of chest x-rays requires reliable classification in the presence of smartphone photo artifacts such as screen glare and poor viewing angle not typically encountered on digital x-rays used to train machine learning models. We introduce CheXphoto, a dataset of smartphone photos and synthetic photographic transformations of chest x-rays sampled from the CheXpert dataset. To generate CheXphoto we (1) automatically and manually captured photos of digital x-rays under different settings, including various lighting conditions and locations, and, (2) generated synthetic transformations of digital x-rays targeted to make them look like photos of digital x-rays and x-ray films. We release this dataset as a resource for testing and improving the robustness of deep learning algorithms for automated chest x-ray interpretation on smartphone photos of chest x-rays.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our latest research efforts in AI+medicine🌟<br><br>Introducing CheXphoto 📸, a dataset of 10,000+ photos of chest X-rays for benchmarking deep learning robustness<a href="https://t.co/0dkMYMAYYM">https://t.co/0dkMYMAYYM</a><br><br>w/ <a href="https://twitter.com/nphill22?ref_src=twsrc%5Etfw">@nphill22</a>, Mark Sabini, <a href="https://twitter.com/RayanKrishnan?ref_src=twsrc%5Etfw">@RayanKrishnan</a> et al. <a href="https://twitter.com/StanfordAILab?ref_src=twsrc%5Etfw">@StanfordAILab</a><br><br>⬇️ 1/n <a href="https://t.co/RJFfsYJANI">pic.twitter.com/RJFfsYJANI</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1282955899196157953?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Meta-Learning Requires Meta-Augmentation

Janarthanan Rajendran, Alex Irpan, Eric Jang

- retweets: 48, favorites: 231 (07/15/2020 09:58:13)

- links: [abs](https://arxiv.org/abs/2007.05549) | [pdf](https://arxiv.org/pdf/2007.05549)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Meta-learning algorithms aim to learn two components: a model that predicts targets for a task, and a base learner that quickly updates that model when given examples from a new task. This additional level of learning can be powerful, but it also creates another potential source for overfitting, since we can now overfit in either the model or the base learner. We describe both of these forms of metalearning overfitting, and demonstrate that they appear experimentally in common meta-learning benchmarks. We then use an information-theoretic framework to discuss meta-augmentation, a way to add randomness that discourages the base learner and model from learning trivial solutions that do not generalize to new tasks. We demonstrate that meta-augmentation produces large complementary benefits to recently proposed meta-regularization techniques.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Simply augmenting the data often yields bigger perf gains than tweaking the model. <br>We formalize &quot;meta-augmentation&quot; and show that you can apply it to pretty much any meta-learning problem and any meta-learner.<a href="https://t.co/uQLvzlS6tX">https://t.co/uQLvzlS6tX</a> <br><br>with Janarthanan Rajendran, <a href="https://twitter.com/AlexIrpan?ref_src=twsrc%5Etfw">@AlexIrpan</a> <a href="https://t.co/2qIVNlhVAw">pic.twitter.com/2qIVNlhVAw</a></p>&mdash; Eric Jang 🇺🇸🇹🇼 (@ericjang11) <a href="https://twitter.com/ericjang11/status/1282845153007263744?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Graph Structure of Neural Networks

Jiaxuan You, Jure Leskovec, Kaiming He, Saining Xie

- retweets: 42, favorites: 235 (07/15/2020 09:58:13)

- links: [abs](https://arxiv.org/abs/2007.06559) | [pdf](https://arxiv.org/pdf/2007.06559)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Neural networks are often represented as graphs of connections between neurons. However, despite their wide use, there is currently little understanding of the relationship between the graph structure of the neural network and its predictive performance. Here we systematically investigate how does the graph structure of neural networks affect their predictive performance. To this end, we develop a novel graph-based representation of neural networks called relational graph, where layers of neural network computation correspond to rounds of message exchange along the graph structure. Using this representation we show that: (1) a "sweet spot" of relational graphs leads to neural networks with significantly improved predictive performance; (2) neural network's performance is approximately a smooth function of the clustering coefficient and average path length of its relational graph; (3) our findings are consistent across many different tasks and datasets; (4) the sweet spot can be identified efficiently; (5) top-performing neural networks have graph structure surprisingly similar to those of real biological neural networks. Our work opens new directions for the design of neural architectures and the understanding on neural networks in general.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Graph Structure of Neural Networks <a href="https://t.co/r2Xo6wF4uJ">https://t.co/r2Xo6wF4uJ</a> MLPやCNNをrelational graphという形式で表現し，計算量を揃えランダムにグラフを得たところ，高い性能を得るような平均パス長とクラスタ係数の組合せの領域が存在した．この枠組みで良いMLPはマカクの神経網とも類似している！面白い． <a href="https://t.co/sYIyNp7tKH">pic.twitter.com/sYIyNp7tKH</a></p>&mdash; ワクワクさん (@mosko_mule) <a href="https://twitter.com/mosko_mule/status/1282989209347743745?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How does the graph structure of a neural network affect its predictive performance?<br><br>Our ICML 2020 paper &quot;Graph Structure of Neural Networks&quot; <a href="https://t.co/chlJX6Qlr3">https://t.co/chlJX6Qlr3</a> reveals many interesting findings on this topic.<br>with Jure Leskovec, Kaiming He, Saining Xie <a href="https://twitter.com/hashtag/ICML2020?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2020</a></p>&mdash; Jiaxuan You (@youjiaxuan) <a href="https://twitter.com/youjiaxuan/status/1282934819358298112?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Data-Efficient Reinforcement Learning with Momentum Predictive  Representations

Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman

- retweets: 35, favorites: 187 (07/15/2020 09:58:13)

- links: [abs](https://arxiv.org/abs/2007.05929) | [pdf](https://arxiv.org/pdf/2007.05929)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

While deep reinforcement learning excels at solving tasks where large amounts of data can be collected through virtually unlimited interaction with the environment, learning from limited interaction remains a key challenge. We posit that an agent can learn more efficiently if we augment reward maximization with self-supervised objectives based on structure in its visual input and sequential interaction with the environment. Our method, Momentum Predictive Representations (MPR), trains an agent to predict its own latent state representations multiple steps into the future. We compute target representations for future states using an encoder which is an exponential moving average of the agent's parameters, and we make predictions using a learned transition model. On its own, this future prediction objective outperforms prior methods for sample-efficient deep RL from pixels. We further improve performance by adding data augmentation to the future prediction loss, which forces the agent's representations to be consistent across multiple views of an observation. Our full self-supervised objective, which combines future prediction and data augmentation, achieves a median human-normalized score of 0.444 on Atari in a setting limited to 100K steps of environment interaction, which is a 66% relative improvement over the previous state-of-the-art. Moreover, even in this limited data regime, MPR exceeds expert human scores on 6 out of 26 games.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint: Data-Efficient RL with Momentum Predictive Representations(<a href="https://t.co/AN8St4eSpC">https://t.co/AN8St4eSpC</a>)<br><br>In 100K steps(&lt;2hrs) on Atari, using self-predictions via a latent model &amp; data aug, MPR:<br>* improves SOTA human-norm’d score from 26.8% to 44.4%<br>* exceeds human scores on 6/26 games <a href="https://t.co/C0nnoCa665">pic.twitter.com/C0nnoCa665</a></p>&mdash; Ankesh Anand (@ankesh_anand) <a href="https://twitter.com/ankesh_anand/status/1282843869508440067?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Contrastive Training for Improved Out-of-Distribution Detection

Jim Winkens, Rudy Bunel, Abhijit Guha Roy, Robert Stanforth, Vivek Natarajan, Joseph R. Ledsam, Patricia MacWilliams, Pushmeet Kohli, Alan Karthikesalingam, Simon Kohl, Taylan Cemgil, S. M. Ali Eslami, Olaf Ronneberger

- retweets: 36, favorites: 180 (07/15/2020 09:58:13)

- links: [abs](https://arxiv.org/abs/2007.05566) | [pdf](https://arxiv.org/pdf/2007.05566)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Reliable detection of out-of-distribution (OOD) inputs is increasingly understood to be a precondition for deployment of machine learning systems. This paper proposes and investigates the use of contrastive training to boost OOD detection performance. Unlike leading methods for OOD detection, our approach does not require access to examples labeled explicitly as OOD, which can be difficult to collect in practice. We show in extensive experiments that contrastive training significantly helps OOD detection performance on a number of common benchmarks. By introducing and employing the Confusion Log Probability (CLP) score, which quantifies the difficulty of the OOD detection task by capturing the similarity of inlier and outlier datasets, we show that our method especially improves performance in the `near OOD' classes -- a particularly challenging setting for previous methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contrastive Training for Improved Out-of-Distribution Detection<a href="https://t.co/okkuQ0v5nG">https://t.co/okkuQ0v5nG</a><br><br>Joint (cross entropy + SimCLR) training gives your network a feature space that is better for OOD detection than cross entropy training alone. <a href="https://t.co/DgGdAVSXAF">pic.twitter.com/DgGdAVSXAF</a></p>&mdash; Ali Eslami (@arkitus) <a href="https://twitter.com/arkitus/status/1283065428084961281?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/2) Our new paper  &quot;Contrastive Training for Improved<br>Out-of-Distribution Detection&quot; <a href="https://t.co/y4xfRwxZXu">https://t.co/y4xfRwxZXu</a> with <a href="https://twitter.com/jimwinkens?ref_src=twsrc%5Etfw">@jimwinkens</a>, <a href="https://twitter.com/BunelR?ref_src=twsrc%5Etfw">@BunelR</a>, <a href="https://twitter.com/abzz4ssj?ref_src=twsrc%5Etfw">@abzz4ssj</a>,  Robert Stanforth, <a href="https://twitter.com/vivnat?ref_src=twsrc%5Etfw">@vivnat</a>, <a href="https://twitter.com/joe_ledsam?ref_src=twsrc%5Etfw">@joe_ledsam</a>, <a href="https://twitter.com/patmacwilliams?ref_src=twsrc%5Etfw">@patmacwilliams</a>, <a href="https://twitter.com/pushmeet?ref_src=twsrc%5Etfw">@pushmeet</a>, <a href="https://twitter.com/alan_karthi?ref_src=twsrc%5Etfw">@alan_karthi</a>, <a href="https://twitter.com/saakohl?ref_src=twsrc%5Etfw">@saakohl</a>, <a href="https://twitter.com/TaylanCemgilML?ref_src=twsrc%5Etfw">@TaylanCemgilML</a>, <a href="https://twitter.com/arkitus?ref_src=twsrc%5Etfw">@arkitus</a> <a href="https://t.co/L9dHisQuJC">pic.twitter.com/L9dHisQuJC</a></p>&mdash; Olaf Ronneberger (@ORonneberger) <a href="https://twitter.com/ORonneberger/status/1283047741019561985?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper! Joint contrastive and supervised training improves OOD detection performance on the challenging near OOD setting by obtaining a rich and task-agnostic feature space.<a href="https://t.co/gvUxbotWAS">https://t.co/gvUxbotWAS</a><br><br>Thread. <a href="https://t.co/qBvmGlVHik">pic.twitter.com/qBvmGlVHik</a></p>&mdash; Jim Winkens (@jimwinkens) <a href="https://twitter.com/jimwinkens/status/1283072624701181955?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Learning Reasoning Strategies in End-to-End Differentiable Proving

Pasquale Minervini, Sebastian Riedel, Pontus Stenetorp, Edward Grefenstette, Tim Rocktäschel

- retweets: 34, favorites: 118 (07/15/2020 09:58:14)

- links: [abs](https://arxiv.org/abs/2007.06477) | [pdf](https://arxiv.org/pdf/2007.06477)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [cs.SC](https://arxiv.org/list/cs.SC/recent)

Attempts to render deep learning models interpretable, data-efficient, and robust have seen some success through hybridisation with rule-based systems, for example, in Neural Theorem Provers (NTPs). These neuro-symbolic models can induce interpretable rules and learn representations from data via back-propagation, while providing logical explanations for their predictions. However, they are restricted by their computational complexity, as they need to consider all possible proof paths for explaining a goal, thus rendering them unfit for large-scale applications. We present Conditional Theorem Provers (CTPs), an extension to NTPs that learns an optimal rule selection strategy via gradient-based optimisation. We show that CTPs are scalable and yield state-of-the-art results on the CLUTRR dataset, which tests systematic generalisation of neural models by learning to reason over smaller graphs and evaluating on larger ones. Finally, CTPs show better link prediction results on standard benchmarks in comparison with other neural-symbolic models, while being explainable. All source code and datasets are available online, at https://github.com/uclnlp/ctp.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Conditional Theorem Provers are scalable neuro-symbolic reasoning models that learn to recursively select and generate rules on-the-fly conditioned on the goal via gradient-based optimisation! To appear at <a href="https://twitter.com/hashtag/ICML2020?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2020</a>, Arxiv <a href="https://t.co/la93KJWGIr">https://t.co/la93KJWGIr</a> Slide <a href="https://t.co/6qbIGZrsZS">https://t.co/6qbIGZrsZS</a> 1/N <a href="https://t.co/QbqiTaXH24">pic.twitter.com/QbqiTaXH24</a></p>&mdash; Pasquale Minervini (@PMinervini) <a href="https://twitter.com/PMinervini/status/1283007068300021766?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Mixed-state entanglement from local randomized measurements

Andreas Elben, Richard Kueng, Hsin-Yuan Huang, Rick van Bijnen, Christian Kokail, Marcello Dalmonte, Pasquale Calabrese, Barbara Kraus, John Preskill, Peter Zoller, Benoît Vermersch

- retweets: 16, favorites: 115 (07/15/2020 09:58:14)

- links: [abs](https://arxiv.org/abs/2007.06305) | [pdf](https://arxiv.org/pdf/2007.06305)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent)

We propose a method for detecting bipartite entanglement in a many-body mixed state based on estimating moments of the partially transposed density matrix. The estimates are obtained by performing local random measurements on the state, followed by post-processing using the classical shadows framework. Our method can be applied to any quantum system with single-qubit control. We provide a detailed analysis of the required number of experimental runs, and demonstrate the protocol using existing experimental data [Brydges et al, Science 364, 260 (2019)].

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this fun collaboration between <a href="https://twitter.com/IQIM_Caltech?ref_src=twsrc%5Etfw">@IQIM_Caltech</a> &amp; <a href="https://twitter.com/iqoqi?ref_src=twsrc%5Etfw">@IQOQI</a> we proposed a more efficient method for verifying quantum entanglement in a many-body system, applied it to real ion-trap data, and fulfilled my longstanding ambition to lower my Zoller number to 1.<a href="https://t.co/ODg9fmncDB">https://t.co/ODg9fmncDB</a></p>&mdash; John Preskill (@preskill) <a href="https://twitter.com/preskill/status/1282839903491907584?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Illuminating Mario Scenes in the Latent Space of a Generative  Adversarial Network

Matthew C. Fontaine, Ruilin Liu, Julian Togelius, Amy K. Hoover, Stefanos Nikolaidis

- retweets: 14, favorites: 65 (07/15/2020 09:58:14)

- links: [abs](https://arxiv.org/abs/2007.05674) | [pdf](https://arxiv.org/pdf/2007.05674)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Recent developments in machine learning techniques have allowed automatic generation of video game levels that are stylistically similar to human-designed examples. While the output of machine learning models such as generative adversarial networks (GANs) is notoriously hard to control, the recently proposed latent variable evolution (LVE) technique searches the space of GAN parameters to generate outputs that optimize some objective performance metric, such as level playability. However, the question remains on how to automatically generate a diverse range of high-quality solutions based on a prespecified set of desired characteristics. We introduce a new method called latent space illumination (LSI), which uses state-of-the-art quality diversity algorithms designed to optimize in continuous spaces, i.e., MAP-Elites with a directional variation operator and Covariance Matrix Adaptation MAP-Elites, to effectively search the parameter space of theGAN along a set of multiple level mechanics. We show the performance of LSI algorithms in three experiments in SuperMario Bros., a benchmark domain for procedural content generation. Results suggest that LSI generates sets of Mario levels that are reliably mechanically diverse as well as playable.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Illuminating Mario Scenes in the Latent Space of a Generative Adversarial Network<br>pdf: <a href="https://t.co/leKS3Md5eB">https://t.co/leKS3Md5eB</a><br>abs: <a href="https://t.co/9p1bAvA2ip">https://t.co/9p1bAvA2ip</a><br>github: <a href="https://t.co/LF0cpcjOoE">https://t.co/LF0cpcjOoE</a> <a href="https://t.co/PTfBlZdETC">pic.twitter.com/PTfBlZdETC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1282870193849536520?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share some recent work with <a href="https://twitter.com/amykhoover?ref_src=twsrc%5Etfw">@amykhoover</a>, <a href="https://twitter.com/Amidos2006?ref_src=twsrc%5Etfw">@Amidos2006</a>, and <a href="https://twitter.com/togelius?ref_src=twsrc%5Etfw">@togelius</a> on latent space illumination (LSI), a method for exploring the latent space of generative models (such as GANs).<a href="https://t.co/584E4sIvV2">https://t.co/584E4sIvV2</a> <a href="https://t.co/VpxSJZsN2d">pic.twitter.com/VpxSJZsN2d</a></p>&mdash; Matt Fontaine (@tehqin17) <a href="https://twitter.com/tehqin17/status/1282857166446006272?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. S2RMs: Spatially Structured Recurrent Modules

Nasim Rahaman, Anirudh Goyal, Muhammad Waleed Gondal, Manuel Wuthrich, Stefan Bauer, Yash Sharma, Yoshua Bengio, Bernhard Schölkopf

- retweets: 10, favorites: 69 (07/15/2020 09:58:14)

- links: [abs](https://arxiv.org/abs/2007.06533) | [pdf](https://arxiv.org/pdf/2007.06533)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Capturing the structure of a data-generating process by means of appropriate inductive biases can help in learning models that generalize well and are robust to changes in the input distribution. While methods that harness spatial and temporal structures find broad application, recent work has demonstrated the potential of models that leverage sparse and modular structure using an ensemble of sparingly interacting modules. In this work, we take a step towards dynamic models that are capable of simultaneously exploiting both modular and spatiotemporal structures. We accomplish this by abstracting the modeled dynamical system as a collection of autonomous but sparsely interacting sub-systems. The sub-systems interact according to a topology that is learned, but also informed by the spatial structure of the underlying real-world system. This results in a class of models that are well suited for modeling the dynamics of systems that only offer local views into their state, along with corresponding spatial locations of those views. On the tasks of video prediction from cropped frames and multi-agent world modeling from partial observations in the challenging Starcraft2 domain, we find our models to be more robust to the number of available views and better capable of generalization to novel tasks without additional training, even when compared against strong baselines that perform equally well or better on the training distribution.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work on Spatially Structured Recurrent Modules! With <a href="https://twitter.com/anirudhg9119?ref_src=twsrc%5Etfw">@anirudhg9119</a>, <a href="https://twitter.com/Wallii_gondal?ref_src=twsrc%5Etfw">@Wallii_gondal</a>, Manuel Wuthrich, Stefan Bauer, <a href="https://twitter.com/yash_j_sharma?ref_src=twsrc%5Etfw">@yash_j_sharma</a>, Yoshua Bengio &amp; <a href="https://twitter.com/bschoelkopf?ref_src=twsrc%5Etfw">@bschoelkopf</a>: <a href="https://t.co/UTn1RQNjga">https://t.co/UTn1RQNjga</a><br><br>It’s all about marrying modular with spatial structures: a thread. 1/5 <a href="https://t.co/zxGXZCW4ef">pic.twitter.com/zxGXZCW4ef</a></p>&mdash; Nasim (@nasim_rahaman) <a href="https://twitter.com/nasim_rahaman/status/1282937311836209153?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work out! &quot;Spatially Structured Recurrent Modules&quot; <br><br>Led by <a href="https://twitter.com/nasim_rahaman?ref_src=twsrc%5Etfw">@nasim_rahaman</a>. Along with <a href="https://twitter.com/Wallii_gondal?ref_src=twsrc%5Etfw">@Wallii_gondal</a> , Manuel Wuthrich, Stefan Bauer, Yash Sharma, Yoshua Bengio &amp; <a href="https://twitter.com/bschoelkopf?ref_src=twsrc%5Etfw">@bschoelkopf</a> <a href="https://t.co/R5DlAiWa9Q">https://t.co/R5DlAiWa9Q</a> <a href="https://t.co/triGWUAiQa">https://t.co/triGWUAiQa</a></p>&mdash; Anirudh Goyal (@anirudhg9119) <a href="https://twitter.com/anirudhg9119/status/1283072807585423361?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. PA-GAN: Progressive Attention Generative Adversarial Network for Facial  Attribute Editing

Zhenliang He, Meina Kan, Jichao Zhang, Shiguang Shan

- retweets: 16, favorites: 55 (07/15/2020 09:58:15)

- links: [abs](https://arxiv.org/abs/2007.05892) | [pdf](https://arxiv.org/pdf/2007.05892)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Facial attribute editing aims to manipulate attributes on the human face, e.g., adding a mustache or changing the hair color. Existing approaches suffer from a serious compromise between correct attribute generation and preservation of the other information such as identity and background, because they edit the attributes in the imprecise area. To resolve this dilemma, we propose a progressive attention GAN (PA-GAN) for facial attribute editing. In our approach, the editing is progressively conducted from high to low feature level while being constrained inside a proper attribute area by an attention mask at each level. This manner prevents undesired modifications to the irrelevant regions from the beginning, and then the network can focus more on correctly generating the attributes within a proper boundary at each level. As a result, our approach achieves correct attribute editing with irrelevant details much better preserved compared with the state-of-the-arts. Codes are released at https://github.com/LynnHo/PA-GAN-Tensorflow.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PA-GAN: Progressive Attention Generative Adversarial Network for Facial Attribute Editing<br>pdf: <a href="https://t.co/fNRpEtSHKf">https://t.co/fNRpEtSHKf</a><br>abs: <a href="https://t.co/DqQDkuTk5k">https://t.co/DqQDkuTk5k</a> <a href="https://t.co/CFlfOoVGU9">pic.twitter.com/CFlfOoVGU9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1282844812471934976?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Tabletop Roleplaying Games as Procedural Content Generators

Matthew Guzdial, Devi Acharya, Max Kreminski, Michael Cook, Mirjam Eladhari, Antonios Liapis, Anne Sullivan

- retweets: 16, favorites: 49 (07/15/2020 09:58:15)

- links: [abs](https://arxiv.org/abs/2007.06108) | [pdf](https://arxiv.org/pdf/2007.06108)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Tabletop roleplaying games (TTRPGs) and procedural content generators can both be understood as systems of rules for producing content. In this paper, we argue that TTRPG design can usefully be viewed as procedural content generator design. We present several case studies linking key concepts from PCG research -- including possibility spaces, expressive range analysis, and generative pipelines -- to key concepts in TTRPG design. We then discuss the implications of these relationships and suggest directions for future work uniting research in TTRPGs and PCG.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our PCG Workshop paper on how we can understand Tabletop Roleplaying Games as Procedural Content Generation is now up on arXiv. Work w/ <a href="https://twitter.com/dacharya64?ref_src=twsrc%5Etfw">@dacharya64</a> <a href="https://twitter.com/maxkreminski?ref_src=twsrc%5Etfw">@maxkreminski</a> <a href="https://twitter.com/mtrc?ref_src=twsrc%5Etfw">@mtrc</a> <a href="https://twitter.com/MirjamPE?ref_src=twsrc%5Etfw">@MirjamPE</a> <a href="https://twitter.com/SentientDesigns?ref_src=twsrc%5Etfw">@SentientDesigns</a> and <a href="https://twitter.com/annetropy?ref_src=twsrc%5Etfw">@annetropy</a>!<a href="https://t.co/9LbRojLKx5">https://t.co/9LbRojLKx5</a></p>&mdash; Matthew Guzdial (@MatthewGuz) <a href="https://twitter.com/MatthewGuz/status/1282846352536293377?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. The Future of Work Is Here: Toward a Comprehensive Approach to  Artificial Intelligence and Labour

Julian Posada

- retweets: 14, favorites: 42 (07/15/2020 09:58:15)

- links: [abs](https://arxiv.org/abs/2007.05843) | [pdf](https://arxiv.org/pdf/2007.05843)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

This commentary traces contemporary discourses on the relationship between artificial intelligence and labour and explains why these principles must be comprehensive in their approach to labour and AI. First, the commentary asserts that ethical frameworks in AI alone are not enough to guarantee workers' rights since they lack enforcement mechanisms and the representation of different stakeholders. Secondly, it argues that current discussions on AI and labour focus on the deployment of these technologies in the workplace but ignore the essential role of human labour in their development, particularly in the different cases of outsourced labour around the world. Finally, it recommends using existing human rights frameworks for working conditions to provide more comprehensive ethical principles and regulations. The commentary concludes by arguing that the central question regarding the future of work should not be whether intelligent machines will replace humans, but who will own these systems and have a say in their development and operation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share my contribution to the recent <a href="https://twitter.com/UofTEthics?ref_src=twsrc%5Etfw">@UofTEthics</a>’s conference on <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> and the <a href="https://twitter.com/hashtag/FutureOfWork?src=hash&amp;ref_src=twsrc%5Etfw">#FutureOfWork</a>. It’s about how ethical AI frameworks and government strategies need a comprehensive approach to labour and AI <a href="https://t.co/9mdwHSFUTC">https://t.co/9mdwHSFUTC</a> (1/9) <a href="https://t.co/5nCpoHm7H4">pic.twitter.com/5nCpoHm7H4</a></p>&mdash; Julian Posada (@JulianPosada0) <a href="https://twitter.com/JulianPosada0/status/1283058375266377735?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. State Space Expectation Propagation: Efficient Inference Schemes for  Temporal Gaussian Processes

William J. Wilkinson, Paul E. Chang, Michael Riis Andersen, Arno Solin

- retweets: 6, favorites: 49 (07/15/2020 09:58:15)

- links: [abs](https://arxiv.org/abs/2007.05994) | [pdf](https://arxiv.org/pdf/2007.05994)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We formulate approximate Bayesian inference in non-conjugate temporal and spatio-temporal Gaussian process models as a simple parameter update rule applied during Kalman smoothing. This viewpoint encompasses most inference schemes, including expectation propagation (EP), the classical (Extended, Unscented, etc.) Kalman smoothers, and variational inference. We provide a unifying perspective on these algorithms, showing how replacing the power EP moment matching step with linearisation recovers the classical smoothers. EP provides some benefits over the traditional methods via introduction of the so-called cavity distribution, and we combine these benefits with the computational efficiency of linearisation, providing extensive empirical analysis demonstrating the efficacy of various algorithms under this unifying framework. We provide a fast implementation of all methods in JAX.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our work on inference in spatio-temporal Gaussian processes is at ICML!<br><br>Classical smoothing and EP / VI under one paradigm.<br><br>Fast learning: JAX is great at autodiff-ing big loops!<br><br>with <a href="https://twitter.com/edchangy?ref_src=twsrc%5Etfw">@edchangy</a> <a href="https://twitter.com/Michael_riis?ref_src=twsrc%5Etfw">@Michael_riis</a> <a href="https://twitter.com/arnosolin?ref_src=twsrc%5Etfw">@arnosolin</a><br>Paper <a href="https://t.co/JVZqMl2dFy">https://t.co/JVZqMl2dFy</a><br>Code <a href="https://t.co/lgxy6OYYXV">https://t.co/lgxy6OYYXV</a> <a href="https://t.co/TekF6zl7m1">pic.twitter.com/TekF6zl7m1</a></p>&mdash; Will Wilkinson (@wil_j_wil) <a href="https://twitter.com/wil_j_wil/status/1282934217953902597?ref_src=twsrc%5Etfw">July 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



