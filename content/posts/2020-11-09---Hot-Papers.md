---
title: Hot Papers 2020-11-09
date: 2020-11-10T09:24:39.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-09"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-09"
socialImage: "/media/flying-marine.jpg"

---

# 1. Underspecification Presents Challenges for Credibility in Modern Machine  Learning

Alexander D'Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D. Hoffman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yian Ma, Cory McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas F. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouff, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai, D. Sculley

- retweets: 3367, favorites: 297 (11/10/2020 09:24:39)

- links: [abs](https://arxiv.org/abs/2011.03395) | [pdf](https://arxiv.org/pdf/2011.03395)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

ML models often exhibit unexpectedly poor behavior when they are deployed in real-world domains. We identify underspecification as a key reason for these failures. An ML pipeline is underspecified when it can return many predictors with equivalently strong held-out performance in the training domain. Underspecification is common in modern ML pipelines, such as those based on deep learning. Predictors returned by underspecified pipelines are often treated as equivalent based on their training domain performance, but we show here that such predictors can behave very differently in deployment domains. This ambiguity can lead to instability and poor model behavior in practice, and is a distinct failure mode from previously identified issues arising from structural mismatch between training and deployment domains. We show that this problem appears in a wide variety of practical ML pipelines, using examples from computer vision, medical imaging, natural language processing, clinical risk prediction based on electronic health records, and medical genomics. Our results show the need to explicitly account for underspecification in modeling pipelines that are intended for real-world deployment in any domain.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NEW from a big collaboration at Google: Underspecification Presents Challenges for Credibility in Modern Machine Learning<br><br>Explores a common failure mode when applying ML to real-world problems. 🧵 1/14<a href="https://t.co/7vX5D8yMhq">https://t.co/7vX5D8yMhq</a> <a href="https://t.co/AqtoNBGzd5">pic.twitter.com/AqtoNBGzd5</a></p>&mdash; Alexander D&#39;Amour (@alexdamour) <a href="https://twitter.com/alexdamour/status/1325921856738701312?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">機械学習モデルが実問題適用時に想定外に性能劣化する問題の多くは、評価データで同じ性能を達成する解が複数ある解不定性（underspecification）が主原因であり、解集合からの解選択（周辺化では不十分）、利用想定したストレステストの設計、タスク特化の正則化が重要。 <a href="https://t.co/cA1wgA1Pq2">https://t.co/cA1wgA1Pq2</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1325934973447106561?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Underspecification Presents Challenges for Credibility in Modern Machine Learning<br><br>Massive collaboration by Googlers to show underspecified ML pipeline can lead to various instability and poor model behavior, incl. shortcuts and spurious correlations.<a href="https://t.co/fpJQuiMLiP">https://t.co/fpJQuiMLiP</a> <a href="https://t.co/uNk1L9H7v3">pic.twitter.com/uNk1L9H7v3</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1325621820750327808?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Modular Primitives for High-Performance Differentiable Rendering

Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, Timo Aila

- retweets: 814, favorites: 120 (11/10/2020 09:24:40)

- links: [abs](https://arxiv.org/abs/2011.03277) | [pdf](https://arxiv.org/pdf/2011.03277)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a modular differentiable renderer design that yields performance superior to previous methods by leveraging existing, highly optimized hardware graphics pipelines. Our design supports all crucial operations in a modern graphics pipeline: rasterizing large numbers of triangles, attribute interpolation, filtered texture lookups, as well as user-programmable shading and geometry processing, all in high resolutions. Our modular primitives allow custom, high-performance graphics pipelines to be built directly within automatic differentiation frameworks such as PyTorch or TensorFlow. As a motivating application, we formulate facial performance capture as an inverse rendering problem and show that it can be solved efficiently using our tools. Our results indicate that this simple and straightforward approach achieves excellent geometric correspondence between rendered results and reference imagery.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Modular Primitives for High-Performance Differentiable Rendering<br>pdf: <a href="https://t.co/ZxEH9LWNo2">https://t.co/ZxEH9LWNo2</a><br>abs: <a href="https://t.co/eQV4Ty6zdJ">https://t.co/eQV4Ty6zdJ</a><br>github: <a href="https://t.co/H59YVj58ym">https://t.co/H59YVj58ym</a> <a href="https://t.co/PsIoGsv49p">pic.twitter.com/PsIoGsv49p</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1325614648524234752?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Complex Query Answering with Neural Link Predictors

Erik Arakelyan, Daniel Daza, Pasquale Minervini, Michael Cochez

- retweets: 758, favorites: 98 (11/10/2020 09:24:40)

- links: [abs](https://arxiv.org/abs/2011.03459) | [pdf](https://arxiv.org/pdf/2011.03459)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LO](https://arxiv.org/list/cs.LO/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Neural link predictors are immensely useful for identifying missing edges in large scale Knowledge Graphs. However, it is still not clear how to use these models for answering more complex queries that arise in a number of domains, such as queries using logical conjunctions, disjunctions, and existential quantifiers, while accounting for missing edges. In this work, we propose a framework for efficiently answering complex queries on incomplete Knowledge Graphs. We translate each query into an end-to-end differentiable objective, where the truth value of each atom is computed by a pre-trained neural link predictor. We then analyse two solutions to the optimisation problem, including gradient-based and combinatorial search. In our experiments, the proposed approach produces more accurate results than state-of-the-art methods -- black-box neural models trained on millions of generated queries -- without the need of training on a large and diverse set of complex queries. Using orders of magnitude less training data, we obtain relative improvements ranging from 8% up to 40% in Hits@3 across different knowledge graphs containing factual information. Finally, we demonstrate that it is possible to explain the outcome of our model in terms of the intermediate solutions identified for each of the complex query atoms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do we need deep black-box models to answer complex logical queries in KGs?<br>We show how a neural link predictor can be used to produce more accurate results than SOTA models while improving explanations! w/ <a href="https://twitter.com/_kire_kara_?ref_src=twsrc%5Etfw">@_kire_kara_</a> <a href="https://twitter.com/danieldazac?ref_src=twsrc%5Etfw">@danieldazac</a> <a href="https://twitter.com/michaelcochez?ref_src=twsrc%5Etfw">@michaelcochez</a>, arxiv: <a href="https://t.co/eWA2QlskUr">https://t.co/eWA2QlskUr</a> <a href="https://t.co/eJfMLOx0oT">pic.twitter.com/eJfMLOx0oT</a></p>&mdash; Pasquale Minervini (@PMinervini) <a href="https://twitter.com/PMinervini/status/1325756402938343424?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Disentangling 3D Prototypical Networks For Few-Shot Concept Learning

Mihir Prabhudesai, Shamit Lal, Darshan Patil, Hsiao-Yu Tung, Adam W Harley, Katerina Fragkiadaki

- retweets: 576, favorites: 106 (11/10/2020 09:24:40)

- links: [abs](https://arxiv.org/abs/2011.03367) | [pdf](https://arxiv.org/pdf/2011.03367)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present neural architectures that disentangle RGB-D images into objects' shapes and styles and a map of the background scene, and explore their applications for few-shot 3D object detection and few-shot concept classification. Our networks incorporate architectural biases that reflect the image formation process, 3D geometry of the world scene, and shape-style interplay. They are trained end-to-end self-supervised by predicting views in static scenes, alongside a small number of 3D object boxes. Objects and scenes are represented in terms of 3D feature grids in the bottleneck of the network. We show that the proposed 3D neural representations are compositional: they can generate novel 3D scene feature maps by mixing object shapes and styles, resizing and adding the resulting object 3D feature maps over background scene feature maps. We show that classifiers for object categories, color, materials, and spatial relationships trained over the disentangled 3D feature sub-spaces generalize better with dramatically fewer examples than the current state-of-the-art, and enable a visual question answering system that uses them as its modules to generalize one-shot to novel objects in the scene.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Disentangling 3D Prototypical Networks For Few-Shot Concept Learning<br>pdf: <a href="https://t.co/lR2jqohJKq">https://t.co/lR2jqohJKq</a><br>abs: <a href="https://t.co/XdNiOIoiDv">https://t.co/XdNiOIoiDv</a><br>project page: <a href="https://t.co/tdB2z6rsFL">https://t.co/tdB2z6rsFL</a> <a href="https://t.co/8QG1oPa2fb">pic.twitter.com/8QG1oPa2fb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1325640317442412547?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Large-scale multilingual audio visual dubbing

Yi Yang, Brendan Shillingford, Yannis Assael, Miaosen Wang, Wendi Liu, Yutian Chen, Yu Zhang, Eren Sezener, Luis C. Cobo, Misha Denil, Yusuf Aytar, Nando de Freitas

- retweets: 190, favorites: 60 (11/10/2020 09:24:40)

- links: [abs](https://arxiv.org/abs/2011.03530) | [pdf](https://arxiv.org/pdf/2011.03530)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We describe a system for large-scale audiovisual translation and dubbing, which translates videos from one language to another. The source language's speech content is transcribed to text, translated, and automatically synthesized into target language speech using the original speaker's voice. The visual content is translated by synthesizing lip movements for the speaker to match the translated audio, creating a seamless audiovisual experience in the target language. The audio and visual translation subsystems each contain a large-scale generic synthesis model trained on thousands of hours of data in the corresponding domain. These generic models are fine-tuned to a specific speaker before translation, either using an auxiliary corpus of data from the target speaker, or using the video to be translated itself as the input to the fine-tuning process. This report gives an architectural overview of the full system, as well as an in-depth discussion of the video dubbing component. The role of the audio and text components in relation to the full system is outlined, but their design is not discussed in detail. Translated and dubbed demo videos generated using our system can be viewed at https://www.youtube.com/playlist?list=PLSi232j2ZA6_1Exhof5vndzyfbxAhhEs5

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here is our latest work on automated video translation <a href="https://t.co/p8OLpi1YLH">https://t.co/p8OLpi1YLH</a>! <a href="https://twitter.com/yangyi02?ref_src=twsrc%5Etfw">@yangyi02</a>, <a href="https://twitter.com/BrendanShilling?ref_src=twsrc%5Etfw">@BrendanShilling</a>, <a href="https://twitter.com/MiaosenWang?ref_src=twsrc%5Etfw">@MiaosenWang</a>, Wendi Liu, <a href="https://twitter.com/yutianc?ref_src=twsrc%5Etfw">@yutianc</a>,  Yu Zhang, Eren Sezener, Luis C. Cobo, <a href="https://twitter.com/notmisha?ref_src=twsrc%5Etfw">@notmisha</a>, <a href="https://twitter.com/yusufaytar?ref_src=twsrc%5Etfw">@yusufaytar</a>, <a href="https://twitter.com/NandoDF?ref_src=twsrc%5Etfw">@NandoDF</a></p>&mdash; Yannis Assael (@iassael) <a href="https://twitter.com/iassael/status/1325877249804021768?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="it" dir="ltr">Large-scale multilingual audio visual dubbing<br>pdf: <a href="https://t.co/9ig4Ct7PhC">https://t.co/9ig4Ct7PhC</a><br>abs: <a href="https://t.co/G1h17gMPHR">https://t.co/G1h17gMPHR</a> <a href="https://t.co/IhLpMyy7hb">pic.twitter.com/IhLpMyy7hb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1325617201970360321?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. "What's This?" -- Learning to Segment Unknown Objects from Manipulation  Sequences

Wout Boerdijk, Martin Sundermeyer, Maximilian Durner, Rudolph Triebel

- retweets: 42, favorites: 42 (11/10/2020 09:24:41)

- links: [abs](https://arxiv.org/abs/2011.03279) | [pdf](https://arxiv.org/pdf/2011.03279)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a novel framework for self-supervised grasped object segmentation with a robotic manipulator. Our method successively learns an agnostic foreground segmentation followed by a distinction between manipulator and object solely by observing the motion between consecutive RGB frames. In contrast to previous approaches, we propose a single, end-to-end trainable architecture which jointly incorporates motion cues and semantic knowledge. Furthermore, while the motion of the manipulator and the object are substantial cues for our algorithm, we present means to robustly deal with distraction objects moving in the background, as well as with completely static scenes. Our method neither depends on any visual registration of a kinematic robot or 3D object models, nor on precise hand-eye calibration or any additional sensor data. By extensive experimental evaluation we demonstrate the superiority of our framework and provide detailed insights on its capability of dealing with the aforementioned extreme cases of motion. We also show that training a semantic segmentation network with the automatically labeled data achieves results on par with manually annotated training data. Code and pretrained models will be made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">“What’s This?” - Learning to Segment Unknown Objects from Manipulation Sequences<a href="https://t.co/bc8mqhMSIm">https://t.co/bc8mqhMSIm</a> <a href="https://t.co/QgRzdwwmRU">pic.twitter.com/QgRzdwwmRU</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1325686436964622337?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer

Daniel Ho, Kanishka Rao, Zhuo Xu, Eric Jang, Mohi Khansari, Yunfei Bai

- retweets: 22, favorites: 60 (11/10/2020 09:24:41)

- links: [abs](https://arxiv.org/abs/2011.03148) | [pdf](https://arxiv.org/pdf/2011.03148)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

The success of deep reinforcement learning (RL) and imitation learning (IL) in vision-based robotic manipulation typically hinges on the expense of large scale data collection. With simulation, data to train a policy can be collected efficiently at scale, but the visual gap between sim and real makes deployment in the real world difficult. We introduce RetinaGAN, a generative adversarial network (GAN) approach to adapt simulated images to realistic ones with object-detection consistency. RetinaGAN is trained in an unsupervised manner without task loss dependencies, and preserves general object structure and texture in adapted images. We evaluate our method on three real world tasks: grasping, pushing, and door opening. RetinaGAN improves upon the performance of prior sim-to-real methods for RL-based object instance grasping and continues to be effective even in the limited data regime. When applied to a pushing task in a similar visual domain, RetinaGAN demonstrates transfer with no additional real data requirements. We also show our method bridges the visual gap for a novel door opening task using imitation learning in a new visual domain. Visit the project website at https://retinagan.github.io/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer<br>pdf: <a href="https://t.co/w5rPnI0fp4">https://t.co/w5rPnI0fp4</a><br>abs: <a href="https://t.co/XLgicF2bfO">https://t.co/XLgicF2bfO</a> <a href="https://t.co/6HbFTGH8PG">pic.twitter.com/6HbFTGH8PG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1325628899473190913?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RetinaGAN: An Object-aware Approach to Sim-to-Real Transfer<a href="https://t.co/l8C7lo94LC">https://t.co/l8C7lo94LC</a> <a href="https://t.co/IHuZ0ZbM9i">pic.twitter.com/IHuZ0ZbM9i</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1325827793553252355?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. RealAnt: An Open-Source Low-Cost Quadruped for Research in Real-World  Reinforcement Learning

Rinu Boney, Jussi Sainio, Mikko Kaivola, Arno Solin, Juho Kannala

- retweets: 60, favorites: 21 (11/10/2020 09:24:41)

- links: [abs](https://arxiv.org/abs/2011.03085) | [pdf](https://arxiv.org/pdf/2011.03085)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Current robot platforms available for research are either very expensive or unable to handle the abuse of exploratory controls in reinforcement learning. We develop RealAnt, a minimal low-cost physical version of the popular 'Ant' benchmark used in reinforcement learning. RealAnt costs only $410 in materials and can be assembled in less than an hour. We validate the platform with reinforcement learning experiments and provide baseline results on a set of benchmark tasks. We demonstrate that the TD3 algorithm can learn to walk the RealAnt from less than 45 minutes of experience. We also provide simulator versions of the robot (with the same dimensions, state-action spaces, and delayed noisy observations) in the MuJoCo and PyBullet simulators. We open-source hardware designs, supporting software, and baseline results for ease of reproducibility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🐜 3D print your own quadruped Ant for <a href="https://twitter.com/hashtag/RL?src=hash&amp;ref_src=twsrc%5Etfw">#RL</a> 🐜 <br>&quot;RealAnt: An Open-Source Low-Cost Quadruped for Research in Real-World Reinforcement Learning&quot; (\w <a href="https://twitter.com/rinuboney?ref_src=twsrc%5Etfw">@rinuboney</a>, <a href="https://twitter.com/JussiSainio?ref_src=twsrc%5Etfw">@JussiSainio</a>, Mikko, and Juho)<br>📄: <a href="https://t.co/iMoDKJ3cyQ">https://t.co/iMoDKJ3cyQ</a><br>🖨: <a href="https://t.co/NPb1LF7nRI">https://t.co/NPb1LF7nRI</a> <br>🧮: <a href="https://t.co/bZV1pWCO3x">https://t.co/bZV1pWCO3x</a> <a href="https://t.co/KHU3jc7B47">pic.twitter.com/KHU3jc7B47</a></p>&mdash; Arno Solin (@arnosolin) <a href="https://twitter.com/arnosolin/status/1325731356190629888?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. STReSSD: Sim-To-Real from Sound for Stochastic Dynamics

Carolyn Matl, Yashraj Narang, Dieter Fox, Ruzena Bajcsy, Fabio Ramos

- retweets: 36, favorites: 22 (11/10/2020 09:24:41)

- links: [abs](https://arxiv.org/abs/2011.03136) | [pdf](https://arxiv.org/pdf/2011.03136)
- [cs.RO](https://arxiv.org/list/cs.RO/recent)

Sound is an information-rich medium that captures dynamic physical events. This work presents STReSSD, a framework that uses sound to bridge the simulation-to-reality gap for stochastic dynamics, demonstrated for the canonical case of a bouncing ball. A physically-motivated noise model is presented to capture stochastic behavior of the balls upon collision with the environment. A likelihood-free Bayesian inference framework is used to infer the parameters of the noise model, as well as a material property called the coefficient of restitution, from audio observations. The same inference framework and the calibrated stochastic simulator are then used to learn a probabilistic model of ball dynamics. The predictive capabilities of the dynamics model are tested in two robotic experiments. First, open-loop predictions anticipate probabilistic success of bouncing a ball into a cup. The second experiment integrates audio perception with a robotic arm to track and deflect a bouncing ball in real-time. We envision that this work is a step towards integrating audio-based inference for dynamic robotic tasks. Experimental results can be viewed at https://youtu.be/b7pOrgZrArk.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">STReSSD: Sim-To-Real from Sound for Stochastic Dynamics<br>pdf: <a href="https://t.co/MIPPHRurws">https://t.co/MIPPHRurws</a><br>abs: <a href="https://t.co/lrTcHdfImA">https://t.co/lrTcHdfImA</a><br>video: <a href="https://t.co/bKQp2KoYBm">https://t.co/bKQp2KoYBm</a> <a href="https://t.co/Cew1Nv8I2M">pic.twitter.com/Cew1Nv8I2M</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1325627914365362176?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Efficient quantum algorithm for dissipative nonlinear differential  equations

Jin-Peng Liu, Herman Øie Kolden, Hari K. Krovi, Nuno F. Loureiro, Konstantina Trivisa, Andrew M. Childs

- retweets: 20, favorites: 36 (11/10/2020 09:24:41)

- links: [abs](https://arxiv.org/abs/2011.03185) | [pdf](https://arxiv.org/pdf/2011.03185)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [math.NA](https://arxiv.org/list/math.NA/recent) | [physics.plasm-ph](https://arxiv.org/list/physics.plasm-ph/recent)

While there has been extensive previous work on efficient quantum algorithms for linear differential equations, analogous progress for nonlinear differential equations has been severely limited due to the linearity of quantum mechanics. Despite this obstacle, we develop a quantum algorithm for initial value problems described by dissipative quadratic $n$-dimensional ordinary differential equations. Assuming $R < 1$, where $R$ is a parameter characterizing the ratio of the nonlinearity to the linear dissipation, this algorithm has complexity $T^2\mathrm{poly}(\log T, \log n)/\epsilon$, where $T$ is the evolution time and $\epsilon$ is the allowed error in the output quantum state. This is an exponential improvement over the best previous quantum algorithms, whose complexity is exponential in $T$. We achieve this improvement using the method of Carleman linearization, for which we give an improved convergence theorem. This method maps a system of nonlinear differential equations to an infinite-dimensional system of linear differential equations, which we discretize, truncate, and solve using the forward Euler method and the quantum linear system algorithm. We also provide a lower bound on the worst-case complexity of quantum algorithms for general quadratic differential equations, showing that the problem is intractable for $R \ge \sqrt{2}$. Finally, we discuss potential applications of this approach to problems arising in biology as well as in fluid and plasma dynamics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper with <a href="https://twitter.com/JinPengLiu__Sky?ref_src=twsrc%5Etfw">@JinPengLiu__Sky</a>, Kolden, Krovi, Loureiro, and Trivisa gives an efficient quantum for nonlinear differential equations with strong enough dissipation, shows hardness for weak dissipation. [1/2] <a href="https://t.co/RguERPF8H4">https://t.co/RguERPF8H4</a></p>&mdash; Andrew Childs (@andrewmchilds) <a href="https://twitter.com/andrewmchilds/status/1325811929475129345?ref_src=twsrc%5Etfw">November 9, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



