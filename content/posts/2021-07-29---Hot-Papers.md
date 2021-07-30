---
title: Hot Papers 2021-07-29
date: 2021-07-30T10:40:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. Neural Rays for Occlusion-aware Image-based Rendering

Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, Wenping Wang

- retweets: 276, favorites: 130 (07/30/2021 10:40:36)

- links: [abs](https://arxiv.org/abs/2107.13421) | [pdf](https://arxiv.org/pdf/2107.13421)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a new neural representation, called Neural Ray (NeuRay), for the novel view synthesis (NVS) task with multi-view images as input. Existing neural scene representations for solving the NVS problem, such as NeRF, cannot generalize to new scenes and take an excessively long time on training on each new scene from scratch. The other subsequent neural rendering methods based on stereo matching, such as PixelNeRF, SRF and IBRNet are designed to generalize to unseen scenes but suffer from view inconsistency in complex scenes with self-occlusions. To address these issues, our NeuRay method represents every scene by encoding the visibility of rays associated with the input views. This neural representation can efficiently be initialized from depths estimated by external MVS methods, which is able to generalize to new scenes and achieves satisfactory rendering images without any training on the scene. Then, the initialized NeuRay can be further optimized on every scene with little training timing to enforce spatial coherence to ensure view consistency in the presence of severe self-occlusion. Experiments demonstrate that NeuRay can quickly generate high-quality novel view images of unseen scenes with little finetuning and can handle complex scenes with severe self-occlusions which previous methods struggle with.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Rays for Occlusion-aware Image-based Rendering<br>pdf: <a href="https://t.co/m8Xrd1Uz8a">https://t.co/m8Xrd1Uz8a</a><br>abs: <a href="https://t.co/EEr8tnfeAQ">https://t.co/EEr8tnfeAQ</a><br>project page: <a href="https://t.co/es33FOHiuu">https://t.co/es33FOHiuu</a> <a href="https://t.co/b6zBmK5OEV">pic.twitter.com/b6zBmK5OEV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420547353799450627?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our work &quot;Neural Rays for Occlusion-aware Image-based rendering&quot; for the novel-view-synthesis task. <br>Paper: <a href="https://t.co/3sClmTXS2J">https://t.co/3sClmTXS2J</a> <br>Project page: <a href="https://t.co/74vX08pf3u">https://t.co/74vX08pf3u</a><br>(Results below are generated without per-scene training.) <a href="https://t.co/EdRLk1Osqh">pic.twitter.com/EdRLk1Osqh</a></p>&mdash; Yuan Liu (@YuanLiu41955461) <a href="https://twitter.com/YuanLiu41955461/status/1420734622359453697?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Causal Support: Modeling Causal Inferences with Visualizations

Alex Kale, Yifan Wu, Jessica Hullman

- retweets: 272, favorites: 70 (07/30/2021 10:40:36)

- links: [abs](https://arxiv.org/abs/2107.13485) | [pdf](https://arxiv.org/pdf/2107.13485)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Analysts often make visual causal inferences about possible data-generating models. However, visual analytics (VA) software tends to leave these models implicit in the mind of the analyst, which casts doubt on the statistical validity of informal visual "insights". We formally evaluate the quality of causal inferences from visualizations by adopting causal support -- a Bayesian cognition model that learns the probability of alternative causal explanations given some data -- as a normative benchmark for causal inferences. We contribute two experiments assessing how well crowdworkers can detect (1) a treatment effect and (2) a confounding relationship. We find that chart users' causal inferences tend to be insensitive to sample size such that they deviate from our normative benchmark. While interactively cross-filtering data in visualizations can improve sensitivity, on average users do not perform reliably better with common visualizations than they do with textual contingency tables. These experiments demonstrate the utility of causal support as an evaluation framework for inferences in VA and point to opportunities to make analysts' mental models more explicit in VA software.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I’m excited to share my VIS 2021 paper with <a href="https://twitter.com/JessicaHullman?ref_src=twsrc%5Etfw">@JessicaHullman</a> and <a href="https://twitter.com/yifanwu?ref_src=twsrc%5Etfw">@yifanwu</a> titled, “Causal Support: Modeling Causal Inferences with Visualizations”<a href="https://t.co/f0ei1Iq0AZ">https://t.co/f0ei1Iq0AZ</a><br>1/</p>&mdash; Alex Kale (@AlexKale17) <a href="https://twitter.com/AlexKale17/status/1420762962437017603?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Neural Rule-Execution Tracking Machine For Transformer-Based Text  Generation

Yufei Wang, Can Xu, Huang Hu, Chongyang Tao, Stephen Wan, Mark Dras, Mark Johnson, Daxin Jiang

- retweets: 238, favorites: 99 (07/30/2021 10:40:36)

- links: [abs](https://arxiv.org/abs/2107.13077) | [pdf](https://arxiv.org/pdf/2107.13077)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Sequence-to-Sequence (S2S) neural text generation models, especially the pre-trained ones (e.g., BART and T5), have exhibited compelling performance on various natural language generation tasks. However, the black-box nature of these models limits their application in tasks where specific rules (e.g., controllable constraints, prior knowledge) need to be executed. Previous works either design specific model structure (e.g., Copy Mechanism corresponding to the rule "the generated output should include certain words in the source input") or implement specialized inference algorithm (e.g., Constrained Beam Search) to execute particular rules through the text generation. These methods require careful design case-by-case and are difficult to support multiple rules concurrently. In this paper, we propose a novel module named Neural Rule-Execution Tracking Machine that can be equipped into various transformer-based generators to leverage multiple rules simultaneously to guide the neural generation model for superior generation performance in a unified and scalable way. Extensive experimental results on several benchmarks verify the effectiveness of our proposed model in both controllable and general text generation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An interesting general approach to augment a transformer with logical predicates to control attributes of generation<br>Neural Rule-Execution Tracking Machine For Transformer-Based Text Generation: Yufei Wang, Can Xu, …, Mark Dras, Mark Johnson, Daxin Jiang<a href="https://t.co/Tkv89KsGaL">https://t.co/Tkv89KsGaL</a> <a href="https://t.co/zUdFhPA2ok">pic.twitter.com/zUdFhPA2ok</a></p>&mdash; Stanford NLP Group (@stanfordnlp) <a href="https://twitter.com/stanfordnlp/status/1420570721344917506?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Rule-Execution Tracking Machine For Transformer-Based Text Generation<br>pdf: <a href="https://t.co/UgFUIvLwUU">https://t.co/UgFUIvLwUU</a><br>abs: <a href="https://t.co/jhpkwTteGR">https://t.co/jhpkwTteGR</a><br><br>can be equipped into various transformer-based generators to leverage multiple rules simultaneously to guide the neural generation model <a href="https://t.co/BOPKkpZsIw">pic.twitter.com/BOPKkpZsIw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420544955118936070?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Growing knowledge culturally across generations to solve novel, complex  tasks

Michael Henry Tessler, Pedro A. Tsividis, Jason Madeano, Brin Harper, Joshua B. Tenenbaum

- retweets: 84, favorites: 62 (07/30/2021 10:40:37)

- links: [abs](https://arxiv.org/abs/2107.13377) | [pdf](https://arxiv.org/pdf/2107.13377)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Knowledge built culturally across generations allows humans to learn far more than an individual could glean from their own experience in a lifetime. Cultural knowledge in turn rests on language: language is the richest record of what previous generations believed, valued, and practiced. The power and mechanisms of language as a means of cultural learning, however, are not well understood. We take a first step towards reverse-engineering cultural learning through language. We developed a suite of complex high-stakes tasks in the form of minimalist-style video games, which we deployed in an iterated learning paradigm. Game participants were limited to only two attempts (two lives) to beat each game and were allowed to write a message to a future participant who read the message before playing. Knowledge accumulated gradually across generations, allowing later generations to advance further in the games and perform more efficient actions. Multigenerational learning followed a strikingly similar trajectory to individuals learning alone with an unlimited number of lives. These results suggest that language provides a sufficient medium to express and accumulate the knowledge people acquire in these diverse tasks: the dynamics of the environment, valuable goals, dangerous risks, and strategies for success. The video game paradigm we pioneer here is thus a rich test bed for theories of cultural transmission and learning from language.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Here’s our (Tsividis, Madeano, Harper, Tenenbaum) <a href="https://twitter.com/hashtag/CogSci2021?src=hash&amp;ref_src=twsrc%5Etfw">#CogSci2021</a> paper looking at cultural transmission in Atari-style video games <a href="https://t.co/V6l5lftrqe">https://t.co/V6l5lftrqe</a>. Also will be at poster 3-E-174 during the poster session today.  1/6</p>&mdash; MH Tessler (@mhtessler) <a href="https://twitter.com/mhtessler/status/1420727664655089667?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Growing knowledge culturally across generations to solve novel, complex tasks<br>paper: <a href="https://t.co/7qwse8L7VM">https://t.co/7qwse8L7VM</a><br>github: <a href="https://t.co/40RpwHlFWK">https://t.co/40RpwHlFWK</a><br><br>results suggest that language provides a sufficient medium to express and accumulate the knowledge people acquire in these diverse tasks <a href="https://t.co/LmqH5vEztB">pic.twitter.com/LmqH5vEztB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420550856773181447?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Statistically Meaningful Approximation: a Case Study on Approximating  Turing Machines with Transformers

Colin Wei, Yining Chen, Tengyu Ma

- retweets: 90, favorites: 51 (07/30/2021 10:40:38)

- links: [abs](https://arxiv.org/abs/2107.13163) | [pdf](https://arxiv.org/pdf/2107.13163)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

A common lens to theoretically study neural net architectures is to analyze the functions they can approximate. However, the constructions from approximation theory often have unrealistic aspects, for example, reliance on infinite precision to memorize target function values, which make these results potentially less meaningful. To address these issues, this work proposes a formal definition of statistically meaningful approximation which requires the approximating network to exhibit good statistical learnability. We present case studies on statistically meaningful approximation for two classes of functions: boolean circuits and Turing machines. We show that overparameterized feedforward neural nets can statistically meaningfully approximate boolean circuits with sample complexity depending only polynomially on the circuit size, not the size of the approximating network. In addition, we show that transformers can statistically meaningfully approximate Turing machines with computation time bounded by $T$, requiring sample complexity polynomial in the alphabet size, state space size, and $\log (T)$. Our analysis introduces new tools for generalization bounds that provide much tighter sample complexity guarantees than the typical VC-dimension or norm-based bounds, which may be of independent interest.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">RNNs and transformers with *infinite*-precision are universal approximators---but are they statistically learnable? We show that even finite-precision transformers can express any Turing machine, and they are learnable with polynomial samples. <a href="https://t.co/cXgvJRmpke">https://t.co/cXgvJRmpke</a> <a href="https://t.co/SEy4Ctkbxs">pic.twitter.com/SEy4Ctkbxs</a></p>&mdash; Tengyu Ma (@tengyuma) <a href="https://twitter.com/tengyuma/status/1420828766507393026?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Tale Of Two Long Tails

Daniel D'souza, Zach Nussbaum, Chirag Agarwal, Sara Hooker

- retweets: 88, favorites: 39 (07/30/2021 10:40:38)

- links: [abs](https://arxiv.org/abs/2107.13098) | [pdf](https://arxiv.org/pdf/2107.13098)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

As machine learning models are increasingly employed to assist human decision-makers, it becomes critical to communicate the uncertainty associated with these model predictions. However, the majority of work on uncertainty has focused on traditional probabilistic or ranking approaches - where the model assigns low probabilities or scores to uncertain examples. While this captures what examples are challenging for the model, it does not capture the underlying source of the uncertainty. In this work, we seek to identify examples the model is uncertain about and characterize the source of said uncertainty. We explore the benefits of designing a targeted intervention - targeted data augmentation of the examples where the model is uncertain over the course of training. We investigate whether the rate of learning in the presence of additional information differs between atypical and noisy examples? Our results show that this is indeed the case, suggesting that well-designed interventions over the course of training can be an effective way to characterize and distinguish between different sources of uncertainty.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Tale Of Two Long Tails<br>pdf: <a href="https://t.co/kX3Ud70XN3">https://t.co/kX3Ud70XN3</a><br>abs: <a href="https://t.co/zqDXTlfctb">https://t.co/zqDXTlfctb</a><br><br>results suggest that targeted interventions are a powerful tool to characterize and distinguish between different<br>sources of uncertainty <a href="https://t.co/Y5gnXq4f0P">pic.twitter.com/Y5gnXq4f0P</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420553217755205634?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Dataset Distillation with Infinitely Wide Convolutional Networks

Timothy Nguyen, Roman Novak, Lechao Xiao, Jaehoon Lee

- retweets: 62, favorites: 63 (07/30/2021 10:40:38)

- links: [abs](https://arxiv.org/abs/2107.13034) | [pdf](https://arxiv.org/pdf/2107.13034)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The effectiveness of machine learning algorithms arises from being able to extract useful features from large amounts of data. As model and dataset sizes increase, dataset distillation methods that compress large datasets into significantly smaller yet highly performant ones will become valuable in terms of training efficiency and useful feature extraction. To that end, we apply a novel distributed kernel based meta-learning framework to achieve state-of-the-art results for dataset distillation using infinitely wide convolutional neural networks. For instance, using only 10 datapoints (0.02% of original dataset), we obtain over 64% test accuracy on CIFAR-10 image classification task, a dramatic improvement over the previous best test accuracy of 40%. Our state-of-the-art results extend across many other settings for MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and SVHN. Furthermore, we perform some preliminary analyses of our distilled datasets to shed light on how they differ from naturally occurring data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dataset Distillation with Infinitely Wide Convolutional Networks<br>paper: <a href="https://t.co/xBg6zBAOjd">https://t.co/xBg6zBAOjd</a><br><br>only 10 datapoints (0.02% of original dataset), obtain over 64% test accuracy on CIFAR10 image classification task, a dramatic improvement over the previous best test accuracy of 40% <a href="https://t.co/WRIVwEDctb">pic.twitter.com/WRIVwEDctb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420551749518217220?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper “Dataset Distillation with Infinitely Wide Convolutional Networks” achieves 64% CIFAR10 test acc using only 10 imgs and exceeds prior art by as much as 37% using large scale distributed meta-learning. Joint w/ <a href="https://twitter.com/ARomanNovak?ref_src=twsrc%5Etfw">@ARomanNovak</a> <a href="https://twitter.com/hoonkp?ref_src=twsrc%5Etfw">@hoonkp</a> <a href="https://twitter.com/Locchiu?ref_src=twsrc%5Etfw">@locchiu</a> <a href="https://t.co/yLOOIwwtaN">https://t.co/yLOOIwwtaN</a> <a href="https://t.co/7t238pOqZ4">pic.twitter.com/7t238pOqZ4</a></p>&mdash; Timothy Nguyen (@IAmTimNguyen) <a href="https://twitter.com/IAmTimNguyen/status/1420803422140305409?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Is Object Detection Necessary for Human-Object Interaction Recognition?

Ying Jin, Yinpeng Chen, Lijuan Wang, Jianfeng Wang, Pei Yu, Zicheng Liu, Jenq-Neng Hwang

- retweets: 64, favorites: 40 (07/30/2021 10:40:39)

- links: [abs](https://arxiv.org/abs/2107.13083) | [pdf](https://arxiv.org/pdf/2107.13083)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

This paper revisits human-object interaction (HOI) recognition at image level without using supervisions of object location and human pose. We name it detection-free HOI recognition, in contrast to the existing detection-supervised approaches which rely on object and keypoint detections to achieve state of the art. With our method, not only the detection supervision is evitable, but superior performance can be achieved by properly using image-text pre-training (such as CLIP) and the proposed Log-Sum-Exp Sign (LSE-Sign) loss function. Specifically, using text embeddings of class labels to initialize the linear classifier is essential for leveraging the CLIP pre-trained image encoder. In addition, LSE-Sign loss facilitates learning from multiple labels on an imbalanced dataset by normalizing gradients over all classes in a softmax format. Surprisingly, our detection-free solution achieves 60.5 mAP on the HICO dataset, outperforming the detection-supervised state of the art by 13.4 mAP

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is Object Detection Necessary for Human-Object Interaction Recognition?<br>pdf: <a href="https://t.co/g1UqY8QMtF">https://t.co/g1UqY8QMtF</a><br>abs: <a href="https://t.co/X2AFZaACos">https://t.co/X2AFZaACos</a><br><br>detection-free solution achieves 60.5 mAP on the HICO dataset, outperforming the detection-supervised state of the art by 13.4 mAP <a href="https://t.co/wTIIdRkskz">pic.twitter.com/wTIIdRkskz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1420544151096107015?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Supervised Learning and the Finite-Temperature String Method for  Computing Committor Functions and Reaction Rates

Muhammad R. Hasyim, Clay H. Batton, Kranthi K. Mandadapu

- retweets: 32, favorites: 24 (07/30/2021 10:40:39)

- links: [abs](https://arxiv.org/abs/2107.13522) | [pdf](https://arxiv.org/pdf/2107.13522)
- [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent)

A central object in the computational studies of rare events is the committor function. Though costly to compute, the committor function encodes complete mechanistic information of the processes involving rare events, including reaction rates and transition-state ensembles. Under the framework of transition path theory (TPT), recent work [1] proposes an algorithm where a feedback loop couples a neural network that models the committor function with importance sampling, mainly umbrella sampling, which collects data needed for adaptive training. In this work, we show additional modifications are needed to improve the accuracy of the algorithm. The first modification adds elements of supervised learning, which allows the neural network to improve its prediction by fitting to sample-mean estimates of committor values obtained from short molecular dynamics trajectories. The second modification replaces the committor-based umbrella sampling with the finite-temperature string (FTS) method, which enables homogeneous sampling in regions where transition pathways are located. We test our modifications on low-dimensional systems with non-convex potential energy where reference solutions can be found via analytical or the finite element methods, and show how combining supervised learning and the FTS method yields accurate computation of committor functions and reaction rates. We also provide an error analysis for algorithms that use the FTS method, using which reaction rates can be accurately estimated during training with a small number of samples.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Got a paper up on the arXiv using machine learning to compute the probability a reaction goes towards completion. We show how you can use supervised learning elements for accuracy, and smart sampling to get the reaction rates down well. Check it out!<a href="https://t.co/kh871cpqWy">https://t.co/kh871cpqWy</a></p>&mdash; Clay Batton (@cbatton35) <a href="https://twitter.com/cbatton35/status/1420555042424164356?ref_src=twsrc%5Etfw">July 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



