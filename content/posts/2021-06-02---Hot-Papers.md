---
title: Hot Papers 2021-06-02
date: 2021-06-03T09:56:29.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. Machine-Learning Non-Conservative Dynamics for New-Physics Detection

Ziming Liu, Bohan Wang, Qi Meng, Wei Chen, Max Tegmark, Tie-Yan Liu

- retweets: 3362, favorites: 302 (06/03/2021 09:56:29)

- links: [abs](https://arxiv.org/abs/2106.00026) | [pdf](https://arxiv.org/pdf/2106.00026)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [gr-qc](https://arxiv.org/list/gr-qc/recent) | [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent)

Energy conservation is a basic physics principle, the breakdown of which often implies new physics. This paper presents a method for data-driven "new physics" discovery. Specifically, given a trajectory governed by unknown forces, our Neural New-Physics Detector (NNPhD) aims to detect new physics by decomposing the force field into conservative and non-conservative components, which are represented by a Lagrangian Neural Network (LNN) and a universal approximator network (UAN), respectively, trained to minimize the force recovery error plus a constant $\lambda$ times the magnitude of the predicted non-conservative force. We show that a phase transition occurs at $\lambda$=1, universally for arbitrary forces. We demonstrate that NNPhD successfully discovers new physics in toy numerical experiments, rediscovering friction (1493) from a damped double pendulum, Neptune from Uranus' orbit (1846) and gravitational waves (2017) from an inspiraling orbit. We also show how NNPhD coupled with an integrator outperforms previous methods for predicting the future of a damped double pendulum.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m excited about our method for machine-learning new physics. It auto-discovers e.g. Neptune &amp; gravitational radiation by detecting energy conservation violation even when the physical laws are unknown: <a href="https://t.co/aNHIu4y3MK">https://t.co/aNHIu4y3MK</a> <a href="https://t.co/lXbR4VvUTQ">pic.twitter.com/lXbR4VvUTQ</a></p>&mdash; Max Tegmark (@tegmark) <a href="https://twitter.com/tegmark/status/1400110950129274881?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Towards Real-time and Light-weight Line Segment Detection

Geonmo Gu, Byungsoo Ko, SeoungHyun Go, Sung-Hyun Lee, Jingeun Lee, Minchul Shin

- retweets: 2448, favorites: 217 (06/03/2021 09:56:29)

- links: [abs](https://arxiv.org/abs/2106.00186) | [pdf](https://arxiv.org/pdf/2106.00186)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Previous deep learning-based line segment detection (LSD) suffer from the immense model size and high computational cost for line prediction. This constrains them from real-time inference on computationally restricted environments. In this paper, we propose a real-time and light-weight line segment detector for resource-constrained environments named Mobile LSD (M-LSD). We design an extremely efficient LSD architecture by minimizing the backbone network and removing the typical multi-module process for line prediction in previous methods. To maintain competitive performance with such a light-weight network, we present novel training schemes: Segments of Line segment (SoL) augmentation and geometric learning scheme. SoL augmentation splits a line segment into multiple subparts, which are used to provide auxiliary line data during the training process. Moreover, the geometric learning scheme allows a model to capture additional geometry cues from matching loss, junction and line segmentation, length and degree regression. Compared with TP-LSD-Lite, previously the best real-time LSD method, our model (M-LSD-tiny) achieves competitive performance with 2.5% of model size and an increase of 130.5% in inference speed on GPU when evaluated with Wireframe and YorkUrban datasets. Furthermore, our model runs at 56.8 FPS and 48.6 FPS on Android and iPhone mobile devices, respectively. To the best of our knowledge, this is the first real-time deep LSD method available on mobile devices.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Light-weight and Real-time Line Segment Detection<br>pdf: <a href="https://t.co/RKLVIFgPCg">https://t.co/RKLVIFgPCg</a><br>abs: <a href="https://t.co/r0B8OZ8SVA">https://t.co/r0B8OZ8SVA</a><br>github: <a href="https://t.co/TRbfwrL39M">https://t.co/TRbfwrL39M</a><br>colab: <a href="https://t.co/AH4ivTi8zr">https://t.co/AH4ivTi8zr</a><br><br>a real-time and light-weight line segment detector for resource-constrained environments <a href="https://t.co/icaCgNnZqN">pic.twitter.com/icaCgNnZqN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399927771674165253?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D  World

Rowan Zellers, Ari Holtzman, Matthew Peters, Roozbeh Mottaghi, Aniruddha Kembhavi, Ali Farhadi, Yejin Choi

- retweets: 884, favorites: 141 (06/03/2021 09:56:29)

- links: [abs](https://arxiv.org/abs/2106.00188) | [pdf](https://arxiv.org/pdf/2106.00188)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We propose PIGLeT: a model that learns physical commonsense knowledge through interaction, and then uses this knowledge to ground language. We factorize PIGLeT into a physical dynamics model, and a separate language model. Our dynamics model learns not just what objects are but also what they do: glass cups break when thrown, plastic ones don't. We then use it as the interface to our language model, giving us a unified model of linguistic form and grounded meaning. PIGLeT can read a sentence, simulate neurally what might happen next, and then communicate that result through a literal symbolic representation, or natural language.   Experimental results show that our model effectively learns world dynamics, along with how to communicate them. It is able to correctly forecast "what happens next" given an English sentence over 80% of the time, outperforming a 100x larger, text-to-text approach by over 10%. Likewise, its natural language summaries of physical interactions are also judged by humans as more accurate than LM alternatives. We present comprehensive analysis showing room for future work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New at ACL 2021: we introduce PIGLeT, a model that learns to link language models with physical understanding -- learned by interacting with a 3D world. This combined approach outperforms 100x bigger text-only models at physical commonsense reasoning.<a href="https://t.co/dcfsPNf9Mp">https://t.co/dcfsPNf9Mp</a> <a href="https://t.co/ATVA02hn7A">pic.twitter.com/ATVA02hn7A</a></p>&mdash; Rowan Zellers (@rown) <a href="https://twitter.com/rown/status/1400168437205278720?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World<br>pdf: <a href="https://t.co/tCY8yVNDQ3">https://t.co/tCY8yVNDQ3</a><br>abs: <a href="https://t.co/SfXQ1DenCS">https://t.co/SfXQ1DenCS</a><br><br>forecasts “what happens next” given an English sentence over 80% of the time, outperforming a 100x larger, text-to-text approach by over 10% <a href="https://t.co/Fmv4AGgbWF">pic.twitter.com/Fmv4AGgbWF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399902994745987076?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. You Only Look at One Sequence: Rethinking Transformer in Vision through  Object Detection

Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu

- retweets: 870, favorites: 141 (06/03/2021 09:56:30)

- links: [abs](https://arxiv.org/abs/2106.00666) | [pdf](https://arxiv.org/pdf/2106.00666)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Can Transformer perform $2\mathrm{D}$ object-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the $2\mathrm{D}$ spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the na\"ive Vision Transformer with the fewest possible modifications as well as inductive biases. We find that YOLOS pre-trained on the mid-sized ImageNet-$1k$ dataset only can already achieve competitive object detection performance on COCO, \textit{e.g.}, YOLOS-Base directly adopted from BERT-Base can achieve $42.0$ box AP. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through object detection. Code and model weights are available at \url{https://github.com/hustvl/YOLOS}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection<br>pdf: <a href="https://t.co/LThmFQ2a6g">https://t.co/LThmFQ2a6g</a><br>abs: <a href="https://t.co/XhhOip5bOw">https://t.co/XhhOip5bOw</a><br>github: <a href="https://t.co/crirLeiGGI">https://t.co/crirLeiGGI</a><br><br>series of object detection models based on the naïve Vision Transformer <a href="https://t.co/Ml38kzqdtt">pic.twitter.com/Ml38kzqdtt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399898816053121024?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Language-Driven Image Style Transfer

Tsu-Jui Fu, Xin Eric Wang, William Yang Wang

- retweets: 650, favorites: 117 (06/03/2021 09:56:30)

- links: [abs](https://arxiv.org/abs/2106.00178) | [pdf](https://arxiv.org/pdf/2106.00178)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Despite having promising results, style transfer, which requires preparing style images in advance, may result in lack of creativity and accessibility. Following human instruction, on the other hand, is the most natural way to perform artistic style transfer that can significantly improve controllability for visual effect applications. We introduce a new task -- language-driven image style transfer (\texttt{LDIST}) -- to manipulate the style of a content image, guided by a text. We propose contrastive language visual artist (CLVA) that learns to extract visual semantics from style instructions and accomplish \texttt{LDIST} by the patch-wise style discriminator. The discriminator considers the correlation between language and patches of style images or transferred results to jointly embed style instructions. CLVA further compares contrastive pairs of content image and style instruction to improve the mutual relativeness between transfer results. The transferred results from the same content image can preserve consistent content structures. Besides, they should present analogous style patterns from style instructions that contain similar visual semantics. The experiments show that our CLVA is effective and achieves superb transferred results on \texttt{LDIST}.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Language-Driven Image Style Transfer<br>pdf: <a href="https://t.co/wwkXSRUy31">https://t.co/wwkXSRUy31</a><br>abs: <a href="https://t.co/8eb8JvAcRL">https://t.co/8eb8JvAcRL</a><br>project page: <a href="https://t.co/ipn7TpHREb">https://t.co/ipn7TpHREb</a> <a href="https://t.co/HrRvPKMVqe">pic.twitter.com/HrRvPKMVqe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399922564378644481?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Omnizart: A General Toolbox for Automatic Music Transcription

Yu-Te Wu, Yin-Jyun Luo, Tsung-Ping Chen, I-Chieh Wei, Jui-Yang Hsu, Yi-Chin Chuang, Li Su

- retweets: 462, favorites: 65 (06/03/2021 09:56:30)

- links: [abs](https://arxiv.org/abs/2106.00497) | [pdf](https://arxiv.org/pdf/2106.00497)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We present and release Omnizart, a new Python library that provides a streamlined solution to automatic music transcription (AMT). Omnizart encompasses modules that construct the life-cycle of deep learning-based AMT, and is designed for ease of use with a compact command-line interface. To the best of our knowledge, Omnizart is the first transcription toolkit which offers models covering a wide class of instruments ranging from solo, instrument ensembles, percussion instruments to vocal, as well as models for chord recognition and beat/downbeat tracking, two music information retrieval (MIR) tasks highly related to AMT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Omnizart: A General Toolbox for Automatic Music Transcription 🎶<br>pdf: <a href="https://t.co/ZrOOOFbs81">https://t.co/ZrOOOFbs81</a><br>abs: <a href="https://t.co/nuSzRpHdnu">https://t.co/nuSzRpHdnu</a><br>github: <a href="https://t.co/8j25s9aJU7">https://t.co/8j25s9aJU7</a><br><br>library that provides a streamlined solution to automatic music transcription <a href="https://t.co/kIb8Kmfnft">pic.twitter.com/kIb8Kmfnft</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399907321862660104?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Bootstrap Your Own Correspondences

Mohamed El Banani, Justin Johnson

- retweets: 380, favorites: 64 (06/03/2021 09:56:30)

- links: [abs](https://arxiv.org/abs/2106.00677) | [pdf](https://arxiv.org/pdf/2106.00677)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Geometric feature extraction is a crucial component of point cloud registration pipelines. Recent work has demonstrated how supervised learning can be leveraged to learn better and more compact 3D features. However, those approaches' reliance on ground-truth annotation limits their scalability. We propose BYOC: a self-supervised approach that learns visual and geometric features from RGB-D video without relying on ground-truth pose or correspondence. Our key observation is that randomly-initialized CNNs readily provide us with good correspondences; allowing us to bootstrap the learning of both visual and geometric features. Our approach combines classic ideas from point cloud registration with more recent representation learning approaches. We evaluate our approach on indoor scene datasets and find that our method outperforms traditional and learned descriptors, while being competitive with current state-of-the-art supervised approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Bootstrap Your Own Correspondences<br>pdf: <a href="https://t.co/JmtTgZVy6i">https://t.co/JmtTgZVy6i</a><br>abs: <a href="https://t.co/TXGBFspXSC">https://t.co/TXGBFspXSC</a><br><br>randomly-initialized CNNs readily provide us with good correspondences; allowing us to bootstrap the learning of both visual and geometric features <a href="https://t.co/ZJLywSl5Gz">pic.twitter.com/ZJLywSl5Gz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399895566830686212?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. KVT: k-NN Attention for Boosting Vision Transformers

Pichao Wang, Xue Wang, Fan Wang, Ming Lin, Shuning Chang, Wen Xie, Hao Li, Rong Jin

- retweets: 342, favorites: 51 (06/03/2021 09:56:30)

- links: [abs](https://arxiv.org/abs/2106.00515) | [pdf](https://arxiv.org/pdf/2106.00515)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Convolutional Neural Networks (CNNs) have dominated computer vision for years, due to its ability in capturing locality and translation invariance. Recently, many vision transformer architectures have been proposed and they show promising performance. A key component in vision transformers is the fully-connected self-attention which is more powerful than CNNs in modelling long range dependencies. However, since the current dense self-attention uses all image patches (tokens) to compute attention matrix, it may neglect locality of images patches and involve noisy tokens (e.g., clutter background and occlusion), leading to a slow training process and potentially degradation of performance. To address these problems, we propose a sparse attention scheme, dubbed k-NN attention, for boosting vision transformers. Specifically, instead of involving all the tokens for attention matrix calculation, we only select the top-k similar tokens from the keys for each query to compute the attention map. The proposed k-NN attention naturally inherits the local bias of CNNs without introducing convolutional operations, as nearby tokens tend to be more similar than others. In addition, the k-NN attention allows for the exploration of long range correlation and at the same time filter out irrelevant tokens by choosing the most similar tokens from the entire image. Despite its simplicity, we verify, both theoretically and empirically, that $k$-NN attention is powerful in distilling noise from input tokens and in speeding up training. Extensive experiments are conducted by using ten different vision transformer architectures to verify that the proposed k-NN attention can work with any existing transformer architectures to improve its prediction performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">KVT: k-NN Attention for Boosting Vision Transformers<br>pdf: <a href="https://t.co/x3YTRopWKD">https://t.co/x3YTRopWKD</a><br>abs: <a href="https://t.co/Gn2DD2SPOi">https://t.co/Gn2DD2SPOi</a><br><br>selecting the most similar keys for each query to calculate the attention, it screens out the most ineffective tokens <a href="https://t.co/Qr76skHa10">pic.twitter.com/Qr76skHa10</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399892645607100416?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Reward is enough for convex MDPs

Tom Zahavy, Brendan O'Donoghue, Guillaume Desjardins, Satinder Singh

- retweets: 240, favorites: 60 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00661) | [pdf](https://arxiv.org/pdf/2106.00661)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Maximising a cumulative reward function that is Markov and stationary, i.e., defined over state-action pairs and independent of time, is sufficient to capture many kinds of goals in a Markov Decision Process (MDP) based on the Reinforcement Learning (RL) problem formulation. However, not all goals can be captured in this manner. Specifically, it is easy to see that Convex MDPs in which goals are expressed as convex functions of stationary distributions cannot, in general, be formulated in this manner. In this paper, we reformulate the convex MDP problem as a min-max game between the policy and cost (negative reward) players using Fenchel duality and propose a meta-algorithm for solving it. We show that the average of the policies produced by an RL agent that maximizes the non-stationary reward produced by the cost player converges to an optimal solution to the convex MDP. Finally, we show that the meta-algorithm unifies several disparate branches of reinforcement learning algorithms in the literature, such as apprenticeship learning, variational intrinsic control, constrained MDPs, and pure exploration into a single framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reward is enough for convex MDPs<br>pdf: <a href="https://t.co/HwvFnqnd1t">https://t.co/HwvFnqnd1t</a><br>abs: <a href="https://t.co/IEim6jLJyr">https://t.co/IEim6jLJyr</a> <a href="https://t.co/hPpCD4aJmI">pic.twitter.com/hPpCD4aJmI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399972705722580992?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Language Model Evaluation Beyond Perplexity

Clara Meister, Ryan Cotterell

- retweets: 132, favorites: 56 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00085) | [pdf](https://arxiv.org/pdf/2106.00085)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We propose an alternate approach to quantifying how well language models learn natural language: we ask how well they match the statistical tendencies of natural language. To answer this question, we analyze whether text generated from language models exhibits the statistical tendencies present in the human-generated text on which they were trained. We provide a framework--paired with significance tests--for evaluating the fit of language models to these trends. We find that neural language models appear to learn only a subset of the tendencies considered, but align much more closely with empirical trends than proposed theoretical distributions (when present). Further, the fit to different distributions is highly-dependent on both model architecture and generation strategy. As concrete examples, text generated under the nucleus sampling scheme adheres more closely to the type--token relationship of natural language than text produced using standard ancestral sampling; text from LSTMs reflects the natural language distributions over length, stopwords, and symbols surprisingly well.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Language Model Evaluation Beyond Perplexity<br>pdf: <a href="https://t.co/dDeRH28Qpc">https://t.co/dDeRH28Qpc</a><br>abs: <a href="https://t.co/acrAkKN0vo">https://t.co/acrAkKN0vo</a> <a href="https://t.co/VFAJ48ivzS">pic.twitter.com/VFAJ48ivzS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399901011125358597?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Gaussian Processes with Differential Privacy

Antti Honkela

- retweets: 90, favorites: 54 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00474) | [pdf](https://arxiv.org/pdf/2106.00474)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Gaussian processes (GPs) are non-parametric Bayesian models that are widely used for diverse prediction tasks. Previous work in adding strong privacy protection to GPs via differential privacy (DP) has been limited to protecting only the privacy of the prediction targets (model outputs) but not inputs. We break this limitation by introducing GPs with DP protection for both model inputs and outputs. We achieve this by using sparse GP methodology and publishing a private variational approximation on known inducing points. The approximation covariance is adjusted to approximately account for the added uncertainty from DP noise. The approximation can be used to compute arbitrary predictions using standard sparse GP techniques. We propose a method for hyperparameter learning using a private selection protocol applied to validation set log-likelihood. Our experiments demonstrate that given sufficient amount of data, the method can produce accurate models under strong privacy protection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New pre-print: Gaussian Processes with Differential Privacy. DP for full input and output, approximate adjustment of posterior covariance for added DP noise. <a href="https://t.co/CUm2tdwS0a">https://t.co/CUm2tdwS0a</a> <a href="https://t.co/lGiBV6eb1A">pic.twitter.com/lGiBV6eb1A</a></p>&mdash; Antti Honkela (@ahonkela) <a href="https://twitter.com/ahonkela/status/1400002903855841283?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Multi-modal Point-of-Care Diagnostics for COVID-19 Based On Acoustics  and Symptoms

Srikanth Raj Chetupalli, Prashant Krishnan, Neeraj Sharma, Ananya Muguli, Rohit Kumar, Viral Nanda, Lancelot Mark Pinto, Prasanta Kumar Ghosh, Sriram Ganapathy

- retweets: 73, favorites: 34 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00639) | [pdf](https://arxiv.org/pdf/2106.00639)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.SP](https://arxiv.org/list/eess.SP/recent)

The research direction of identifying acoustic bio-markers of respiratory diseases has received renewed interest following the onset of COVID-19 pandemic. In this paper, we design an approach to COVID-19 diagnostic using crowd-sourced multi-modal data. The data resource, consisting of acoustic signals like cough, breathing, and speech signals, along with the data of symptoms, are recorded using a web-application over a period of ten months. We investigate the use of statistical descriptors of simple time-frequency features for acoustic signals and binary features for the presence of symptoms. Unlike previous works, we primarily focus on the application of simple linear classifiers like logistic regression and support vector machines for acoustic data while decision tree models are employed on the symptoms data. We show that a multi-modal integration of acoustics and symptoms classifiers achieves an area-under-curve (AUC) of 92.40, a significant improvement over any individual modality. Several ablation experiments are also provided which highlight the acoustic and symptom dimensions that are important for the task of COVID-19 diagnostics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we detect Covid using acoustics and symptoms ?<br>Our findings on Coswara dataset (under review)<a href="https://t.co/C2ANA33qU2">https://t.co/C2ANA33qU2</a><br>Summary - 92.7% acc., spec. of 95%, sensitivity of 69%.    <br>We deeply appreciate if you contribute(d) data to our study <a href="https://t.co/ukVVjafEyT">https://t.co/ukVVjafEyT</a><a href="https://twitter.com/coswara_iisc?ref_src=twsrc%5Etfw">@coswara_iisc</a> <a href="https://t.co/3gLdXUQrSK">pic.twitter.com/3gLdXUQrSK</a></p>&mdash; Sriram Ganapathy (@tweet4sri) <a href="https://twitter.com/tweet4sri/status/1399953534456532993?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. What Matters for Adversarial Imitation Learning?

Manu Orsini, Anton Raichuk, Léonard Hussenot, Damien Vincent, Robert Dadashi, Sertan Girgin, Matthieu Geist, Olivier Bachem, Olivier Pietquin, Marcin Andrychowicz

- retweets: 58, favorites: 32 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00672) | [pdf](https://arxiv.org/pdf/2106.00672)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Adversarial imitation learning has become a popular framework for imitation in continuous control. Over the years, several variations of its components were proposed to enhance the performance of the learned policies as well as the sample complexity of the algorithm. In practice, these choices are rarely tested all together in rigorous empirical studies. It is therefore difficult to discuss and understand what choices, among the high-level algorithmic options as well as low-level implementation details, matter. To tackle this issue, we implement more than 50 of these choices in a generic adversarial imitation learning framework and investigate their impacts in a large-scale study (>500k trained agents) with both synthetic and human-generated demonstrations. While many of our findings confirm common practices, some of them are surprising or even contradict prior work. In particular, our results suggest that artificial demonstrations are not a good proxy for human data and that the very common practice of evaluating imitation algorithms only with synthetic demonstrations may lead to algorithms which perform poorly in the more realistic scenarios with human demonstrations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What Matters for Adversarial Imitation Learning?<br>pdf: <a href="https://t.co/5sKmrIwUC1">https://t.co/5sKmrIwUC1</a><br>abs: <a href="https://t.co/vfDyAB4SMh">https://t.co/vfDyAB4SMh</a><br><br>large-scale study (&gt;500k trained agents) with both synthetic and human-generated demonstrations <a href="https://t.co/tFhLKmk5DV">pic.twitter.com/tFhLKmk5DV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399890263162634240?ref_src=twsrc%5Etfw">June 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Privacy and Confidentiality in Process Mining -- Threats and Research  Challenges

Gamal Elkoumy, Stephan A. Fahrenkrog-Petersen, Mohammadreza Fani Sani, Agnes Koschmider, Felix Mannhardt, Saskia Nuñez von Voigt, Majid Rafiei, Leopold von Waldthausen

- retweets: 42, favorites: 16 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00388) | [pdf](https://arxiv.org/pdf/2106.00388)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent)

Privacy and confidentiality are very important prerequisites for applying process mining in order to comply with regulations and keep company secrets. This paper provides a foundation for future research on privacy-preserving and confidential process mining techniques. Main threats are identified and related to an motivation application scenario in a hospital context as well as to the current body of work on privacy and confidentiality in process mining. A newly developed conceptual model structures the discussion that existing techniques leave room for improvement. This results in a number of important research challenges that should be addressed by future process mining research.




# 15. Construction of Simplicial Complexes with Prescribed Degree-Size  Sequences

Tzu-Chi Yen

- retweets: 42, favorites: 8 (06/03/2021 09:56:31)

- links: [abs](https://arxiv.org/abs/2106.00185) | [pdf](https://arxiv.org/pdf/2106.00185)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent) | [math.AT](https://arxiv.org/list/math.AT/recent) | [math.CO](https://arxiv.org/list/math.CO/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

We study the realizability of simplicial complexes with a given pair of integer sequences, representing the node degree distribution and facet size distribution, respectively. While the $s$-uniform variant of the problem is $\mathsf{NP}$-complete when $s \geq 3$, we identify two populations of input sequences, most of which can be solved in polynomial time using a recursive algorithm that we contribute. Combining with a sampler for the simplicial configuration model [Young $\textit{et al.}$, Phys. Rev. E $\textbf{96}$, 032312 (2017)], we facilitate efficient sampling of simplicial ensembles from arbitrary degree and size distributions. We find that, contrary to expectations based on dyadic networks, increasing nodes' degrees reduces the number of loops in simplicial complexes. Our work unveils a fundamental constraint on the degree-size sequences and sheds light on further analysis of higher-order phenomena based on local structures.



