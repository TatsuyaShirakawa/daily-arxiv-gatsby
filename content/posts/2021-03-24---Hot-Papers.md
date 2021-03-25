---
title: Hot Papers 2021-03-24
date: 2021-03-25T09:03:24.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. iMAP: Implicit Mapping and Positioning in Real-Time

Edgar Sucar, Shikun Liu, Joseph Ortiz, Andrew J. Davison

- retweets: 6090, favorites: 370 (03/25/2021 09:03:24)

- links: [abs](https://arxiv.org/abs/2103.12352) | [pdf](https://arxiv.org/pdf/2103.12352)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We show for the first time that a multilayer perceptron (MLP) can serve as the only scene representation in a real-time SLAM system for a handheld RGB-D camera. Our network is trained in live operation without prior data, building a dense, scene-specific implicit 3D model of occupancy and colour which is also immediately used for tracking.   Achieving real-time SLAM via continual training of a neural network against a live image stream requires significant innovation. Our iMAP algorithm uses a keyframe structure and multi-processing computation flow, with dynamic information-guided pixel sampling for speed, with tracking at 10 Hz and global map updating at 2 Hz. The advantages of an implicit MLP over standard dense SLAM techniques include efficient geometry representation with automatic detail control and smooth, plausible filling-in of unobserved regions such as the back surfaces of objects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share iMAP, first real-time SLAM system to use an implicit scene network as map representation.<br><br>Work with: <a href="https://twitter.com/liu_shikun?ref_src=twsrc%5Etfw">@liu_shikun</a>, <a href="https://twitter.com/joeaortiz?ref_src=twsrc%5Etfw">@joeaortiz</a>, <a href="https://twitter.com/AjdDavison?ref_src=twsrc%5Etfw">@AjdDavison</a> <br><br>Project page: <a href="https://t.co/Tagk4jFN2M">https://t.co/Tagk4jFN2M</a><br>Paper: <a href="https://t.co/OQA1QdLY4Q">https://t.co/OQA1QdLY4Q</a> <a href="https://t.co/KG0cY68MXn">pic.twitter.com/KG0cY68MXn</a></p>&mdash; Edgar Sucar (@SucarEdgar) <a href="https://twitter.com/SucarEdgar/status/1374625443211440130?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Generative Minimization Networks: Training GANs Without Competition

Paulina Grnarova, Yannic Kilcher, Kfir Y. Levy, Aurelien Lucchi, Thomas Hofmann

- retweets: 2496, favorites: 287 (03/25/2021 09:03:24)

- links: [abs](https://arxiv.org/abs/2103.12685) | [pdf](https://arxiv.org/pdf/2103.12685)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Many applications in machine learning can be framed as minimization problems and solved efficiently using gradient-based techniques. However, recent applications of generative models, particularly GANs, have triggered interest in solving min-max games for which standard optimization techniques are often not suitable. Among known problems experienced by practitioners is the lack of convergence guarantees or convergence to a non-optimum cycle. At the heart of these problems is the min-max structure of the GAN objective which creates non-trivial dependencies between the players. We propose to address this problem by optimizing a different objective that circumvents the min-max structure using the notion of duality gap from game theory. We provide novel convergence guarantees on this objective and demonstrate why the obtained limit point solves the problem better than known techniques.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generative Minimization Networks: Training GANs Without Competition<br>pdf: <a href="https://t.co/xreEHRgww1">https://t.co/xreEHRgww1</a><br>abs: <a href="https://t.co/ZbJLAbh5j8">https://t.co/ZbJLAbh5j8</a> <a href="https://t.co/WEc4YgQ4WN">pic.twitter.com/WEc4YgQ4WN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374529277740380162?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Scaling Local Self-Attention For Parameter Efficient Visual Backbones

Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, Jonathon Shlens

- retweets: 688, favorites: 127 (03/25/2021 09:03:25)

- links: [abs](https://arxiv.org/abs/2103.12731) | [pdf](https://arxiv.org/pdf/2103.12731)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-attention has the promise of improving computer vision systems due to parameter-independent scaling of receptive fields and content-dependent interactions, in contrast to parameter-dependent scaling and content-independent interactions of convolutions. Self-attention models have recently been shown to have encouraging improvements on accuracy-parameter trade-offs compared to baseline convolutional models such as ResNet-50. In this work, we aim to develop self-attention models that can outperform not just the canonical baseline models, but even the high-performing convolutional models. We propose two extensions to self-attention that, in conjunction with a more efficient implementation of self-attention, improve the speed, memory usage, and accuracy of these models. We leverage these improvements to develop a new self-attention model family, \emph{HaloNets}, which reach state-of-the-art accuracies on the parameter-limited setting of the ImageNet classification benchmark. In preliminary transfer learning experiments, we find that HaloNet models outperform much larger models and have better inference performance. On harder tasks such as object detection and instance segmentation, our simple local self-attention and convolutional hybrids show improvements over very strong baselines. These results mark another step in demonstrating the efficacy of self-attention models on settings traditionally dominated by convolutional models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Local Self-Attention For Parameter Efficient Visual Backbones<br><br>Self-attention-based HaloNets achieves SotA parameter-accuracy on ImageNet and perform well on object detection.<a href="https://t.co/C4wfZJcu1F">https://t.co/C4wfZJcu1F</a> <a href="https://t.co/vmCeJKxTLt">pic.twitter.com/vmCeJKxTLt</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1374525155448295427?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Self-Supervised Pretraining Improves Self-Supervised Pretraining

Colorado J. Reed, Xiangyu Yue, Ani Nrusimha, Sayna Ebrahimi, Vivek Vijaykumar, Richard Mao, Bo Li, Shanghang Zhang, Devin Guillory, Sean Metzger, Kurt Keutzer, Trevor Darrell

- retweets: 433, favorites: 172 (03/25/2021 09:03:25)

- links: [abs](https://arxiv.org/abs/2103.12718) | [pdf](https://arxiv.org/pdf/2103.12718)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

While self-supervised pretraining has proven beneficial for many computer vision tasks, it requires expensive and lengthy computation, large amounts of data, and is sensitive to data augmentation. Prior work demonstrates that models pretrained on datasets dissimilar to their target data, such as chest X-ray models trained on ImageNet, underperform models trained from scratch. Users that lack the resources to pretrain must use existing models with lower performance. This paper explores Hierarchical PreTraining (HPT), which decreases convergence time and improves accuracy by initializing the pretraining process with an existing pretrained model. Through experimentation on 16 diverse vision datasets, we show HPT converges up to 80x faster, improves accuracy across tasks, and improves the robustness of the self-supervised pretraining process to changes in the image augmentation policy or amount of pretraining data. Taken together, HPT provides a simple framework for obtaining better pretrained representations with less computational resources.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Pretraining Improves Self-Supervised Pretraining<br><br>Proposes HPT, which accelerates convergence and improves accuracy by initializing the pretraining process with an existing pretrained model.<br><br>abs: <a href="https://t.co/A1LQMLJWeB">https://t.co/A1LQMLJWeB</a><br>code: <a href="https://t.co/7K6UK1xRY4">https://t.co/7K6UK1xRY4</a> <a href="https://t.co/KRceZus2QY">pic.twitter.com/KRceZus2QY</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1374528421787693058?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Pretraining Improves Self-Supervised Pretraining<br>pdf: <a href="https://t.co/fL2mxN1vzX">https://t.co/fL2mxN1vzX</a><br>abs: <a href="https://t.co/fPPTidRWaE">https://t.co/fPPTidRWaE</a><br>github: <a href="https://t.co/guywFouV07">https://t.co/guywFouV07</a> <a href="https://t.co/NGJv9nyOid">pic.twitter.com/NGJv9nyOid</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374527520784125952?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Transformers Solve the Limited Receptive Field for Monocular Depth  Prediction

Guanglei Yang, Hao Tang, Mingli Ding, Nicu Sebe, Elisa Ricci

- retweets: 304, favorites: 103 (03/25/2021 09:03:25)

- links: [abs](https://arxiv.org/abs/2103.12091) | [pdf](https://arxiv.org/pdf/2103.12091)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

While convolutional neural networks have shown a tremendous impact on various computer vision tasks, they generally demonstrate limitations in explicitly modeling long-range dependencies due to the intrinsic locality of the convolution operation. Transformers, initially designed for natural language processing tasks, have emerged as alternative architectures with innate global self-attention mechanisms to capture long-range dependencies. In this paper, we propose TransDepth, an architecture which benefits from both convolutional neural networks and transformers. To avoid the network to loose its ability to capture local-level details due to the adoption of transformers, we propose a novel decoder which employs on attention mechanisms based on gates. Notably, this is the first paper which applies transformers into pixel-wise prediction problems involving continuous labels (i.e., monocular depth prediction and surface normal estimation). Extensive experiments demonstrate that the proposed TransDepth achieves state-of-the-art performance on three challenging datasets. The source code and trained models are available at https://github.com/ygjwd12345/TransDepth.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Transformers Solve the Limited Receptive Field for Monocular Depth Prediction<br>pdf: <a href="https://t.co/hEhkLBLBCB">https://t.co/hEhkLBLBCB</a><br>abs: <a href="https://t.co/egR5yL6Ehu">https://t.co/egR5yL6Ehu</a><br>github: <a href="https://t.co/lSvOVOKUns">https://t.co/lSvOVOKUns</a> <a href="https://t.co/0JE7YKfkf1">pic.twitter.com/0JE7YKfkf1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374524793555361799?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. End-to-End Trainable Multi-Instance Pose Estimation with Transformers

Lucas Stoffl, Maxime Vidal, Alexander Mathis

- retweets: 136, favorites: 130 (03/25/2021 09:03:25)

- links: [abs](https://arxiv.org/abs/2103.12115) | [pdf](https://arxiv.org/pdf/2103.12115)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a new end-to-end trainable approach for multi-instance pose estimation by combining a convolutional neural network with a transformer. We cast multi-instance pose estimation from images as a direct set prediction problem. Inspired by recent work on end-to-end trainable object detection with transformers, we use a transformer encoder-decoder architecture together with a bipartite matching scheme to directly regress the pose of all individuals in a given image. Our model, called POse Estimation Transformer (POET), is trained using a novel set-based global loss that consists of a keypoint loss, a keypoint visibility loss, a center loss and a class loss. POET reasons about the relations between detected humans and the full image context to directly predict the poses in parallel. We show that POET can achieve high accuracy on the challenging COCO keypoint detection task. To the best of our knowledge, this model is the first end-to-end trainable multi-instance human pose estimation method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ℙ𝕆se 𝔼stimation 𝕋ransformer (ℙ𝕆𝔼𝕋): End-to-End Trainable Multi-Instance Pose Estimation with Transformers <br><br>🔥1st end-to-end trainable multi-human pose estimation method<br><br>👏 Super proud of <a href="https://twitter.com/TrackingPlumes?ref_src=twsrc%5Etfw">@TrackingPlumes</a> + <a href="https://twitter.com/LStoffl?ref_src=twsrc%5Etfw">@LStoffl</a> + <a href="https://twitter.com/vmaxmc2?ref_src=twsrc%5Etfw">@vmaxmc2</a>! cc <a href="https://twitter.com/amathislab?ref_src=twsrc%5Etfw">@amathislab</a> <a href="https://t.co/TfILlzrymb">https://t.co/TfILlzrymb</a> <a href="https://t.co/jBGhQK92ia">pic.twitter.com/jBGhQK92ia</a></p>&mdash; Dr. Mackenzie Mathis (@TrackingActions) <a href="https://twitter.com/TrackingActions/status/1374679260846247936?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We developed an end-to-end trainable multi-instance pose estimation model with transformers -<a href="https://t.co/2HZix5mMzc">https://t.co/2HZix5mMzc</a> - great work by PhD student <a href="https://twitter.com/LStoffl?ref_src=twsrc%5Etfw">@LStoffl</a> and Master&#39;s student <a href="https://twitter.com/vmaxmc2?ref_src=twsrc%5Etfw">@vmaxmc2</a>!!! <a href="https://t.co/EI99usQkow">pic.twitter.com/EI99usQkow</a></p>&mdash; A. Mathis Lab (@amathislab) <a href="https://twitter.com/amathislab/status/1374668485587046400?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Spatial Intention Maps for Multi-Agent Mobile Manipulation

Jimmy Wu, Xingyuan Sun, Andy Zeng, Shuran Song, Szymon Rusinkiewicz, Thomas Funkhouser

- retweets: 196, favorites: 69 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12710) | [pdf](https://arxiv.org/pdf/2103.12710)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

The ability to communicate intention enables decentralized multi-agent robots to collaborate while performing physical tasks. In this work, we present spatial intention maps, a new intention representation for multi-agent vision-based deep reinforcement learning that improves coordination between decentralized mobile manipulators. In this representation, each agent's intention is provided to other agents, and rendered into an overhead 2D map aligned with visual observations. This synergizes with the recently proposed spatial action maps framework, in which state and action representations are spatially aligned, providing inductive biases that encourage emergent cooperative behaviors requiring spatial coordination, such as passing objects to each other or avoiding collisions. Experiments across a variety of multi-agent environments, including heterogeneous robot teams with different abilities (lifting, pushing, or throwing), show that incorporating spatial intention maps improves performance for different mobile manipulation tasks while significantly enhancing cooperative behaviors.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Spatial Intention Maps for Multi-Agent Mobile Manipulation<br>pdf: <a href="https://t.co/urckpHHIyr">https://t.co/urckpHHIyr</a><br>abs: <a href="https://t.co/GdYymtBekO">https://t.co/GdYymtBekO</a><br>project page: <a href="https://t.co/laYcqOHAmr">https://t.co/laYcqOHAmr</a><br>github: <a href="https://t.co/1Kc9ag5mJm">https://t.co/1Kc9ag5mJm</a> <a href="https://t.co/YvG82bzG8o">pic.twitter.com/YvG82bzG8o</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374573132489621505?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Detecting Hate Speech with GPT-3

Ke-Li Chiu, Rohan Alexander

- retweets: 100, favorites: 29 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12407) | [pdf](https://arxiv.org/pdf/2103.12407)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Sophisticated language models such as OpenAI's GPT-3 can generate hateful text that targets marginalized groups. Given this capacity, we are interested in whether large language models can be used to identify hate speech and classify text as sexist or racist? We use GPT-3 to identify sexist and racist text passages with zero-, one-, and few-shot learning. We find that with zero- and one-shot learning, GPT-3 is able to identify sexist or racist text with an accuracy between 48 per cent and 69 per cent. With few-shot learning and an instruction included in the prompt, the model's accuracy can be as high as 78 per cent. We conclude that large language models have a role to play in hate speech detection, and that with further development language models could be used to counter hate speech and even self-police.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&#39;Detecting Hate Speech with GPT-3&#39; co-authored with <a href="https://twitter.com/UofTInfoFaculty?ref_src=twsrc%5Etfw">@UofTInfoFaculty</a> student Ke-Li Chiu is now available on arXiv and we&#39;d love any feedback that you have:<a href="https://t.co/40NEnOBfqM">https://t.co/40NEnOBfqM</a><br>Thank you to <a href="https://twitter.com/ghadfield?ref_src=twsrc%5Etfw">@ghadfield</a> and <a href="https://twitter.com/TorontoSRI?ref_src=twsrc%5Etfw">@TorontoSRI</a> for enabling this paper. <a href="https://t.co/y315KZMpxF">pic.twitter.com/y315KZMpxF</a></p>&mdash; Rohan Alexander (@RohanAlexander) <a href="https://twitter.com/RohanAlexander/status/1374523585818030083?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Instance-level Image Retrieval using Reranking Transformers

Fuwen Tan, Jiangbo Yuan, Vicente Ordonez

- retweets: 48, favorites: 54 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12236) | [pdf](https://arxiv.org/pdf/2103.12236)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Instance-level image retrieval is the task of searching in a large database for images that match an object in a query image. To address this task, systems usually rely on a retrieval step that uses global image descriptors, and a subsequent step that performs domain-specific refinements or reranking by leveraging operations such as geometric verification based on local features. In this work, we propose Reranking Transformers (RRTs) as a general model to incorporate both local and global features to rerank the matching images in a supervised fashion and thus replace the relatively expensive process of geometric verification. RRTs are lightweight and can be easily parallelized so that reranking a set of top matching results can be performed in a single forward-pass. We perform extensive experiments on the Revisited Oxford and Paris datasets, and the Google Landmark v2 dataset, showing that RRTs outperform previous reranking approaches while using much fewer local descriptors. Moreover, we demonstrate that, unlike existing approaches, RRTs can be optimized jointly with the feature extractor, which can lead to feature representations tailored to downstream tasks and further accuracy improvements. Training code and pretrained models will be made public.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Instance-level Image Retrieval using Reranking Transformers<a href="https://t.co/g6l4FRy9wV">https://t.co/g6l4FRy9wV</a> <a href="https://t.co/VCrIiBDqsp">pic.twitter.com/VCrIiBDqsp</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1374540336911896577?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Multilingual Autoregressive Entity Linking

Nicola De Cao, Ledell Wu, Kashyap Popat, Mikel Artetxe, Naman Goyal, Mikhail Plekhanov, Luke Zettlemoyer, Nicola Cancedda, Sebastian Riedel, Fabio Petroni

- retweets: 56, favorites: 20 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12528) | [pdf](https://arxiv.org/pdf/2103.12528)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We present mGENRE, a sequence-to-sequence system for the Multilingual Entity Linking (MEL) problem -- the task of resolving language-specific mentions to a multilingual Knowledge Base (KB). For a mention in a given language, mGENRE predicts the name of the target entity left-to-right, token-by-token in an autoregressive fashion. The autoregressive formulation allows us to effectively cross-encode mention string and entity names to capture more interactions than the standard dot product between mention and entity vectors. It also enables fast search within a large KB even for mentions that do not appear in mention tables and with no need for large-scale vector indices. While prior MEL works use a single representation for each entity, we match against entity names of as many languages as possible, which allows exploiting language connections between source input and target name. Moreover, in a zero-shot setting on languages with no training data at all, mGENRE treats the target language as a latent variable that is marginalized at prediction time. This leads to over 50% improvements in average accuracy. We show the efficacy of our approach through extensive evaluation including experiments on three popular MEL benchmarks where mGENRE establishes new state-of-the-art results. Code and pre-trained models at https://github.com/facebookresearch/GENRE.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multilingual Autoregressive Entity Linking<br>pdf: <a href="https://t.co/9aPikbeQTZ">https://t.co/9aPikbeQTZ</a><br>abs: <a href="https://t.co/8ge2vz3knd">https://t.co/8ge2vz3knd</a><br>github: <a href="https://t.co/kANlsxEj7X">https://t.co/kANlsxEj7X</a> <a href="https://t.co/848IJLuFxc">pic.twitter.com/848IJLuFxc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374526286178226176?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Leveraging background augmentations to encourage semantic focus in  self-supervised contrastive learning

Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

- retweets: 42, favorites: 28 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12719) | [pdf](https://arxiv.org/pdf/2103.12719)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Unsupervised representation learning is an important challenge in computer vision, with self-supervised learning methods recently closing the gap to supervised representation learning. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, disregarding the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Background augmentations lead to substantial improvements (+1-2% on ImageNet-1k) in performance across a spectrum of state-of-the art self-supervised methods (MoCov2, BYOL, SwAV) on a variety of tasks, allowing us to reach within 0.3% of supervised performance. We also demonstrate that background augmentations improve robustness to a number of out of distribution settings, including natural adversarial examples, the backgrounds challenge, adversarial attacks, and ReaL ImageNet.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Leveraging background augmentations to encourage semantic focus in self-supervised contrastive learning<br>pdf: <a href="https://t.co/heJfQ5Ic3A">https://t.co/heJfQ5Ic3A</a><br>abs: <a href="https://t.co/nwM8yhMjZm">https://t.co/nwM8yhMjZm</a> <a href="https://t.co/dWwlCO2F88">pic.twitter.com/dWwlCO2F88</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374528206737338375?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Moving from Linear to Conic Markets for Electricity

Anubhav Ratha, Pierre Pinson, Hélène Le Cadre, Ana Virag, Jalal Kazempour

- retweets: 20, favorites: 38 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12122) | [pdf](https://arxiv.org/pdf/2103.12122)
- [econ.TH](https://arxiv.org/list/econ.TH/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

We propose a new forward electricity market framework that admits heterogeneous market participants with second-order cone strategy sets, who accurately express the nonlinearities in their costs and constraints through conic bids, and a network operator facing conic operational constraints. In contrast to the prevalent linear-programming-based electricity markets, we highlight how the inclusion of second-order cone constraints enables uncertainty-, asset- and network-awareness of the market, which is key to the successful transition towards an electricity system based on weather-dependent renewable energy sources. We analyze our general market-clearing proposal using conic duality theory to derive efficient spatially-differentiated prices for the multiple commodities, comprising of energy and flexibility services. Under the assumption of perfect competition, we prove the equivalence of the centrally-solved market-clearing optimization problem to a competitive spatial price equilibrium involving a set of rational and self-interested participants and a price setter. Finally, under common assumptions, we prove that moving towards conic markets does not incur the loss of desirable economic properties of markets, namely market efficiency, cost recovery and revenue adequacy. Our numerical studies focus on the specific use case of uncertainty-aware market design and demonstrate that the proposed conic market brings advantages over existing alternatives within the linear programming market framework.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Despite recent mathematical &amp; computational advances, electricity markets are still using a linear model, where simplifying assumptions are necessary. Shall we go beyond LP, and use a conic model? You may find this paper interesting to read: <a href="https://t.co/H13tQKZLzD">https://t.co/H13tQKZLzD</a><a href="https://twitter.com/anubhavratha?ref_src=twsrc%5Etfw">@anubhavratha</a> <a href="https://t.co/LHRE6fOFqB">pic.twitter.com/LHRE6fOFqB</a></p>&mdash; Jalal Kazempour (@JalalKazempour) <a href="https://twitter.com/JalalKazempour/status/1374740943136509952?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. DeFLOCNet: Deep Image Editing via Flexible Low-level Controls

Hongyu Liu, Ziyu Wan, Wei Huang, Yibing Song, Xintong Han, Jing Liao, Bing Jiang, Wei Liu

- retweets: 36, favorites: 22 (03/25/2021 09:03:26)

- links: [abs](https://arxiv.org/abs/2103.12723) | [pdf](https://arxiv.org/pdf/2103.12723)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

User-intended visual content fills the hole regions of an input image in the image editing scenario. The coarse low-level inputs, which typically consist of sparse sketch lines and color dots, convey user intentions for content creation (\ie, free-form editing). While existing methods combine an input image and these low-level controls for CNN inputs, the corresponding feature representations are not sufficient to convey user intentions, leading to unfaithfully generated content. In this paper, we propose DeFLOCNet which relies on a deep encoder-decoder CNN to retain the guidance of these controls in the deep feature representations. In each skip-connection layer, we design a structure generation block. Instead of attaching low-level controls to an input image, we inject these controls directly into each structure generation block for sketch line refinement and color propagation in the CNN feature space. We then concatenate the modulated features with the original decoder features for structure generation. Meanwhile, DeFLOCNet involves another decoder branch for texture generation and detail enhancement. Both structures and textures are rendered in the decoder, leading to user-intended editing results. Experiments on benchmarks demonstrate that DeFLOCNet effectively transforms different user intentions to create visually pleasing content.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeFLOCNet: Deep Image Editing via Flexible Low-level Controls<br>pdf: <a href="https://t.co/ex3LMpLQaF">https://t.co/ex3LMpLQaF</a><br>abs: <a href="https://t.co/VGllcDYX9J">https://t.co/VGllcDYX9J</a> <a href="https://t.co/2dHjYMtGWb">pic.twitter.com/2dHjYMtGWb</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374533554953150464?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. BossNAS: Exploring Hybrid CNN-transformers with Block-wisely  Self-supervised Neural Architecture Search

Changlin Li, Tao Tang, Guangrun Wang, Jiefeng Peng, Bing Wang, Xiaodan Liang, Xiaojun Chang

- retweets: 30, favorites: 23 (03/25/2021 09:03:27)

- links: [abs](https://arxiv.org/abs/2103.12424) | [pdf](https://arxiv.org/pdf/2103.12424)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A myriad of recent breakthroughs in hand-crafted neural architectures for visual recognition have highlighted the urgent need to explore hybrid architectures consisting of diversified building blocks. Meanwhile, neural architecture search methods are surging with an expectation to reduce human efforts. However, whether NAS methods can efficiently and effectively handle diversified search spaces with disparate candidates (e.g. CNNs and transformers) is still an open question. In this work, we present Block-wisely Self-supervised Neural Architecture Search (BossNAS), an unsupervised NAS method that addresses the problem of inaccurate architecture rating caused by large weight-sharing space and biased supervision in previous methods. More specifically, we factorize the search space into blocks and utilize a novel self-supervised training scheme, named ensemble bootstrapping, to train each block separately before searching them as a whole towards the population center. Additionally, we present HyTra search space, a fabric-like hybrid CNN-transformer search space with searchable down-sampling positions. On this challenging search space, our searched model, BossNet-T, achieves up to 82.2% accuracy on ImageNet, surpassing EfficientNet by 2.1% with comparable compute time. Moreover, our method achieves superior architecture rating accuracy with 0.78 and 0.76 Spearman correlation on the canonical MBConv search space with ImageNet and on NATS-Bench size search space with CIFAR-100, respectively, surpassing state-of-the-art NAS methods. Code and pretrained models are available at https://github.com/changlin31/BossNAS .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Star Wars Episode One fans 🤝 machine learning fans<br><br>&quot;BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search,&quot; Li et al.: <a href="https://t.co/fkKZDs88Jw">https://t.co/fkKZDs88Jw</a></p>&mdash; Miles Brundage (@Miles_Brundage) <a href="https://twitter.com/Miles_Brundage/status/1374556045104750594?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Replacing Rewards with Examples: Example-Based Policy Search via  Recursive Classification

Benjamin Eysenbach, Sergey Levine, Ruslan Salakhutdinov

- retweets: 14, favorites: 36 (03/25/2021 09:03:27)

- links: [abs](https://arxiv.org/abs/2103.12656) | [pdf](https://arxiv.org/pdf/2103.12656)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

In the standard Markov decision process formalism, users specify tasks by writing down a reward function. However, in many scenarios, the user is unable to describe the task in words or numbers, but can readily provide examples of what the world would look like if the task were solved. Motivated by this observation, we derive a control algorithm from first principles that aims to visit states that have a high probability of leading to successful outcomes, given only examples of successful outcome states. Prior work has approached similar problem settings in a two-stage process, first learning an auxiliary reward function and then optimizing this reward function using another reinforcement learning algorithm. In contrast, we derive a method based on recursive classification that eschews auxiliary reward functions and instead directly learns a value function from transitions and successful outcomes. Our method therefore requires fewer hyperparameters to tune and lines of code to debug. We show that our method satisfies a new data-driven Bellman equation, where examples take the place of the typical reward function term. Experiments show that our approach outperforms prior methods that learn explicit reward functions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification<br>pdf: <a href="https://t.co/wXfTiamtEN">https://t.co/wXfTiamtEN</a><br>abs: <a href="https://t.co/Lk8b0qyJvO">https://t.co/Lk8b0qyJvO</a><br>project page: <a href="https://t.co/MURPUjuPda">https://t.co/MURPUjuPda</a> <a href="https://t.co/aUUbRulUP1">pic.twitter.com/aUUbRulUP1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374571655394172933?ref_src=twsrc%5Etfw">March 24, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



