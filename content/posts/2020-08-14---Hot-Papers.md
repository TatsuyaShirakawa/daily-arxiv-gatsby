---
title: Hot Papers 2020-08-14
date: 2020-08-15T13:07:55.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-14"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-14"
socialImage: "/media/flying-marine.jpg"

---

# 1. What Should Not Be Contrastive in Contrastive Learning

Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell

- retweets: 35, favorites: 135 (08/15/2020 13:07:55)

- links: [abs](https://arxiv.org/abs/2008.05659) | [pdf](https://arxiv.org/pdf/2008.05659)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent self-supervised contrastive methods have been able to produce impressive transferable visual representations by learning to be invariant to different data augmentations. However, these methods implicitly assume a particular set of representational invariances (e.g., invariance to color), and can perform poorly when a downstream task violates this assumption (e.g., distinguishing red vs. yellow cars). We introduce a contrastive learning framework which does not require prior knowledge of specific, task-dependent invariances. Our model learns to capture varying and invariant factors for visual representations by constructing separate embedding spaces, each of which is invariant to all but one augmentation. We use a multi-head network with a shared backbone which captures information across each augmentation and alone outperforms all baselines on downstream tasks. We further find that the concatenation of the invariant and varying spaces performs best across all tasks we investigate, including coarse-grained, fine-grained, and few-shot downstream classification tasks, and various data corruptions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What Should Not Be Contrastive in Contrastive Learning. <br>The model learns to capture varying and invariant factors for visual representations by constructing separate embedding spaces, each of which is invariant to all but one augmentation.<a href="https://t.co/YKQUzLN6TH">https://t.co/YKQUzLN6TH</a><a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://t.co/l7SzTkCvMG">pic.twitter.com/l7SzTkCvMG</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1294094315942227968?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Full-Body Awareness from Partial Observations

Chris Rockwell, David F. Fouhey

- retweets: 29, favorites: 128 (08/15/2020 13:07:55)

- links: [abs](https://arxiv.org/abs/2008.06046) | [pdf](https://arxiv.org/pdf/2008.06046)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

There has been great progress in human 3D mesh recovery and great interest in learning about the world from consumer video data. Unfortunately current methods for 3D human mesh recovery work rather poorly on consumer video data, since on the Internet, unusual camera viewpoints and aggressive truncations are the norm rather than a rarity. We study this problem and make a number of contributions to address it: (i) we propose a simple but highly effective self-training framework that adapts human 3D mesh recovery systems to consumer videos and demonstrate its application to two recent systems; (ii) we introduce evaluation protocols and keypoint annotations for 13K frames across four consumer video datasets for studying this task, including evaluations on out-of-image keypoints; and (iii) we show that our method substantially improves PCK and human-subject judgments compared to baselines, both on test videos from the dataset it was trained on, as well as on three other datasets without further adaptation. Project website: https://crockwell.github.io/partial_humans

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can we understand humans in internet video? Our <a href="https://twitter.com/hashtag/ECCV2020?src=hash&amp;ref_src=twsrc%5Etfw">#ECCV2020</a> work presents a simple but highly effective method for self-training on unlabeled video! We annotate four datasets to evaluate &amp; show large gains.<br><br>Project Page: <a href="https://t.co/tLLNqkLqhS">https://t.co/tLLNqkLqhS</a><br>arXiv: <a href="https://t.co/nW0WOJNfUP">https://t.co/nW0WOJNfUP</a> <a href="https://t.co/xx39I6UJAA">pic.twitter.com/xx39I6UJAA</a></p>&mdash; Chris Rockwell (@_crockwell) <a href="https://twitter.com/_crockwell/status/1294078655321247744?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Compiling a Higher-Order Smart Contract Language to LLVM

Vaivaswatha Nagaraj, Jacob Johannsen, Anton Trunov, George Pîrlea, Amrit Kumar, Ilya Sergey

- retweets: 41, favorites: 108 (08/15/2020 13:07:56)

- links: [abs](https://arxiv.org/abs/2008.05555) | [pdf](https://arxiv.org/pdf/2008.05555)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

Scilla is a higher-order polymorphic typed intermediate level language for implementing smart contracts. In this talk, we describe a Scilla compiler targeting LLVM, with a focus on mapping Scilla types, values, and its functional language constructs to LLVM-IR.   The compiled LLVM-IR, when executed with LLVM's JIT framework, achieves a speedup of about 10x over the reference interpreter on a typical Scilla contract. This reduced latency is crucial in the setting of blockchains, where smart contracts are executed as parts of transactions, to achieve peak transactions processed per second. Experiments on the Ackermann function achieved a speedup of more than 45x. This talk abstract is aimed at both programming language researchers looking to implement an LLVM based compiler for their functional language, as well as at LLVM practitioners.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scilla to LLVM compiler project by <a href="https://twitter.com/VaivaswathaN?ref_src=twsrc%5Etfw">@VaivaswathaN</a> is in full swing for <a href="https://twitter.com/search?q=%24ZIL&amp;src=ctag&amp;ref_src=twsrc%5Etfw">$ZIL</a>. We are seeing 10x improvement on performance with the compiler compared to the ref interpreter. Details on the mapping in this paper: <a href="https://t.co/tx1wibceKp">https://t.co/tx1wibceKp</a><br>CC: <a href="https://twitter.com/secondstateinc?ref_src=twsrc%5Etfw">@secondstateinc</a> <a href="https://twitter.com/stevanlohja?ref_src=twsrc%5Etfw">@stevanlohja</a> <a href="https://twitter.com/etclabs?ref_src=twsrc%5Etfw">@etclabs</a> <a href="https://t.co/A1czqqPrsD">pic.twitter.com/A1czqqPrsD</a></p>&mdash; Amrit Kummer (@maqstik) <a href="https://twitter.com/maqstik/status/1294170007891566593?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Powers of layers for image-to-image translation

Hugo Touvron, Matthijs Douze, Matthieu Cord, Hervé Jégou

- retweets: 14, favorites: 53 (08/15/2020 13:07:56)

- links: [abs](https://arxiv.org/abs/2008.05763) | [pdf](https://arxiv.org/pdf/2008.05763)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

We propose a simple architecture to address unpaired image-to-image translation tasks: style or class transfer, denoising, deblurring, deblocking, etc. We start from an image autoencoder architecture with fixed weights. For each task we learn a residual block operating in the latent space, which is iteratively called until the target domain is reached. A specific training schedule is required to alleviate the exponentiation effect of the iterations. At test time, it offers several advantages: the number of weight parameters is limited and the compositional design allows one to modulate the strength of the transformation with the number of iterations. This is useful, for instance, when the type or amount of noise to suppress is not known in advance. Experimentally, we provide proofs of concepts showing the interest of our method for many transformations. The performance of our model is comparable or better than CycleGAN with significantly fewer parameters.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Powers of layers for image-to-image translation<br>pdf: <a href="https://t.co/bdW96IkDF1">https://t.co/bdW96IkDF1</a><br>abs: <a href="https://t.co/qbm88TJ8Ej">https://t.co/qbm88TJ8Ej</a> <a href="https://t.co/pQVm80W4SG">pic.twitter.com/pQVm80W4SG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1294085249853534208?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Generating Person-Scene Interactions in 3D Scenes

Siwei Zhang, Yan Zhang, Qianli Ma, Michael J. Black, Siyu Tang

- retweets: 12, favorites: 47 (08/15/2020 13:07:56)

- links: [abs](https://arxiv.org/abs/2008.05570) | [pdf](https://arxiv.org/pdf/2008.05570)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

High fidelity digital 3D environments have been proposed in recent years; however, it remains extreme challenging to automatically equip such environment with realistic human bodies. Existing work utilizes images, depths, or semantic maps to represent the scene, and parametric human models to represent 3D bodies in the scene. While being straightforward, their generated human-scene interactions are often lack of naturalness and physical plausibility. Our key observation is that humans interact with the world through body-scene contact. To explicitly and effectively represent the physical contact between the body and the world is essential for modeling human-scene interaction. To that end, we propose a novel interaction representation, which explicitly encodes the proximity between the human body and the 3D scene around it. Specifically, given a set of basis points on a scene mesh, we leverage a conditional variational autoencoder to synthesize the distance from every basis point to its closest point on a human body. The synthesized proximal relationship between the human body and the scene can indicate which region a person tends to contact. Furthermore, based on such synthesized proximity, we can effectively obtain expressive 3D human bodies that naturally interact with the 3D scene. Our perceptual study shows that our model significantly improves the state-of-the-art method, approaching the realism of real human-scene interaction. We believe our method makes an important step towards the fully automatic synthesis of realistic 3D human bodies in 3D scenes. Our code and model will be publicly available for research purpose.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating Person-Scene Interactions in 3D Scenes<br>pdf: <a href="https://t.co/TSUtX7llnT">https://t.co/TSUtX7llnT</a><br>abs: <a href="https://t.co/JzfR21G8F6">https://t.co/JzfR21G8F6</a> <a href="https://t.co/vPK66UgDN0">pic.twitter.com/vPK66UgDN0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1294095787211870210?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Overcoming Model Bias for Robust Offline Deep Reinforcement Learning

Phillip Swazinna, Steffen Udluft, Thomas Runkler

- retweets: 32, favorites: 18 (08/15/2020 13:07:56)

- links: [abs](https://arxiv.org/abs/2008.05533) | [pdf](https://arxiv.org/pdf/2008.05533)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

State-of-the-art reinforcement learning algorithms mostly rely on being allowed to directly interact with their environment to collect millions of observations. This makes it hard to transfer their success to industrial control problems, where simulations are often very costly or do not exist at all. Furthermore, interacting with (and especially exploring in) the real, physical environment has the potential to lead to catastrophic events. We thus propose a novel model-based RL algorithm, called MOOSE (MOdel-based Offline policy Search with Ensembles) which can train a policy from a pre-existing, fixed dataset. It ensures that dynamics models are able to accurately assess policy performance by constraining the policy to stay within the support of the data. We design MOOSE deliberately similar to state-of-the-art model-free, offline (a.k.a. batch) RL algorithms BEAR and BCQ, with the main difference being that our algorithm is model-based. We compare the algorithms on the Industrial Benchmark and Mujoco continuous control tasks in terms of robust performance and find that MOOSE almost always outperforms its model-free counterparts by far.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Overcoming Model Bias for Robust Offline Deep Reinforcement Learning. <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/TensorFlow?src=hash&amp;ref_src=twsrc%5Etfw">#TensorFlow</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/GoLang?src=hash&amp;ref_src=twsrc%5Etfw">#GoLang</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/IIoT?src=hash&amp;ref_src=twsrc%5Etfw">#IIoT</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a><a href="https://t.co/21PhHaUHe0">https://t.co/21PhHaUHe0</a> <a href="https://t.co/ynDiyMsdRK">pic.twitter.com/ynDiyMsdRK</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1294394500136280064?ref_src=twsrc%5Etfw">August 14, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



