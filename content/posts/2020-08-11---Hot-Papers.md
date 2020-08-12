---
title: Hot Papers 2020-08-11
date: 2020-08-12T10:10:58.Z
template: "post"
draft: false
slug: "hot-papers-2020-08-11"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-08-11"
socialImage: "/media/flying-marine.jpg"

---

# 1. I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human  Pose and Mesh Estimation from a Single RGB Image

Gyeongsik Moon, Kyoung Mu Lee

- retweets: 145, favorites: 429 (08/12/2020 10:10:58)

- links: [abs](https://arxiv.org/abs/2008.03713) | [pdf](https://arxiv.org/pdf/2008.03713)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Most of the previous image-based 3D human pose and mesh estimation methods estimate parameters of the human mesh model from an input image. However, directly regressing the parameters from the input image is a highly non-linear mapping because it breaks the spatial relationship between pixels in the input image. In addition, it cannot model the prediction uncertainty, which can make training harder. To resolve the above issues, we propose I2L-MeshNet, an image-to-lixel (line+pixel) prediction network. The proposed I2L-MeshNet predicts the per-lixel likelihood on 1D heatmaps for each mesh vertex coordinate instead of directly regressing the parameters. Our lixel-based 1D heatmap preserves the spatial relationship in the input image and models the prediction uncertainty. We demonstrate the benefit of the image-to-lixel prediction and show that the proposed I2L-MeshNet outperforms previous methods. The code is publicly available \footnote{\url{https://github.com/mks0601/I2L-MeshNet_RELEASE}}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image<br>pdf: <a href="https://t.co/9WSyvFAbiX">https://t.co/9WSyvFAbiX</a><br>abs: <a href="https://t.co/mOYhmhf1z3">https://t.co/mOYhmhf1z3</a><br>github: <a href="https://t.co/9b50dl2nbj">https://t.co/9b50dl2nbj</a> <a href="https://t.co/MXsF99CcU1">pic.twitter.com/MXsF99CcU1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1292995989096235010?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Neural Light Transport for Relighting and View Synthesis

Xiuming Zhang, Sean Fanello, Yun-Ta Tsai, Tiancheng Sun, Tianfan Xue, Rohit Pandey, Sergio Orts-Escolano, Philip Davidson, Christoph Rhemann, Paul Debevec, Jonathan T. Barron, Ravi Ramamoorthi, William T. Freeman

- retweets: 88, favorites: 365 (08/12/2020 10:10:58)

- links: [abs](https://arxiv.org/abs/2008.03806) | [pdf](https://arxiv.org/pdf/2008.03806)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

The light transport (LT) of a scene describes how it appears under different lighting and viewing directions, and complete knowledge of a scene's LT enables the synthesis of novel views under arbitrary lighting. In this paper, we focus on image-based LT acquisition, primarily for human bodies within a light stage setup. We propose a semi-parametric approach to learn a neural representation of LT that is embedded in the space of a texture atlas of known geometric properties, and model all non-diffuse and global LT as residuals added to a physically-accurate diffuse base rendering. In particular, we show how to fuse previously seen observations of illuminants and views to synthesize a new image of the same scene under a desired lighting condition from a chosen viewpoint. This strategy allows the network to learn complex material effects (such as subsurface scattering) and global illumination, while guaranteeing the physical correctness of the diffuse LT (such as hard shadows). With this learned LT, one can relight the scene photorealistically with a directional light or an HDRI map, synthesize novel views with view-dependent effects, or do both simultaneously, all in a unified framework using a set of sparse, previously seen observations. Qualitative and quantitative experiments demonstrate that our neural LT (NLT) outperforms state-of-the-art solutions for relighting and view synthesis, without separate treatment for both problems that prior work requires.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Light Transport for Relighting and View Synthesis<br>pdf: <a href="https://t.co/uhqAVm3J7N">https://t.co/uhqAVm3J7N</a><br>abs: <a href="https://t.co/vkpIekQ4gF">https://t.co/vkpIekQ4gF</a><br>project page: <a href="https://t.co/mXvcUHw6Ue">https://t.co/mXvcUHw6Ue</a> <a href="https://t.co/FmmMcM7mDA">pic.twitter.com/FmmMcM7mDA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1292998751875932163?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Robust Bayesian inference of network structure from unreliable data

Jean-Gabriel Young, George T. Cantwell, M. E. J. Newman

- retweets: 63, favorites: 209 (08/12/2020 10:10:58)

- links: [abs](https://arxiv.org/abs/2008.03334) | [pdf](https://arxiv.org/pdf/2008.03334)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [stat.AP](https://arxiv.org/list/stat.AP/recent)

Most empirical studies of complex networks do not return direct, error-free measurements of network structure. Instead, they typically rely on indirect measurements that are often error-prone and unreliable. A fundamental problem in empirical network science is how to make the best possible estimates of network structure given such unreliable data. In this paper we describe a fully Bayesian method for reconstructing networks from observational data in any format, even when the data contain substantial measurement error and when the nature and magnitude of that error is unknown. The method is introduced through pedagogical case studies using real-world example networks, and specifically tailored to allow straightforward, computationally efficient implementation with a minimum of technical input. Computer code implementing the method is publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing &quot;Robust Bayesian inference of network structure from unreliable data,&quot; a (hopefully!) pedagogical introduction to inferring networks from noisy data -- with code.<br><br>w/ George T. Cantwell and MEJ Newman<br><br>📃Preprint: <a href="https://t.co/ne5jezRxuB">https://t.co/ne5jezRxuB</a> <a href="https://t.co/tqXufIbHwk">pic.twitter.com/tqXufIbHwk</a></p>&mdash; Jean-Gabriel Young (@_jgyou) <a href="https://twitter.com/_jgyou/status/1293193430647164928?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. EagerPy: Writing Code That Works Natively with PyTorch, TensorFlow, JAX,  and NumPy

Jonas Rauber, Matthias Bethge, Wieland Brendel

- retweets: 21, favorites: 98 (08/12/2020 10:10:58)

- links: [abs](https://arxiv.org/abs/2008.04175) | [pdf](https://arxiv.org/pdf/2008.04175)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MS](https://arxiv.org/list/cs.MS/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

EagerPy is a Python framework that lets you write code that automatically works natively with PyTorch, TensorFlow, JAX, and NumPy. Library developers no longer need to choose between supporting just one of these frameworks or reimplementing the library for each framework and dealing with code duplication. Users of such libraries can more easily switch frameworks without being locked in by a specific 3rd party library. Beyond multi-framework support, EagerPy also brings comprehensive type annotations and consistent support for method chaining to any framework. The latest documentation is available online at https://eagerpy.jonasrauber.de and the code can be found on GitHub at https://github.com/jonasrauber/eagerpy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;EagerPy: Writing Code That Works Natively with PyTorch, TensorFlow, JAX, and NumPy&quot;: <a href="https://t.co/7981r7nN5H">https://t.co/7981r7nN5H</a> Looks like a neat tool when collaborating with people who prefer a different Deep Learning lib, and you want to find a common denominator.</p>&mdash; Sebastian Raschka (@rasbt) <a href="https://twitter.com/rasbt/status/1293255652144484352?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Improving the Speed and Quality of GAN by Adversarial Training

Jiachen Zhong, Xuanqing Liu, Cho-Jui Hsieh

- retweets: 13, favorites: 69 (08/12/2020 10:10:58)

- links: [abs](https://arxiv.org/abs/2008.03364) | [pdf](https://arxiv.org/pdf/2008.03364)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Generative adversarial networks (GAN) have shown remarkable results in image generation tasks. High fidelity class-conditional GAN methods often rely on stabilization techniques by constraining the global Lipschitz continuity. Such regularization leads to less expressive models and slower convergence speed; other techniques, such as the large batch training, require unconventional computing power and are not widely accessible. In this paper, we develop an efficient algorithm, namely FastGAN (Free AdverSarial Training), to improve the speed and quality of GAN training based on the adversarial training technique. We benchmark our method on CIFAR10, a subset of ImageNet, and the full ImageNet datasets. We choose strong baselines such as SNGAN and SAGAN; the results demonstrate that our training algorithm can achieve better generation quality (in terms of the Inception score and Frechet Inception distance) with less overall training time. Most notably, our training algorithm brings ImageNet training to the broader public by requiring 2-4 GPUs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improving the Speed and Quality of GAN by Adversarial<br>Training<br>pdf: <a href="https://t.co/j9Qlg8Mv3H">https://t.co/j9Qlg8Mv3H</a><br>abs: <a href="https://t.co/3ySOiL5beQ">https://t.co/3ySOiL5beQ</a> <a href="https://t.co/cQ8wtKkzGD">pic.twitter.com/cQ8wtKkzGD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1293008289933004800?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Deep Sketch-guided Cartoon Video Synthesis

Xiaoyu Li, Bo Zhang, Jing Liao, Pedro V. Sander

- retweets: 12, favorites: 70 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.04149) | [pdf](https://arxiv.org/pdf/2008.04149)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel framework to produce cartoon videos by fetching the color information from two input keyframes while following the animated motion guided by a user sketch. The key idea of the proposed approach is to estimate the dense cross-domain correspondence between the sketch and cartoon video frames, following by a blending module with occlusion estimation to synthesize the middle frame guided by the sketch. After that, the inputs and the synthetic frame equipped with established correspondence are fed into an arbitrary-time frame interpolation pipeline to generate and refine additional inbetween frames. Finally, a video post-processing approach is used to further improve the result. Compared to common frame interpolation methods, our approach can address frames with relatively large motion and also has the flexibility to enable users to control the generated video sequences by editing the sketch guidance. By explicitly considering the correspondence between frames and the sketch, our methods can achieve high-quality synthetic results compared with image synthesis methods. Our results show that our system generalizes well to different movie frames, achieving better results than existing solutions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Sketch-guided Cartoon Video Synthesis<br>pdf: <a href="https://t.co/KrSbL0zsiN">https://t.co/KrSbL0zsiN</a><br>abs: <a href="https://t.co/gQfHqfMCVY">https://t.co/gQfHqfMCVY</a> <a href="https://t.co/VoXGslKIZe">pic.twitter.com/VoXGslKIZe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1293015394668642309?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. The Chess Transformer: Mastering Play using Generative Language Models

David Noever, Matt Ciolino, Josh Kalin

- retweets: 11, favorites: 55 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.04057) | [pdf](https://arxiv.org/pdf/2008.04057)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This work demonstrates that natural language transformers can support more generic strategic modeling, particularly for text-archived games. In addition to learning natural language skills, the abstract transformer architecture can generate meaningful moves on a chessboard. With further fine-tuning, the transformer learns complex gameplay by training on 2.8 million chess games in Portable Game Notation. After 30,000 training steps, OpenAI's Generative Pre-trained Transformer (GPT-2) optimizes weights for 774 million parameters. This fine-tuned Chess Transformer generates plausible strategies and displays game formations identifiable as classic openings, such as English or the Slav Exchange. Finally, in live play, the novel model demonstrates a human-to-transformer interface that correctly filters illegal moves and provides a novel method to challenge the transformer's chess strategies. We anticipate future work will build on this transformer's promise, particularly in other strategy games where features can capture the underlying complex rule syntax from simple but expressive player annotations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Chess Transformer: Mastering Play using Generative Language Models<br><br>pdf: <a href="https://t.co/1jCbVcApCg">https://t.co/1jCbVcApCg</a> <a href="https://t.co/hQqwFRfDVz">pic.twitter.com/hQqwFRfDVz</a></p>&mdash; Shawn Presser (@theshawwn) <a href="https://twitter.com/theshawwn/status/1293016222041178112?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Chess Transformer: Mastering Play using Generative Language Models<br>pdf: <a href="https://t.co/s7wsWenqGo">https://t.co/s7wsWenqGo</a><br>abs: <a href="https://t.co/C7FnlunhQH">https://t.co/C7FnlunhQH</a> <a href="https://t.co/80zd8Pyj58">pic.twitter.com/80zd8Pyj58</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1293013414781100032?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Spatiotemporal Contrastive Video Representation Learning

Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, Yin Cui

- retweets: 13, favorites: 51 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.03800) | [pdf](https://arxiv.org/pdf/2008.03800)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a self-supervised Contrastive Video Representation Learning (CVRL) method to learn spatiotemporal visual representations from unlabeled videos. Inspired by the recently proposed self-supervised contrastive learning framework, our representations are learned using a contrastive loss, where two clips from the same short video are pulled together in the embedding space, while clips from different videos are pushed away. We study what makes for good data augmentation for video self-supervised learning and find both spatial and temporal information are crucial. In particular, we propose a simple yet effective temporally consistent spatial augmentation method to impose strong spatial augmentations on each frame of a video clip while maintaining the temporal consistency across frames. For Kinetics-600 action recognition, a linear classifier trained on representations learned by CVRL achieves 64.1\% top-1 accuracy with a 3D-ResNet50 backbone, outperforming ImageNet supervised pre-training by 9.4\% and SimCLR unsupervised pre-training by 16.1\% using the same inflated 3D-ResNet50. The performance of CVRL can be further improved to 68.2\% with a larger 3D-ResNet50 (4$\times$) backbone, significantly closing the gap between unsupervised and supervised video representation learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new work: Spatiotemporal Contrastive Video Representation Learning (CVRL).<br><br>On Kinetics-600, we achieve 64.1% top-1 linear classification accuracy with a 3D-ResNet50 backbone and 68.2% with a larger 3D-ResNet50 (4x) backbone.<br><br>Link: <a href="https://t.co/96avHJPFFB">https://t.co/96avHJPFFB</a> <a href="https://t.co/vHJi6tDN1W">pic.twitter.com/vHJi6tDN1W</a></p>&mdash; Yin Cui (@YinCui1) <a href="https://twitter.com/YinCui1/status/1293215426982240263?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. TriFinger: An Open-Source Robot for Learning Dexterity

Manuel Wüthrich, Felix Widmaier, Felix Grimminger, Joel Akpo, Shruti Joshi, Vaibhav Agrawal, Bilal Hammoud, Majid Khadiv, Miroslav Bogdanovic, Vincent Berenz, Julian Viereck, Maximilien Naveau, Ludovic Righetti, Bernhard Schölkopf, Stefan Bauer

- retweets: 11, favorites: 48 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.03596) | [pdf](https://arxiv.org/pdf/2008.03596)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Dexterous object manipulation remains an open problem in robotics, despite the rapid progress in machine learning during the past decade. We argue that a hindrance is the high cost of experimentation on real systems, in terms of both time and money. We address this problem by proposing an open-source robotic platform which can safely operate without human supervision. The hardware is inexpensive (about \SI{5000}[\$]{}) yet highly dynamic, robust, and capable of complex interaction with external objects. The software operates at 1-kilohertz and performs safety checks to prevent the hardware from breaking. The easy-to-use front-end (in C++ and Python) is suitable for real-time control as well as deep reinforcement learning. In addition, the software framework is largely robot-agnostic and can hence be used independently of the hardware proposed herein. Finally, we illustrate the potential of the proposed platform through a number of experiments, including real-time optimal control, deep reinforcement learning from scratch, throwing, and writing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TriFinger: An Open-Source Robot for Learning Dexterity <a href="https://t.co/PXLkKkbU9e">https://t.co/PXLkKkbU9e</a> <a href="https://t.co/yjAJxl3wzq">pic.twitter.com/yjAJxl3wzq</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1292995473255460864?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. VAW-GAN for Singing Voice Conversion with Non-parallel Training Data

Junchen Lu, Kun Zhou, Berrak Sisman, Haizhou Li

- retweets: 11, favorites: 42 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.03992) | [pdf](https://arxiv.org/pdf/2008.03992)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Singing voice conversion aims to convert singer's voice from source to target without changing singing content. Parallel training data is typically required for the training of singing voice conversion system, that is however not practical in real-life applications. Recent encoder-decoder structures, such as variational autoencoding Wasserstein generative adversarial network (VAW-GAN), provide an effective way to learn a mapping through non-parallel training data. In this paper, we propose a singing voice conversion framework that is based on VAW-GAN. We train an encoder to disentangle singer identity and singing prosody (F0 contour) from phonetic content. By conditioning on singer identity and F0, the decoder generates output spectral features with unseen target singer identity, and improves the F0 rendering. Experimental results show that the proposed framework achieves better performance than the baseline frameworks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wasserstein Generative Adversarial Networks for Singing Voice Conversion<br>pdf: <a href="https://t.co/P8rqMSJ1HS">https://t.co/P8rqMSJ1HS</a><br>abs: <a href="https://t.co/uVdYTaKIkD">https://t.co/uVdYTaKIkD</a><br>project page: <a href="https://t.co/coMbVkzj4L">https://t.co/coMbVkzj4L</a><br>github: <a href="https://t.co/UU6slqY9sy">https://t.co/UU6slqY9sy</a> <a href="https://t.co/sEUPOJIxVj">pic.twitter.com/sEUPOJIxVj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1293020341426675712?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Two-branch Recurrent Network for Isolating Deepfakes in Videos

Iacopo Masi, Aditya Killekar, Royston Marian Mascarenhas, Shenoy Pratik Gurudatt, Wael AbdAlmageed

- retweets: 35, favorites: 17 (08/12/2020 10:10:59)

- links: [abs](https://arxiv.org/abs/2008.03412) | [pdf](https://arxiv.org/pdf/2008.03412)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The current spike of hyper-realistic faces artificially generated using deepfakes calls for media forensics solutions that are tailored to video streams and work reliably with a low false alarm rate at the video level. We present a method for deepfake detection based on a two-branch network structure that isolates digitally manipulated faces by learning to amplify artifacts while suppressing the high-level face content. Unlike current methods that extract spatial frequencies as a preprocessing step, we propose a two-branch structure: one branch propagates the original information, while the other branch suppresses the face content yet amplifies multi-band frequencies using a Laplacian of Gaussian (LoG) as a bottleneck layer. To better isolate manipulated faces, we derive a novel cost function that, unlike regular classification, compresses the variability of natural faces and pushes away the unrealistic facial samples in the feature space. Our two novel components show promising results on the FaceForensics++, Celeb-DF, and Facebook's DFDC preview benchmarks, when compared to prior work. We then offer a full, detailed ablation study of our network architecture and cost function. Finally, although the bar is still high to get very remarkable figures at a very low false alarm rate, our study shows that we can achieve good video-level performance when cross-testing in terms of video-level AUC.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Two-branch Recurrent Network for Isolating Deepfakes in Videos. <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/IoT?src=hash&amp;ref_src=twsrc%5Etfw">#IoT</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/RStats?src=hash&amp;ref_src=twsrc%5Etfw">#RStats</a> <a href="https://twitter.com/hashtag/TensorFlow?src=hash&amp;ref_src=twsrc%5Etfw">#TensorFlow</a> <a href="https://twitter.com/hashtag/Java?src=hash&amp;ref_src=twsrc%5Etfw">#Java</a> <a href="https://twitter.com/hashtag/JavaScript?src=hash&amp;ref_src=twsrc%5Etfw">#JavaScript</a> <a href="https://twitter.com/hashtag/ReactJS?src=hash&amp;ref_src=twsrc%5Etfw">#ReactJS</a> <a href="https://twitter.com/hashtag/GoLang?src=hash&amp;ref_src=twsrc%5Etfw">#GoLang</a> <a href="https://twitter.com/hashtag/Serverless?src=hash&amp;ref_src=twsrc%5Etfw">#Serverless</a> <a href="https://twitter.com/hashtag/Linux?src=hash&amp;ref_src=twsrc%5Etfw">#Linux</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/Programming?src=hash&amp;ref_src=twsrc%5Etfw">#Programming</a>  <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/Deepfakes?src=hash&amp;ref_src=twsrc%5Etfw">#Deepfakes</a> <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a><a href="https://t.co/KOuq6oWcBT">https://t.co/KOuq6oWcBT</a> <a href="https://t.co/0qkoqDYP6Y">pic.twitter.com/0qkoqDYP6Y</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1293295836974252032?ref_src=twsrc%5Etfw">August 11, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



