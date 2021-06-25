---
title: Hot Papers 2021-06-25
date: 2021-06-26T08:14:47.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. Provably efficient machine learning for quantum many-body problems

Hsin-Yuan Huang, Richard Kueng, Giacomo Torlai, Victor V. Albert, John Preskill

- retweets: 6903, favorites: 93 (06/26/2021 08:14:47)

- links: [abs](https://arxiv.org/abs/2106.12627) | [pdf](https://arxiv.org/pdf/2106.12627)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Classical machine learning (ML) provides a potentially powerful approach to solving challenging quantum many-body problems in physics and chemistry. However, the advantages of ML over more traditional methods have not been firmly established. In this work, we prove that classical ML algorithms can efficiently predict ground state properties of gapped Hamiltonians in finite spatial dimensions, after learning from data obtained by measuring other Hamiltonians in the same quantum phase of matter. In contrast, under widely accepted complexity theory assumptions, classical algorithms that do not learn from data cannot achieve the same guarantee. We also prove that classical ML algorithms can efficiently classify a wide range of quantum phases of matter. Our arguments are based on the concept of a classical shadow, a succinct classical description of a many-body quantum state that can be constructed in feasible quantum experiments and be used to predict many properties of the state. Extensive numerical experiments corroborate our theoretical results in a variety of scenarios, including Rydberg atom systems, 2D random Heisenberg models, symmetry-protected topological phases, and topologically ordered phases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">❓Can classical machine learning models solve challenging problems in physics that no classical algorithms could solve?<br><br>We prove the affirmative (for quantum many-body problems) in <a href="https://t.co/DqhbpBLdGL">https://t.co/DqhbpBLdGL</a><br>with <a href="https://twitter.com/RichardKueng?ref_src=twsrc%5Etfw">@RichardKueng</a>, <a href="https://twitter.com/giactorlai?ref_src=twsrc%5Etfw">@giactorlai</a>, <a href="https://twitter.com/victorvalbert?ref_src=twsrc%5Etfw">@victorvalbert</a>, <a href="https://twitter.com/preskill?ref_src=twsrc%5Etfw">@preskill</a> [🧵1/13] <a href="https://t.co/e6YLyqh8BW">pic.twitter.com/e6YLyqh8BW</a></p>&mdash; Hsin-Yuan (Robert) Huang (@RobertHuangHY) <a href="https://twitter.com/RobertHuangHY/status/1408230497512087554?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. HyperNeRF: A Higher-Dimensional Representation for Topologically Varying  Neural Radiance Fields

Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, Steven M. Seitz

- retweets: 3782, favorites: 383 (06/26/2021 08:14:47)

- links: [abs](https://arxiv.org/abs/2106.13228) | [pdf](https://arxiv.org/pdf/2106.13228)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Neural Radiance Fields (NeRF) are able to reconstruct scenes with unprecedented fidelity, and various recent works have extended NeRF to handle dynamic scenes. A common approach to reconstruct such non-rigid scenes is through the use of a learned deformation field mapping from coordinates in each input image into a canonical template coordinate space. However, these deformation-based approaches struggle to model changes in topology, as topological changes require a discontinuity in the deformation field, but these deformation fields are necessarily continuous. We address this limitation by lifting NeRFs into a higher dimensional space, and by representing the 5D radiance field corresponding to each individual input image as a slice through this "hyper-space". Our method is inspired by level set methods, which model the evolution of surfaces as slices through a higher dimensional surface. We evaluate our method on two tasks: (i) interpolating smoothly between "moments", i.e., configurations of the scene, seen in the input images while maintaining visual plausibility, and (ii) novel-view synthesis at fixed moments. We show that our method, which we dub HyperNeRF, outperforms existing methods on both tasks by significant margins. Compared to Nerfies, HyperNeRF reduces average error rates by 8.6% for interpolation and 8.8% for novel-view synthesis, as measured by LPIPS.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing HyperNeRF! By lifting NeRF into higher dimensions, we fix a big limitation of Nerfies: topological changes. HyperNeRF can handle topological variations such as facial expressions that Nerfies can&#39;t.<br>Website: <a href="https://t.co/fcH57u0EIM">https://t.co/fcH57u0EIM</a><br>arXiv: <a href="https://t.co/AezSuU4RF6">https://t.co/AezSuU4RF6</a><br>(1/N) <a href="https://t.co/BAEoQnbGeT">pic.twitter.com/BAEoQnbGeT</a></p>&mdash; Keunhong Park (@KeunhongP) <a href="https://twitter.com/KeunhongP/status/1408223966431375362?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Charformer: Fast Character Transformers via Gradient-based Subword  Tokenization

Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, Donald Metzler

- retweets: 3552, favorites: 525 (06/26/2021 08:14:47)

- links: [abs](https://arxiv.org/abs/2106.12672) | [pdf](https://arxiv.org/pdf/2106.12672)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

State-of-the-art models in natural language processing rely on separate rigid subword tokenization algorithms, which limit their generalization ability and adaptation to new settings. In this paper, we propose a new model inductive bias that learns a subword tokenization end-to-end as part of the model. To this end, we introduce a soft gradient-based subword tokenization module (GBST) that automatically learns latent subword representations from characters in a data-driven fashion. Concretely, GBST enumerates candidate subword blocks and learns to score them in a position-wise fashion using a block scoring network. We additionally introduce Charformer, a deep Transformer model that integrates GBST and operates on the byte level. Via extensive experiments on English GLUE, multilingual, and noisy text datasets, we show that Charformer outperforms a series of competitive byte-level baselines while generally performing on par and sometimes outperforming subword-based models. Additionally, Charformer is fast, improving the speed of both vanilla byte-level and subword-level Transformers by 28%-100% while maintaining competitive quality. We believe this work paves the way for highly performant token-free models that are trained completely end-to-end.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Charformer: Fast Character Transformers via Gradient-based Subword Tokenization<br><br>- Learns a subword tokenization end-to-end as part of the model <br>- Outperforms byte-level baselines on GLUE etc while generally performing on par<a href="https://t.co/bjRtaBmWAp">https://t.co/bjRtaBmWAp</a> <a href="https://t.co/77pm3zccF5">pic.twitter.com/77pm3zccF5</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1408227412098240513?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our new work from <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> and <a href="https://twitter.com/DeepMind?ref_src=twsrc%5Etfw">@DeepMind</a>. &quot;Charformer: Fast Character Transformers via Gradient-based Subword Tokenization (paper: <a href="https://t.co/kqdTbRRK9f">https://t.co/kqdTbRRK9f</a>) <a href="https://t.co/MeUtMoAQMw">pic.twitter.com/MeUtMoAQMw</a></p>&mdash; Yi Tay (@ytay017) <a href="https://twitter.com/ytay017/status/1408469060052946951?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. AudioCLIP: Extending CLIP to Image, Text and Audio

Andrey Guzhov, Federico Raue, Jörn Hees, Andreas Dengel

- retweets: 2254, favorites: 210 (06/26/2021 08:14:48)

- links: [abs](https://arxiv.org/abs/2106.13043) | [pdf](https://arxiv.org/pdf/2106.13043)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

In the past, the rapidly evolving field of sound classification greatly benefited from the application of methods from other domains. Today, we observe the trend to fuse domain-specific tasks and approaches together, which provides the community with new outstanding models.   In this work, we present an extension of the CLIP model that handles audio in addition to text and images. Our proposed model incorporates the ESResNeXt audio-model into the CLIP framework using the AudioSet dataset. Such a combination enables the proposed model to perform bimodal and unimodal classification and querying, while keeping CLIP's ability to generalize to unseen datasets in a zero-shot inference fashion.   AudioCLIP achieves new state-of-the-art results in the Environmental Sound Classification (ESC) task, out-performing other approaches by reaching accuracies of 90.07% on the UrbanSound8K and 97.15% on the ESC-50 datasets. Further it sets new baselines in the zero-shot ESC-task on the same datasets 68.78% and 69.40%, respectively).   Finally, we also assess the cross-modal querying performance of the proposed model as well as the influence of full and partial training on the results. For the sake of reproducibility, our code is published.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AudioCLIP: Extending CLIP to Image, Text and Audio⋆<br>pdf: <a href="https://t.co/aYXK7gYjRs">https://t.co/aYXK7gYjRs</a><br>abs: <a href="https://t.co/XUT9AGNGwy">https://t.co/XUT9AGNGwy</a><br><br>achieves new sota results in the ESC task, out-performing other approaches by reaching accuracies of 90.07 % on the UrbanSound8K and 97.15 % on the ESC-50 datasets <a href="https://t.co/N4ApIxfAgp">pic.twitter.com/N4ApIxfAgp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408222986562326533?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Task-agnostic Continual Learning with Hybrid Probabilistic Models

Polina Kirichenko, Mehrdad Farajtabar, Dushyant Rao, Balaji Lakshminarayanan, Nir Levine, Ang Li, Huiyi Hu, Andrew Gordon Wilson, Razvan Pascanu

- retweets: 1564, favorites: 280 (06/26/2021 08:14:48)

- links: [abs](https://arxiv.org/abs/2106.12772) | [pdf](https://arxiv.org/pdf/2106.12772)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Learning new tasks continuously without forgetting on a constantly changing data distribution is essential for real-world problems but extremely challenging for modern deep learning. In this work we propose HCL, a Hybrid generative-discriminative approach to Continual Learning for classification. We model the distribution of each task and each class with a normalizing flow. The flow is used to learn the data distribution, perform classification, identify task changes, and avoid forgetting, all leveraging the invertibility and exact likelihood which are uniquely enabled by the normalizing flow model. We use the generative capabilities of the flow to avoid catastrophic forgetting through generative replay and a novel functional regularization technique. For task identification, we use state-of-the-art anomaly detection techniques based on measuring the typicality of the model's statistics. We demonstrate the strong performance of HCL on a range of continual learning benchmarks such as split-MNIST, split-CIFAR, and SVHN-MNIST.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share that<br>(1) It is my birthday 🎂<br>(2) We have a new paper &quot;Task-agnostic Continual Learning with Hybrid Probabilistic Models&quot; on arxiv today!  We design a hybrid generative-discriminative model based on normalizing flows for continual learning <a href="https://t.co/Xbp9fVky7W">https://t.co/Xbp9fVky7W</a> <a href="https://t.co/XvLRDwMxuf">pic.twitter.com/XvLRDwMxuf</a></p>&mdash; Polina Kirichenko (@polkirichenko) <a href="https://twitter.com/polkirichenko/status/1408286165858410503?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Real-time gravitational-wave science with neural posterior estimation

Maximilian Dax, Stephen R. Green, Jonathan Gair, Jakob H. Macke, Alessandra Buonanno, Bernhard Schölkopf

- retweets: 462, favorites: 80 (06/26/2021 08:14:48)

- links: [abs](https://arxiv.org/abs/2106.12594) | [pdf](https://arxiv.org/pdf/2106.12594)
- [gr-qc](https://arxiv.org/list/gr-qc/recent) | [astro-ph.IM](https://arxiv.org/list/astro-ph.IM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We demonstrate unprecedented accuracy for rapid gravitational-wave parameter estimation with deep learning. Using neural networks as surrogates for Bayesian posterior distributions, we analyze eight gravitational-wave events from the first LIGO-Virgo Gravitational-Wave Transient Catalog and find very close quantitative agreement with standard inference codes, but with inference times reduced from O(day) to a minute per event. Our networks are trained using simulated data, including an estimate of the detector-noise characteristics near the event. This encodes the signal and noise models within millions of neural-network parameters, and enables inference for any observed data consistent with the training distribution, accounting for noise nonstationarity from event to event. Our algorithm -- called "DINGO" -- sets a new standard in fast-and-accurate inference of physical parameters of detected gravitational-wave events, which should enable real-time data analysis without sacrificing accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Preprint by <a href="https://twitter.com/maximilian_dax?ref_src=twsrc%5Etfw">@maximilian_dax</a> <a href="https://twitter.com/stephen_r_green?ref_src=twsrc%5Etfw">@stephen_r_green</a> J Gair A Buonanno <a href="https://twitter.com/bschoelkopf?ref_src=twsrc%5Etfw">@bschoelkopf</a>: DINGO infers posteriors of big (15 d) gravitational wave models, with amazingly high accuracy, in real time (~ 1min, MCMC takes ~days), on real data! <a href="https://twitter.com/mpi_grav?ref_src=twsrc%5Etfw">@mpi_grav</a> <a href="https://twitter.com/MPI_IS?ref_src=twsrc%5Etfw">@MPI_IS</a> <a href="https://twitter.com/ml4science?ref_src=twsrc%5Etfw">@ml4science</a> <a href="https://t.co/KkYJt3xTLE">https://t.co/KkYJt3xTLE</a> <a href="https://t.co/8Zdr7GYutG">pic.twitter.com/8Zdr7GYutG</a></p>&mdash; Jakob Macke (@jakhmack) <a href="https://twitter.com/jakhmack/status/1408400497912987650?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Gender differences in scientific careers: A large-scale bibliometric  analysis

Hanjo Boekhout, Inge van der Weijden, Ludo Waltman

- retweets: 484, favorites: 32 (06/26/2021 08:14:48)

- links: [abs](https://arxiv.org/abs/2106.12624) | [pdf](https://arxiv.org/pdf/2106.12624)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

We present a large-scale bibliometric analysis of gender differences in scientific careers, covering all scientific disciplines and a large number of countries worldwide. We take a longitudinal perspective in which we trace the publication careers of almost six million male and female researchers in the period 1996-2018. Our analysis reveals an increasing trend in the percentage of women starting a career as publishing researcher, from 33% in 2000 to about 40% in recent years. Looking at cohorts of male and female researchers that started their publication career in the same year, we find that women seem to be somewhat less likely to continue their career as publishing researcher than men, but the difference is small. We also observe that men produce on average between 15% and 20% more publications than women. Moreover, in biomedical disciplines, men are about 25% more likely than women to be last author of a publication, suggesting that men tend to have more senior roles than women. Compared with cross-sectional studies, our longitudinal analysis has the advantage of providing a more in-depth understanding of gender imbalances among authors of scientific publications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large-scale bibliometric analysis of gender differences in scientific careers, by Hanjo Boekhout, <a href="https://twitter.com/WeijdenInge?ref_src=twsrc%5Etfw">@WeijdenInge</a> and myself <a href="https://t.co/09AJdXhK3N">https://t.co/09AJdXhK3N</a>; unlike many earlier studies, we take longitudinal rather than cross-sectional approach, leading to more detailed insights <a href="https://twitter.com/cwtsleiden?ref_src=twsrc%5Etfw">@cwtsleiden</a></p>&mdash; Ludo Waltman (@LudoWaltman) <a href="https://twitter.com/LudoWaltman/status/1408371831757852675?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. GaussiGAN: Controllable Image Synthesis with 3D Gaussians from Unposed  Silhouettes

Youssef A.Mejjati, Isa Milefchik, Aaron Gokaslan, Oliver Wang, Kwang In Kim, James Tompkin

- retweets: 272, favorites: 68 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13215) | [pdf](https://arxiv.org/pdf/2106.13215)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present an algorithm that learns a coarse 3D representation of objects from unposed multi-view 2D mask supervision, then uses it to generate detailed mask and image texture. In contrast to existing voxel-based methods for unposed object reconstruction, our approach learns to represent the generated shape and pose with a set of self-supervised canonical 3D anisotropic Gaussians via a perspective camera, and a set of per-image transforms. We show that this approach can robustly estimate a 3D space for the camera and object, while recent baselines sometimes struggle to reconstruct coherent 3D spaces in this setting. We show results on synthetic datasets with realistic lighting, and demonstrate object insertion with interactive posing. With our work, we help move towards structured representations that handle more real-world variation in learning-based object reconstruction.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GaussiGAN: Controllable Image Synthesis with 3D Gaussians from Unposed Silhouettes<br>pdf: <a href="https://t.co/Ey8uBJUo0L">https://t.co/Ey8uBJUo0L</a><br>abs: <a href="https://t.co/vl4g5S0MSg">https://t.co/vl4g5S0MSg</a><br>project page: <a href="https://t.co/k1bSlZgxA9">https://t.co/k1bSlZgxA9</a> <a href="https://t.co/cObffpfm7t">pic.twitter.com/cObffpfm7t</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408231412348633088?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Model-Based Reinforcement Learning via Latent-Space Collocation

Oleh Rybkin, Chuning Zhu, Anusha Nagabandi, Kostas Daniilidis, Igor Mordatch, Sergey Levine

- retweets: 233, favorites: 107 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13229) | [pdf](https://arxiv.org/pdf/2106.13229)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

The ability to plan into the future while utilizing only raw high-dimensional observations, such as images, can provide autonomous agents with broad capabilities. Visual model-based reinforcement learning (RL) methods that plan future actions directly have shown impressive results on tasks that require only short-horizon reasoning, however, these methods struggle on temporally extended tasks. We argue that it is easier to solve long-horizon tasks by planning sequences of states rather than just actions, as the effects of actions greatly compound over time and are harder to optimize. To achieve this, we draw on the idea of collocation, which has shown good results on long-horizon tasks in optimal control literature, and adapt it to the image-based setting by utilizing learned latent state space models. The resulting latent collocation method (LatCo) optimizes trajectories of latent states, which improves over previously proposed shooting methods for visual model-based RL on tasks with sparse rewards and long-term goals. Videos and code at https://orybkin.github.io/latco/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Visual model-based RL with a real trajectory optimizer -- collocation + visual model-based RL can &quot;imagine&quot; moving objects to the goal, and then figure out how to actually accomplish it. See below for <a href="https://twitter.com/_oleh?ref_src=twsrc%5Etfw">@_oleh</a>&#39;s excellent summary, check out the paper here: <a href="https://t.co/9AGIzCY9jz">https://t.co/9AGIzCY9jz</a> <a href="https://t.co/Q7vecze7K3">https://t.co/Q7vecze7K3</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1408285102765871106?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Model-Based Reinforcement Learning via Latent-Space Collocation<br>pdf: <a href="https://t.co/ZvfP9mNeOl">https://t.co/ZvfP9mNeOl</a><br>abs: <a href="https://t.co/t1wmCY13Nw">https://t.co/t1wmCY13Nw</a><br>project page: <a href="https://t.co/xHVq3Cc75C">https://t.co/xHVq3Cc75C</a> <a href="https://t.co/CfAylmNV0L">pic.twitter.com/CfAylmNV0L</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408270745478774789?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Video Swin Transformer

Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu

- retweets: 182, favorites: 39 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13230) | [pdf](https://arxiv.org/pdf/2106.13230)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The vision community is witnessing a modeling shift from CNNs to Transformers, where pure Transformer architectures have attained top accuracy on the major video recognition benchmarks. These video models are all built on Transformer layers that globally connect patches across the spatial and temporal dimensions. In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including on action recognition (84.9 top-1 accuracy on Kinetics-400 and 86.1 top-1 accuracy on Kinetics-600 with ~20x less pre-training data and ~3x smaller model size) and temporal modeling (69.6 top-1 accuracy on Something-Something v2). The code and models will be made publicly available at https://github.com/SwinTransformer/Video-Swin-Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Video Swin Transformer<br>pdf: <a href="https://t.co/sgwrWyNgg2">https://t.co/sgwrWyNgg2</a><br>abs: <a href="https://t.co/BLUtR3gfmR">https://t.co/BLUtR3gfmR</a><br>github: <a href="https://t.co/mOLVY9dls6">https://t.co/mOLVY9dls6</a><br><br>84.9 top-1 accuracy on Kinetics-400 and 86.1 top-1 accuracy on Kinetics-600 with ∼20× less pre-training data and ∼3× smaller model size <a href="https://t.co/WrpLVt9uHE">pic.twitter.com/WrpLVt9uHE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408228725905408005?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Unsupervised Learning of Depth and Depth-of-Field Effect from Natural  Images with Aperture Rendering Generative Adversarial Networks

Takuhiro Kaneko

- retweets: 90, favorites: 34 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13041) | [pdf](https://arxiv.org/pdf/2106.13041)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Understanding the 3D world from 2D projected natural images is a fundamental challenge in computer vision and graphics. Recently, an unsupervised learning approach has garnered considerable attention owing to its advantages in data collection. However, to mitigate training limitations, typical methods need to impose assumptions for viewpoint distribution (e.g., a dataset containing various viewpoint images) or object shape (e.g., symmetric objects). These assumptions often restrict applications; for instance, the application to non-rigid objects or images captured from similar viewpoints (e.g., flower or bird images) remains a challenge. To complement these approaches, we propose aperture rendering generative adversarial networks (AR-GANs), which equip aperture rendering on top of GANs, and adopt focus cues to learn the depth and depth-of-field (DoF) effect of unlabeled natural images. To address the ambiguities triggered by unsupervised setting (i.e., ambiguities between smooth texture and out-of-focus blurs, and between foreground and background blurs), we develop DoF mixture learning, which enables the generator to learn real image distribution while generating diverse DoF images. In addition, we devise a center focus prior to guiding the learning direction. In the experiments, we demonstrate the effectiveness of AR-GANs in various datasets, such as flower, bird, and face images, demonstrate their portability by incorporating them into other 3D representation learning GANs, and validate their applicability in shallow DoF rendering.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Learning of Depth and Depth-of-Field Effect from Natural Images with Aperture Rendering Generative Adversarial Networks<br>pdf: <a href="https://t.co/rtsy1fwTkW">https://t.co/rtsy1fwTkW</a><br><br>a family of GANs, AR-GANs, which can learn depth and DoF effect from unconstrained natural images <a href="https://t.co/4A9Cbp8SPS">pic.twitter.com/4A9Cbp8SPS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408237010058350596?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Towards Biologically Plausible Convolutional Networks

Roman Pogodin, Yash Mehta, Timothy P. Lillicrap, Peter E. Latham

- retweets: 90, favorites: 33 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13031) | [pdf](https://arxiv.org/pdf/2106.13031)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent)

Convolutional networks are ubiquitous in deep learning. They are particularly useful for images, as they reduce the number of parameters, reduce training time, and increase accuracy. However, as a model of the brain they are seriously problematic, since they require weight sharing - something real neurons simply cannot do. Consequently, while neurons in the brain can be locally connected (one of the features of convolutional networks), they cannot be convolutional. Locally connected but non-convolutional networks, however, significantly underperform convolutional ones. This is troublesome for studies that use convolutional networks to explain activity in the visual system. Here we study plausible alternatives to weight sharing that aim at the same regularization principle, which is to make each neuron within a pool react similarly to identical inputs. The most natural way to do that is by showing the network multiple translations of the same image, akin to saccades in animal vision. However, this approach requires many translations, and doesn't remove the performance gap. We propose instead to add lateral connectivity to a locally connected network, and allow learning via Hebbian plasticity. This requires the network to pause occasionally for a sleep-like phase of "weight sharing". This method enables locally connected networks to achieve nearly convolutional performance on ImageNet, thus supporting convolutional networks as a model of the visual stream.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Biologically Plausible Convolutional Networks<br>pdf: <a href="https://t.co/e9ZU2uweR0">https://t.co/e9ZU2uweR0</a><br>abs: <a href="https://t.co/ItzGsz9pvA">https://t.co/ItzGsz9pvA</a><br><br>enables locally connected networks to achieve nearly convolutional performance on ImageNet, thus supporting convolutional networks as a model of<br>the visual stream <a href="https://t.co/BgR2AhRIJA">pic.twitter.com/BgR2AhRIJA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408278790384541697?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. VOLO: Vision Outlooker for Visual Recognition

Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, Shuicheng Yan

- retweets: 38, favorites: 70 (06/26/2021 08:14:49)

- links: [abs](https://arxiv.org/abs/2106.13112) | [pdf](https://arxiv.org/pdf/2106.13112)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Visual recognition has been dominated by convolutionalneural networks (CNNs) for years. Though recently the pre-vailing vision transformers (ViTs) have shown great poten-tial of self-attention based models in ImageNet classifica-tion, their performance is still inferior to latest SOTA CNNsif no extra data are provided. In this work, we aim to closethe performance gap and demonstrate that attention-basedmodels are indeed able to outperform CNNs. We found thatthe main factor limiting the performance of ViTs for Ima-geNet classification is their low efficacy in encoding fine-level features into the token representations. To resolvethis, we introduce a noveloutlook attentionand present asimple and general architecture, termed Vision Outlooker(VOLO). Unlike self-attention that focuses on global depen-dency modeling at a coarse level, the outlook attention aimsto efficiently encode finer-level features and contexts intotokens, which are shown to be critical for recognition per-formance but largely ignored by the self-attention. Experi-ments show that our VOLO achieves 87.1% top-1 accuracyon ImageNet-1K classification, being the first model exceed-ing 87% accuracy on this competitive benchmark, withoutusing any extra training data. In addition, the pre-trainedVOLO transfers well to downstream tasks, such as seman-tic segmentation. We achieve 84.3% mIoU score on thecityscapes validation set and 54.3% on the ADE20K valida-tion set. Code is available at https://github.com/sail-sg/volo.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VOLO: Vision Outlooker for Visual Recognition<br>pdf: <a href="https://t.co/9c0618jY5N">https://t.co/9c0618jY5N</a><br>abs: <a href="https://t.co/RmUeCEXhRu">https://t.co/RmUeCEXhRu</a><br><br>achieves 84.3% mIoU score on the cityscapes validation set and 54.3% on the ADE20K validation set <a href="https://t.co/b1d8bTRU7p">pic.twitter.com/b1d8bTRU7p</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408224104809910274?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">VOLO、名前から Object Detection かと思いきや Classification やんけ。ViT 系にまた強いやつが来たのか<a href="https://t.co/IKIy6z1O0N">https://t.co/IKIy6z1O0N</a></p>&mdash; イ 表 (@tawatawara) <a href="https://twitter.com/tawatawara/status/1408353390422224911?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Unsupervised Topic Segmentation of Meetings with BERT Embeddings

Alessandro Solbiati, Kevin Heffernan, Georgios Damaskinos, Shivani Poddar, Shubham Modi, Jacques Cali

- retweets: 72, favorites: 27 (06/26/2021 08:14:50)

- links: [abs](https://arxiv.org/abs/2106.12978) | [pdf](https://arxiv.org/pdf/2106.12978)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Topic segmentation of meetings is the task of dividing multi-person meeting transcripts into topic blocks. Supervised approaches to the problem have proven intractable due to the difficulties in collecting and accurately annotating large datasets. In this paper we show how previous unsupervised topic segmentation methods can be improved using pre-trained neural architectures. We introduce an unsupervised approach based on BERT embeddings that achieves a 15.5% reduction in error rate over existing unsupervised approaches applied to two popular datasets for meeting transcripts.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Topic Segmentation of Meetings with BERT Embeddings<br>pdf: <a href="https://t.co/y4E15ma6PJ">https://t.co/y4E15ma6PJ</a><br><br>unsupervised approach based on BERT embeddings, achieves a 15.5% reduction in error rate over existing unsupervised approaches applied to two popular datasets for meeting transcripts <a href="https://t.co/Nmm6eezLL5">pic.twitter.com/Nmm6eezLL5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408280539115704324?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Exploring Corruption Robustness: Inductive Biases in Vision Transformers  and MLP-Mixers

Katelyn Morrison, Benjamin Gilby, Colton Lipchak, Adam Mattioli, Adriana Kovashka

- retweets: 56, favorites: 31 (06/26/2021 08:14:50)

- links: [abs](https://arxiv.org/abs/2106.13122) | [pdf](https://arxiv.org/pdf/2106.13122)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently, vision transformers and MLP-based models have been developed in order to address some of the prevalent weaknesses in convolutional neural networks. Due to the novelty of transformers being used in this domain along with the self-attention mechanism, it remains unclear to what degree these architectures are robust to corruptions. Despite some works proposing that data augmentation remains essential for a model to be robust against corruptions, we propose to explore the impact that the architecture has on corruption robustness. We find that vision transformer architectures are inherently more robust to corruptions than the ResNet-50 and MLP-Mixers. We also find that vision transformers with 5 times fewer parameters than a ResNet-50 have more shape bias. Our code is available to reproduce.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Exploring Corruption Robustness: Inductive Biases in Vision Transformers and MLP-Mixers<br>pdf: <a href="https://t.co/X1ENjbSSaf">https://t.co/X1ENjbSSaf</a><br>github: <a href="https://t.co/UCwzfQ9dDR">https://t.co/UCwzfQ9dDR</a><br><br>vision transformer architectures are inherently more robust to corruptions<br>than the ResNet-50 and MLP-Mixers <a href="https://t.co/07lwgC0pRZ">pic.twitter.com/07lwgC0pRZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408223557981675523?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. FitVid: Overfitting in Pixel-Level Video Prediction

Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, Dumitru Erhan

- retweets: 21, favorites: 48 (06/26/2021 08:14:50)

- links: [abs](https://arxiv.org/abs/2106.13195) | [pdf](https://arxiv.org/pdf/2106.13195)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

An agent that is capable of predicting what happens next can perform a variety of tasks through planning with no additional training. Furthermore, such an agent can internally represent the complex dynamics of the real-world and therefore can acquire a representation useful for a variety of visual perception tasks. This makes predicting the future frames of a video, conditioned on the observed past and potentially future actions, an interesting task which remains exceptionally challenging despite many recent advances. Existing video prediction models have shown promising results on simple narrow benchmarks but they generate low quality predictions on real-life datasets with more complicated dynamics or broader domain. There is a growing body of evidence that underfitting on the training data is one of the primary causes for the low quality predictions. In this paper, we argue that the inefficient use of parameters in the current video models is the main reason for underfitting. Therefore, we introduce a new architecture, named FitVid, which is capable of severe overfitting on the common benchmarks while having similar parameter count as the current state-of-the-art models. We analyze the consequences of overfitting, illustrating how it can produce unexpected outcomes such as generating high quality output by repeating the training data, and how it can be mitigated using existing image augmentation techniques. As a result, FitVid outperforms the current state-of-the-art models across four different video prediction benchmarks on four different metrics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FitVid: Overfitting in Pixel-Level Video Prediction<br>pdf: <a href="https://t.co/uPHgciqaLM">https://t.co/uPHgciqaLM</a><br>project page: <a href="https://t.co/zNb2huGkPH">https://t.co/zNb2huGkPH</a><br>github: <a href="https://t.co/Wf1CUGQblC">https://t.co/Wf1CUGQblC</a><br><br>outperforms the current sota models across four different video prediction benchmarks on four different metrics <a href="https://t.co/vhYAgndISN">pic.twitter.com/vhYAgndISN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1408233696965103620?ref_src=twsrc%5Etfw">June 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



