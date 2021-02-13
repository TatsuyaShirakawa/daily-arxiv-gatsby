---
title: Hot Papers 2021-02-12
date: 2021-02-13T22:40:53.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. High-Performance Large-Scale Image Recognition Without Normalization

Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan

- retweets: 10105, favorites: 10 (02/13/2021 22:40:53)

- links: [abs](https://arxiv.org/abs/2102.06171) | [pdf](https://arxiv.org/pdf/2102.06171)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Batch normalization is a key component of most image classification models, but it has many undesirable properties stemming from its dependence on the batch size and interactions between examples. Although recent work has succeeded in training deep ResNets without normalization layers, these models do not match the test accuracies of the best batch-normalized networks, and are often unstable for large learning rates or strong data augmentations. In this work, we develop an adaptive gradient clipping technique which overcomes these instabilities, and design a significantly improved class of Normalizer-Free ResNets. Our smaller models match the test accuracy of an EfficientNet-B7 on ImageNet while being up to 8.7x faster to train, and our largest models attain a new state-of-the-art top-1 accuracy of 86.5%. In addition, Normalizer-Free models attain significantly better performance than their batch-normalized counterparts when finetuning on ImageNet after large-scale pre-training on a dataset of 300 million labeled images, with our best models obtaining an accuracy of 89.2%. Our code is available at https://github.com/deepmind/ deepmind-research/tree/master/nfnets

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Releasing NFNets: SOTA on ImageNet. Without normalization layers!<a href="https://t.co/eUPSNle0r2">https://t.co/eUPSNle0r2</a><br>Code: <a href="https://t.co/YuwULNqpTR">https://t.co/YuwULNqpTR</a><br><br>This is the third paper in a series that began by studying the benefits of BatchNorm and ended by designing highly performant networks w/o it.<br><br>A thread:<br><br>1/8 <a href="https://t.co/8MQH2uqsJE">pic.twitter.com/8MQH2uqsJE</a></p>&mdash; Soham De (@sohamde_) <a href="https://twitter.com/sohamde_/status/1360219419977342984?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Neural Re-rendering for Full-frame Video Stabilization

Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, Jia-Bin Huang

- retweets: 1434, favorites: 210 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.06205) | [pdf](https://arxiv.org/pdf/2102.06205)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Existing video stabilization methods either require aggressive cropping of frame boundaries or generate distortion artifacts on the stabilized frames. In this work, we present an algorithm for full-frame video stabilization by first estimating dense warp fields. Full-frame stabilized frames can then be synthesized by fusing warped contents from neighboring frames. The core technical novelty lies in our learning-based hybrid-space fusion that alleviates artifacts caused by optical flow inaccuracy and fast-moving objects. We validate the effectiveness of our method on the NUS and selfie video datasets. Extensive experiment results demonstrate the merits of our approach over prior video stabilization methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Re-rendering for Full-frame Video Stabilization<br>pdf: <a href="https://t.co/tWE5kwWvfB">https://t.co/tWE5kwWvfB</a><br>abs: <a href="https://t.co/b4W3LU9Gag">https://t.co/b4W3LU9Gag</a><br>project page: <a href="https://t.co/6jhm8iRUSr">https://t.co/6jhm8iRUSr</a> <a href="https://t.co/GDUXvaWlXs">pic.twitter.com/GDUXvaWlXs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360086383294091269?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals

Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool

- retweets: 532, favorites: 133 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.06191) | [pdf](https://arxiv.org/pdf/2102.06191)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Being able to learn dense semantic representations of images without supervision is an important problem in computer vision. However, despite its significance, this problem remains rather unexplored, with a few exceptions that considered unsupervised semantic segmentation on small-scale datasets with a narrow visual domain. In this paper, we make a first attempt to tackle the problem on datasets that have been traditionally utilized for the supervised case. To achieve this, we introduce a novel two-step framework that adopts a predetermined prior in a contrastive optimization objective to learn pixel embeddings. This marks a large deviation from existing works that relied on proxy tasks or end-to-end clustering. Additionally, we argue about the importance of having a prior that contains information about objects, or their parts, and discuss several possibilities to obtain such a prior in an unsupervised manner.   Extensive experimental evaluation shows that the proposed method comes with key advantages over existing works. First, the learned pixel embeddings can be directly clustered in semantic groups using K-Means. Second, the method can serve as an effective unsupervised pre-training for the semantic segmentation task. In particular, when fine-tuning the learned representations using just 1% of labeled examples on PASCAL, we outperform supervised ImageNet pre-training by 7.1% mIoU. The code is available at https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals<br>pdf: <a href="https://t.co/jrykNmyQIX">https://t.co/jrykNmyQIX</a><br>abs: <a href="https://t.co/HKv1ycu7SV">https://t.co/HKv1ycu7SV</a><br>github: <a href="https://t.co/d2w6imxNlK">https://t.co/d2w6imxNlK</a> <a href="https://t.co/YSXhohnq2q">pic.twitter.com/YSXhohnq2q</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360079945796567045?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Scaling Up Visual and Vision-Language Representation Learning With Noisy  Text Supervision

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig

- retweets: 464, favorites: 143 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.05918) | [pdf](https://arxiv.org/pdf/2102.05918)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Pre-trained representations are becoming crucial for many NLP and perception tasks. While representation learning in NLP has transitioned to training on raw text without human annotations, visual and vision-language representations still rely heavily on curated training datasets that are expensive or require expert knowledge. For vision applications, representations are mostly learned using datasets with explicit class labels such as ImageNet or OpenImages. For vision-language, popular datasets like Conceptual Captions, MSCOCO, or CLIP all involve a non-trivial data collection (and cleaning) process. This costly curation process limits the size of datasets and hence hinders the scaling of trained models. In this paper, we leverage a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset. A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss. We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme. Our visual representation achieves strong performance when transferred to classification tasks such as ImageNet and VTAB. The aligned visual and language representations also set new state-of-the-art results on Flickr30K and MSCOCO benchmarks, even when compared with more sophisticated cross-attention models. The representations also enable cross-modality search with complex text and text + image queries.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision<br>pdf: <a href="https://t.co/G21HZ8ZWZl">https://t.co/G21HZ8ZWZl</a><br>abs: <a href="https://t.co/OAS4fAEoIi">https://t.co/OAS4fAEoIi</a> <a href="https://t.co/DLkOOaHo3M">pic.twitter.com/DLkOOaHo3M</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360059261624197123?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Brain Modelling as a Service: The Virtual Brain on EBRAINS

Michael Schirner, Lia Domide, Dionysios Perdikis, Paul Triebkorn, Leon Stefanovski, Roopa Pai, Paula Popa, Bogdan Valean, Jessica Palmer, Chloê Langford, André Blickensdörfer, Michiel van der Vlag, Sandra Diaz-Pier, Alexander Peyser, Wouter Klijn, Dirk Pleiter, Anne Nahm, Oliver Schmid, Marmaduke Woodman, Lyuba Zehl, Jan Fousek, Spase Petkoski, Lionel Kusch, Meysam Hashemi, Daniele Marinazzo, Jean-François Mangin, Agnes Flöel, Simisola Akintoye, Bernd Carsten Stahl, Michael Cepic, Emily Johnson, Anthony R. McIntosh, Claus C. Hilgetag, Marc Morgan, Bernd Schuller, Alex Upton, Colin McMurtrie, Timo Dickscheid, Jan G. Bjaalie, Katrin Amunts, Jochen Mersmann, Viktor Jirsa, Petra Ritter

- retweets: 441, favorites: 68 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.05888) | [pdf](https://arxiv.org/pdf/2102.05888)
- [cs.CE](https://arxiv.org/list/cs.CE/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

The Virtual Brain (TVB) is now available as open-source cloud ecosystem on EBRAINS, a shared digital research platform for brain science. It offers services for constructing, simulating and analysing brain network models (BNMs) including the TVB network simulator; magnetic resonance imaging (MRI) processing pipelines to extract structural and functional connectomes; multiscale co-simulation of spiking and large-scale networks; a domain specific language for automatic high-performance code generation from user-specified models; simulation-ready BNMs of patients and healthy volunteers; Bayesian inference of epilepsy spread; data and code for mouse brain simulation; and extensive educational material. TVB cloud services facilitate reproducible online collaboration and discovery of data assets, models, and software embedded in scalable and secure workflows, a precondition for research on large cohort data sets, better generalizability and clinical translation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Brain Modelling as a Service: The Virtual Brain on EBRAINS<a href="https://t.co/NHyXaV782M">https://t.co/NHyXaV782M</a> - Fantastic team work!<a href="https://twitter.com/HumanBrainProj?ref_src=twsrc%5Etfw">@HumanBrainProj</a> <a href="https://twitter.com/EBRAINS_eu?ref_src=twsrc%5Etfw">@EBRAINS_eu</a> <a href="https://twitter.com/pswieboda?ref_src=twsrc%5Etfw">@pswieboda</a> <a href="https://twitter.com/EOSC_eu?ref_src=twsrc%5Etfw">@EOSC_eu</a> <a href="https://twitter.com/berlinnovation?ref_src=twsrc%5Etfw">@berlinnovation</a> <a href="https://twitter.com/ChariteBerlin?ref_src=twsrc%5Etfw">@ChariteBerlin</a> <a href="https://twitter.com/Fenix_RI_eu?ref_src=twsrc%5Etfw">@Fenix_RI_eu</a> <a href="https://twitter.com/juniquefr?ref_src=twsrc%5Etfw">@juniquefr</a> <a href="https://twitter.com/ECDigitalFuture?ref_src=twsrc%5Etfw">@ECDigitalFuture</a> <a href="https://twitter.com/BIDSstandard?ref_src=twsrc%5Etfw">@BIDSstandard</a>  <a href="https://twitter.com/ECN_Berlin?ref_src=twsrc%5Etfw">@ECN_Berlin</a> <a href="https://twitter.com/bccn_berlin?ref_src=twsrc%5Etfw">@bccn_berlin</a> <a href="https://twitter.com/TVB_cloud?ref_src=twsrc%5Etfw">@TVB_cloud</a> <a href="https://t.co/xGFtyUV7bB">pic.twitter.com/xGFtyUV7bB</a></p>&mdash; Petra Ritter (@_PetraRitter) <a href="https://twitter.com/_PetraRitter/status/1360155766133567489?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Self-Supervised VQ-VAE For One-Shot Music Style Transfer

Ondřej Cífka, Alexey Ozerov, Umut Şimşekli, Gaël Richard

- retweets: 420, favorites: 60 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.05749) | [pdf](https://arxiv.org/pdf/2102.05749)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Neural style transfer, allowing to apply the artistic style of one image to another, has become one of the most widely showcased computer vision applications shortly after its introduction. In contrast, related tasks in the music audio domain remained, until recently, largely untackled. While several style conversion methods tailored to musical signals have been proposed, most lack the 'one-shot' capability of classical image style transfer algorithms. On the other hand, the results of existing one-shot audio style transfer methods on musical inputs are not as compelling. In this work, we are specifically interested in the problem of one-shot timbre transfer. We present a novel method for this task, based on an extension of the vector-quantized variational autoencoder (VQ-VAE), along with a simple self-supervised learning strategy designed to obtain disentangled representations of timbre and pitch. We evaluate the method using a set of objective metrics and show that it is able to outperform selected baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised VQ-VAE For One-Shot Music Style Transfer<br>pdf: <a href="https://t.co/iQ3mWA7phF">https://t.co/iQ3mWA7phF</a><br>abs: <a href="https://t.co/NmbCKePTCX">https://t.co/NmbCKePTCX</a><br>project page: <a href="https://t.co/vcwWaHzazA">https://t.co/vcwWaHzazA</a><br>github: <a href="https://t.co/N9gQvpTJAw">https://t.co/N9gQvpTJAw</a> <a href="https://t.co/U3ImrdGAvg">pic.twitter.com/U3ImrdGAvg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360056385833226241?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. A-NeRF: Surface-free Human 3D Pose Refinement via Neural Rendering

Shih-Yang Su, Frank Yu, Michael Zollhoefer, Helge Rhodin

- retweets: 274, favorites: 137 (02/13/2021 22:40:54)

- links: [abs](https://arxiv.org/abs/2102.06199) | [pdf](https://arxiv.org/pdf/2102.06199)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

While deep learning has reshaped the classical motion capture pipeline, generative, analysis-by-synthesis elements are still in use to recover fine details if a high-quality 3D model of the user is available. Unfortunately, obtaining such a model for every user a priori is challenging, time-consuming, and limits the application scenarios. We propose a novel test-time optimization approach for monocular motion capture that learns a volumetric body model of the user in a self-supervised manner. To this end, our approach combines the advantages of neural radiance fields with an articulated skeleton representation. Our proposed skeleton embedding serves as a common reference that links constraints across time, thereby reducing the number of required camera views from traditionally dozens of calibrated cameras, down to a single uncalibrated one. As a starting point, we employ the output of an off-the-shelf model that predicts the 3D skeleton pose. The volumetric body shape and appearance is then learned from scratch, while jointly refining the initial pose estimate. Our approach is self-supervised and does not require any additional ground truth labels for appearance, pose, or 3D shape. We demonstrate that our novel combination of a discriminative pose estimation technique with surface-free analysis-by-synthesis outperforms purely discriminative monocular pose estimation approaches and generalizes well to multiple views.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A-NeRF: Surface-free Human 3D Pose Refinement via Neural Rendering<br>pdf: <a href="https://t.co/N5hvjkDhtM">https://t.co/N5hvjkDhtM</a><br>abs: <a href="https://t.co/4yIwcCGSBd">https://t.co/4yIwcCGSBd</a><br>project page: <a href="https://t.co/I7XiUdq3g7">https://t.co/I7XiUdq3g7</a> <a href="https://t.co/BlsAWAj1Uu">pic.twitter.com/BlsAWAj1Uu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360052016723218432?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our A-NeRF: Surface-free Human 3D Pose Refinement via Neural Rendering<br>pdf: <a href="https://t.co/JUua8ilSty">https://t.co/JUua8ilSty</a><br>project page: <a href="https://t.co/5SDn87T1HG">https://t.co/5SDn87T1HG</a> <a href="https://t.co/LJ6WtPF8xT">pic.twitter.com/LJ6WtPF8xT</a></p>&mdash; Helge Rhodin (@HelgeRhodin) <a href="https://twitter.com/HelgeRhodin/status/1360355821670998018?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. SWAGAN: A Style-based Wavelet-driven Generative Model

Rinon Gal, Dana Cohen, Amit Bermano, Daniel Cohen-Or

- retweets: 114, favorites: 136 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.06108) | [pdf](https://arxiv.org/pdf/2102.06108)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

In recent years, considerable progress has been made in the visual quality of Generative Adversarial Networks (GANs). Even so, these networks still suffer from degradation in quality for high-frequency content, stemming from a spectrally biased architecture, and similarly unfavorable loss functions. To address this issue, we present a novel general-purpose Style and WAvelet based GAN (SWAGAN) that implements progressive generation in the frequency domain. SWAGAN incorporates wavelets throughout its generator and discriminator architectures, enforcing a frequency-aware latent representation at every step of the way. This approach yields enhancements in the visual quality of the generated images, and considerably increases computational performance. We demonstrate the advantage of our method by integrating it into the SyleGAN2 framework, and verifying that content generation in the wavelet domain leads to higher quality images with more realistic high-frequency content. Furthermore, we verify that our model's latent space retains the qualities that allow StyleGAN to serve as a basis for a multitude of editing tasks, and show that our frequency-aware approach also induces improved downstream visual quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SWAGAN: A Style-based WAvelet-driven Generative Model<br>pdf: <a href="https://t.co/pp9p90Sc5M">https://t.co/pp9p90Sc5M</a><br>abs: <a href="https://t.co/PjaI9xRYEn">https://t.co/PjaI9xRYEn</a> <a href="https://t.co/cY8sHU0g3U">pic.twitter.com/cY8sHU0g3U</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360054317219078144?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">SWAGAN: A Style-based Wavelet-driven Generative Model<a href="https://t.co/w4rlhal32I">https://t.co/w4rlhal32I</a> StyleGAN2のアーキテクチャをwavelet係数の予測に置き換えたことでStyleGAN2より実時間、イテレーション数ともに高速に良いFIDに達する。補完もより自然な気がする。 <a href="https://t.co/a5M5eLzaE2">pic.twitter.com/a5M5eLzaE2</a></p>&mdash; ワクワクさん（深層学習） (@mosko_mule) <a href="https://twitter.com/mosko_mule/status/1360064732045082630?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Efficient Neural Networks for Real-time Analog Audio Effect Modeling

Christian J. Steinmetz, Joshua D. Reiss

- retweets: 121, favorites: 76 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.06200) | [pdf](https://arxiv.org/pdf/2102.06200)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Deep learning approaches have demonstrated success in the task of modeling analog audio effects such as distortion and overdrive. Nevertheless, challenges remain in modeling more complex effects, such as dynamic range compressors, along with their variable parameters. Previous methods are computationally complex, and noncausal, prohibiting real-time operation, which is critical for use in audio production contexts. They additionally utilize large training datasets, which are time-intensive to generate. In this work, we demonstrate that shallower temporal convolutional networks (TCNs) that exploit very large dilation factors for significant receptive field can achieve state-of-the-art performance, while remaining efficient. Not only are these models found to be perceptually similar to the original effect, they achieve a 4x speedup, enabling real-time operation on CPU, and can be trained using only 1% of the data from previous methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We show that neural networks can emulate a vintage analog compressor along with its controls. With some architectural changes, they can run in real-time on CPU, and be trained using only a few minutes of data.<br><br>demo: <a href="https://t.co/lSwVP0GWch">https://t.co/lSwVP0GWch</a><br>paper: <a href="https://t.co/hbPzReGHqP">https://t.co/hbPzReGHqP</a> <a href="https://t.co/QoLVHtObdx">pic.twitter.com/QoLVHtObdx</a></p>&mdash; Christian Steinmetz (@csteinmetz1) <a href="https://twitter.com/csteinmetz1/status/1360207590035255297?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Zero-one laws for provability logic: Axiomatizing validity in almost all  models and almost all frames

Rineke Verbrugge

- retweets: 110, favorites: 53 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.05947) | [pdf](https://arxiv.org/pdf/2102.05947)
- [cs.LO](https://arxiv.org/list/cs.LO/recent)

It has been shown in the late 1960s that each formula of first-order logic without constants and function symbols obeys a zero-one law: As the number of elements of finite models increases, every formula holds either in almost all or in almost no models of that size. Therefore, many properties of models, such as having an even number of elements, cannot be expressed in the language of first-order logic. Halpern and Kapron proved zero-one laws for classes of models corresponding to the modal logics K, T, S4, and S5.   In this paper, we prove zero-one laws for provability logic with respect to both model and frame validity. Moreover, we axiomatize validity in almost all relevant finite models and in almost all relevant finite frames, leading to two different axiom systems. In the proofs, we use a combinatorial result by Kleitman and Rothschild about the structure of almost all finite partial orders. On the way, we also show that a previous result by Halpern and Kapron about the axiomatization of almost sure frame validity for S4 is not correct. Finally, we consider the complexity of deciding whether a given formula is almost surely valid in the relevant finite models and frames.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In this new paper, I prove 2 zero-one laws for provability logic: with respect to model and frame validity. I axiomatize validity in almost all relevant finite models and almost all relevant finite frames, both by infinite sets of axioms. Comments welcome! <a href="https://t.co/hlTUJL6A9I">https://t.co/hlTUJL6A9I</a> <a href="https://t.co/PpRZH2DhcK">pic.twitter.com/PpRZH2DhcK</a></p>&mdash; Rineke Verbrugge (@RinekeV) <a href="https://twitter.com/RinekeV/status/1360145614735048705?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Less is More: ClipBERT for Video-and-Language Learning via Sparse  Sampling

Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, Jingjing Liu

- retweets: 90, favorites: 64 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.06183) | [pdf](https://arxiv.org/pdf/2102.06183)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

The canonical approach to video-and-language learning (e.g., video question answering) dictates a neural model to learn from offline-extracted dense video features from vision models and text features from language models. These feature extractors are trained independently and usually on tasks different from the target domains, rendering these fixed features sub-optimal for downstream tasks. Moreover, due to the high computational overload of dense video features, it is often difficult (or infeasible) to plug feature extractors directly into existing approaches for easy finetuning. To provide a remedy to this dilemma, we propose a generic framework ClipBERT that enables affordable end-to-end learning for video-and-language tasks, by employing sparse sampling, where only a single or a few sparsely sampled short clips from a video are used at each training step. Experiments on text-to-video retrieval and video question answering on six datasets demonstrate that ClipBERT outperforms (or is on par with) existing methods that exploit full-length videos, suggesting that end-to-end learning with just a few sparsely sampled clips is often more accurate than using densely extracted offline features from full-length videos, proving the proverbial less-is-more principle. Videos in the datasets are from considerably different domains and lengths, ranging from 3-second generic domain GIF videos to 180-second YouTube human activity videos, showing the generalization ability of our approach. Comprehensive ablation studies and thorough analyses are provided to dissect what factors lead to this success. Our code is publicly available at https://github.com/jayleicn/ClipBERT

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Less is More: CLIPBERT for Video-and-Language Learning<br>via Sparse Sampling<br>pdf: <a href="https://t.co/ZTQtr2SYW9">https://t.co/ZTQtr2SYW9</a><br>abs: <a href="https://t.co/a9uO4aAZhS">https://t.co/a9uO4aAZhS</a><br>github: <a href="https://t.co/jZgu5Ut7Gt">https://t.co/jZgu5Ut7Gt</a> <a href="https://t.co/yMjHIbSkCk">pic.twitter.com/yMjHIbSkCk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360048161943941123?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Shelf-Supervised Mesh Prediction in the Wild

Yufei Ye, Shubham Tulsiani, Abhinav Gupta

- retweets: 81, favorites: 64 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.06195) | [pdf](https://arxiv.org/pdf/2102.06195)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We aim to infer 3D shape and pose of object from a single image and propose a learning-based approach that can train from unstructured image collections, supervised by only segmentation outputs from off-the-shelf recognition systems (i.e. 'shelf-supervised'). We first infer a volumetric representation in a canonical frame, along with the camera pose. We enforce the representation geometrically consistent with both appearance and masks, and also that the synthesized novel views are indistinguishable from image collections. The coarse volumetric prediction is then converted to a mesh-based representation, which is further refined in the predicted camera frame. These two steps allow both shape-pose factorization from image collections and per-instance reconstruction in finer details. We examine the method on both synthetic and real-world datasets and demonstrate its scalability on 50 categories in the wild, an order of magnitude more classes than existing works.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Shelf-Supervised Mesh Prediction in the Wild<br>pdf: <a href="https://t.co/0UdtFAHWgS">https://t.co/0UdtFAHWgS</a><br>abs: <a href="https://t.co/0eBlRqLf3x">https://t.co/0eBlRqLf3x</a><br>project page: <a href="https://t.co/xjxhLgGc4U">https://t.co/xjxhLgGc4U</a> <a href="https://t.co/BHlQW0tGYI">pic.twitter.com/BHlQW0tGYI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360083055289974785?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Multichannel-based learning for audio object extraction

Daniel Arteaga, Jordi Pons

- retweets: 90, favorites: 44 (02/13/2021 22:40:55)

- links: [abs](https://arxiv.org/abs/2102.06142) | [pdf](https://arxiv.org/pdf/2102.06142)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

The current paradigm for creating and deploying immersive audio content is based on audio objects, which are composed of an audio track and position metadata. While rendering an object-based production into a multichannel mix is straightforward, the reverse process involves sound source separation and estimating the spatial trajectories of the extracted sources. Besides, cinematic object-based productions are often composed by dozens of simultaneous audio objects, which poses a scalability challenge for audio object extraction. Here, we propose a novel deep learning approach to object extraction that learns from the multichannel renders of object-based productions, instead of directly learning from the audio objects themselves. This approach allows tackling the object scalability challenge and also offers the possibility to formulate the problem in a supervised or an unsupervised fashion. Since, to our knowledge, no other works have previously addressed this topic, we first define the task and propose an evaluation methodology, and then discuss under what circumstances our methods outperform the proposed baselines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We just released a new paper, together with <a href="https://twitter.com/jordiponsdotme?ref_src=twsrc%5Etfw">@jordiponsdotme</a>:<br><br>&quot;Multichannel-based learning for audio object extraction&quot;<br><br>Accepted for presentation in <a href="https://twitter.com/hashtag/ICASSP2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICASSP2021</a>.<a href="https://t.co/T8Dls40SAd">https://t.co/T8Dls40SAd</a> <a href="https://t.co/HswsGMaMjy">pic.twitter.com/HswsGMaMjy</a></p>&mdash; Daniel Arteaga (@dnlrtg) <a href="https://twitter.com/dnlrtg/status/1360251013765214214?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Deep Photo Scan: Semi-supervised learning for dealing with the  real-world degradation in smartphone photo scanning

Man M. Ho, Jinjia Zhou

- retweets: 86, favorites: 27 (02/13/2021 22:40:56)

- links: [abs](https://arxiv.org/abs/2102.06120) | [pdf](https://arxiv.org/pdf/2102.06120)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Physical photographs now can be conveniently scanned by smartphones and stored forever as a digital version, but the scanned photos are not restored well. One solution is to train a supervised deep neural network on many digital photos and the corresponding scanned photos. However, human annotation costs a huge resource leading to limited training data. Previous works create training pairs by simulating degradation using image processing techniques. Their synthetic images are formed with perfectly scanned photos in latent space. Even so, the real-world degradation in smartphone photo scanning remains unsolved since it is more complicated due to real lens defocus, lighting conditions, losing details via printing, various photo materials, and more. To solve these problems, we propose a Deep Photo Scan (DPScan) based on semi-supervised learning. First, we present the way to produce real-world degradation and provide the DIV2K-SCAN dataset for smartphone-scanned photo restoration. Second, by using DIV2K-SCAN, we adopt the concept of Generative Adversarial Networks to learn how to degrade a high-quality image as if it were scanned by a real smartphone, then generate pseudo-scanned photos for unscanned photos. Finally, we propose to train on the scanned and pseudo-scanned photos representing a semi-supervised approach with a cycle process as: high-quality images --> real-/pseudo-scanned photos --> reconstructed images. The proposed semi-supervised scheme can balance between supervised and unsupervised errors while optimizing to limit imperfect pseudo inputs but still enhance restoration. As a result, the proposed DPScan quantitatively and qualitatively outperforms its baseline architecture, state-of-the-art academic research, and industrial products in smartphone photo scanning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Photo Scan: Semi-supervised learning for dealing with the real-world degradation in smartphone photo scanning<br>pdf: <a href="https://t.co/Dl9e0WmWK2">https://t.co/Dl9e0WmWK2</a><br>abs: <a href="https://t.co/6mxRcGj1Yv">https://t.co/6mxRcGj1Yv</a> <a href="https://t.co/bZvRm7E5dM">pic.twitter.com/bZvRm7E5dM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1360087744198885379?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Private Prediction Sets

Anastasios N. Angelopoulos, Stephen Bates, Tijana Zrnic, Michael I. Jordan

- retweets: 72, favorites: 28 (02/13/2021 22:40:56)

- links: [abs](https://arxiv.org/abs/2102.06202) | [pdf](https://arxiv.org/pdf/2102.06202)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

In real-world settings involving consequential decision-making, the deployment of machine learning systems generally requires both reliable uncertainty quantification and protection of individuals' privacy. We present a framework that treats these two desiderata jointly. Our framework is based on conformal prediction, a methodology that augments predictive models to return prediction sets that provide uncertainty quantification -- they provably cover the true response with a user-specified probability, such as 90%. One might hope that when used with privately-trained models, conformal prediction would yield privacy guarantees for the resulting prediction sets; unfortunately this is not the case. To remedy this key problem, we develop a method that takes any pre-trained predictive model and outputs differentially private prediction sets. Our method follows the general approach of split conformal prediction; we use holdout data to calibrate the size of the prediction sets but preserve privacy by using a privatized quantile subroutine. This subroutine compensates for the noise introduced to preserve privacy in order to guarantee correct coverage. We evaluate the method with experiments on the CIFAR-10, ImageNet, and CoronaHack datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Today we released “Private Prediction Sets”, a differentially <a href="https://twitter.com/hashtag/private?src=hash&amp;ref_src=twsrc%5Etfw">#private</a> way to output rigorous, finite-sample uncertainty quantification for any model and dataset.<a href="https://t.co/kzucW6FZg9">https://t.co/kzucW6FZg9</a><br><br>The method builds on conformal prediction. 🧵1/n <a href="https://t.co/qIWDP540H8">pic.twitter.com/qIWDP540H8</a></p>&mdash; Anastasios Angelopoulos (@ml_angelopoulos) <a href="https://twitter.com/ml_angelopoulos/status/1360316919383945218?ref_src=twsrc%5Etfw">February 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Dot-Science Top Level Domain: academic websites or dumpsites?

Enrique Orduna-Malea

- retweets: 58, favorites: 11 (02/13/2021 22:40:56)

- links: [abs](https://arxiv.org/abs/2102.05706) | [pdf](https://arxiv.org/pdf/2102.05706)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

Dot-science was launched in 2015 as a new academic top-level domain (TLD) aimed to provide 'a dedicated, easily accessible location for global Internet users with an interest in science'. The main objective of this work is to find out the general scholarly usage of this top-level domain. In particular, the following three questions are pursued: usage (number of web domains registered with the dot-science), purpose (main function and category of websites linked to these web domains), and impact (websites' visibility and authority). To do this, 13,900 domain names were gathered through ICANN's Domain Name Registration Data Lookup database. Each web domain was subsequently categorized, and data on web impact were obtained from Majestic's API. Based on the results obtained, it is concluded that the dot-science top-level domain is scarcely adopted by the academic community, and mainly used by registrar companies for reselling purposes (35.5% of all web domains were parked). Websites receiving the highest number of backlinks were generally related to non-academic websites applying intensive link building practices and offering leisure or even fraudulent contents. Majestic's Trust Flow metric has been proved an effective method to filter reputable academic websites. As regards primary academic-related dot-science web domain categories, 1,175 (8.5% of all web domains registered) were found, mainly personal academic websites (342 web domains), blogs (261) and research groups (133). All dubious content reveals bad practices on the Web, where the tag 'science' is fundamentally used as a mechanism to deceive search engine algorithms.



