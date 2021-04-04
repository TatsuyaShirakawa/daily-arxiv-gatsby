---
title: Hot Papers 2021-04-02
date: 2021-04-04T11:47:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-02"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-02"
socialImage: "/media/flying-marine.jpg"

---

# 1. EfficientNetV2: Smaller Models and Faster Training

Mingxing Tan, Quoc V. Le

- retweets: 10199, favorites: 64 (04/04/2021 11:47:36)

- links: [abs](https://arxiv.org/abs/2104.00298) | [pdf](https://arxiv.org/pdf/2104.00298)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller.   Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy.   With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. Code will be available at https://github.com/google/automl/efficientnetv2.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to introduce EfficientNetV2: Smaller Models and Faster Training<br><br>Achieved faster training and inference speed, AND also with better parameters efficiency.<br><br>Arxiv: <a href="https://t.co/YHWEb8pHmR">https://t.co/YHWEb8pHmR</a><br><br>Thread 1/4 <a href="https://t.co/LY2oZ4tSbN">pic.twitter.com/LY2oZ4tSbN</a></p>&mdash; Mingxing Tan (@tanmingxing) <a href="https://twitter.com/tanmingxing/status/1377807813049679873?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. NeRF-VAE: A Geometry Aware 3D Scene Generative Model

Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Soňa Mokrá, Danilo J. Rezende

- retweets: 5335, favorites: 49 (04/04/2021 11:47:36)

- links: [abs](https://arxiv.org/abs/2104.00587) | [pdf](https://arxiv.org/pdf/2104.00587)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose NeRF-VAE, a 3D scene generative model that incorporates geometric structure via NeRF and differentiable volume rendering. In contrast to NeRF, our model takes into account shared structure across scenes, and is able to infer the structure of a novel scene -- without the need to re-train -- using amortized inference. NeRF-VAE's explicit 3D rendering process further contrasts previous generative models with convolution-based rendering which lacks geometric structure. Our model is a VAE that learns a distribution over radiance fields by conditioning them on a latent scene representation. We show that, once trained, NeRF-VAE is able to infer and render geometrically-consistent scenes from previously unseen 3D environments using very few input images. We further demonstrate that NeRF-VAE generalizes well to out-of-distribution cameras, while convolutional models do not. Finally, we introduce and study an attention-based conditioning mechanism of NeRF-VAE's decoder, which improves model performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeRF-VAE: A Geometry Aware 3D Scene Generative Model<br>pdf: <a href="https://t.co/943n2Sspab">https://t.co/943n2Sspab</a><br>abs: <a href="https://t.co/CYlYRFvOkU">https://t.co/CYlYRFvOkU</a><br><br>&quot;NeRF-VAE is able to infer and render geometrically-consistent scenes from previously unseen 3D environments using very few input images&quot; <a href="https://t.co/MxlAwmbCu1">pic.twitter.com/MxlAwmbCu1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377790225590673410?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. In&Out : Diverse Image Outpainting via GAN Inversion

Yen-Chi Cheng, Chieh Hubert Lin, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Ming-Hsuan Yang

- retweets: 2209, favorites: 269 (04/04/2021 11:47:36)

- links: [abs](https://arxiv.org/abs/2104.00675) | [pdf](https://arxiv.org/pdf/2104.00675)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image outpainting seeks for a semantically consistent extension of the input image beyond its available content. Compared to inpainting -- filling in missing pixels in a way coherent with the neighboring pixels -- outpainting can be achieved in more diverse ways since the problem is less constrained by the surrounding pixels. Existing image outpainting methods pose the problem as a conditional image-to-image translation task, often generating repetitive structures and textures by replicating the content available in the input image. In this work, we formulate the problem from the perspective of inverting generative adversarial networks. Our generator renders micro-patches conditioned on their joint latent code as well as their individual positions in the image. To outpaint an image, we seek for multiple latent codes not only recovering available patches but also synthesizing diverse outpainting by patch-based generation. This leads to richer structure and content in the outpainted regions. Furthermore, our formulation allows for outpainting conditioned on the categorical input, thereby enabling flexible user controls. Extensive experimental results demonstrate the proposed method performs favorably against existing in- and outpainting methods, featuring higher visual quality and diversity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wanna generate panorama from images you took during vacation? Check out our recent paper &quot;𝐈𝐧&amp;𝐎𝐮𝐭: 𝐃𝐢𝐯𝐞𝐫𝐬𝐞 𝐈𝐦𝐚𝐠𝐞 𝐎𝐮𝐭𝐩𝐚𝐢𝐧𝐭𝐢𝐧𝐠 𝐯𝐢𝐚 𝐆𝐀𝐍 𝐈𝐧𝐯𝐞𝐫𝐬𝐢𝐨𝐧&quot;!<br>Project: <a href="https://t.co/jQXut5xjAp">https://t.co/jQXut5xjAp</a><br>Paper: <a href="https://t.co/BLlMoJ4V2H">https://t.co/BLlMoJ4V2H</a><a href="https://twitter.com/hashtag/snap?src=hash&amp;ref_src=twsrc%5Etfw">#snap</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a><br>(1/2) <a href="https://t.co/dyEFX836Np">pic.twitter.com/dyEFX836Np</a></p>&mdash; Hsin-Ying James Lee (@hyjameslee) <a href="https://twitter.com/hyjameslee/status/1378019214921932801?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In&amp;Out : Diverse Image Outpainting via GAN Inversion<br>pdf: <a href="https://t.co/wCPVXSu1Pw">https://t.co/wCPVXSu1Pw</a><br>abs: <a href="https://t.co/yvFnkVJpiQ">https://t.co/yvFnkVJpiQ</a><br>project page: <a href="https://t.co/fFTrsY6no8">https://t.co/fFTrsY6no8</a> <a href="https://t.co/JP8Lu2K0qe">pic.twitter.com/JP8Lu2K0qe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377794992148713481?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. LoFTR: Detector-Free Local Feature Matching with Transformers

Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, Xiaowei Zhou

- retweets: 1968, favorites: 284 (04/04/2021 11:47:37)

- links: [abs](https://arxiv.org/abs/2104.00680) | [pdf](https://arxiv.org/pdf/2104.00680)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present a novel method for local image feature matching. Instead of performing image feature detection, description, and matching sequentially, we propose to first establish pixel-wise dense matches at a coarse level and later refine the good matches at a fine level. In contrast to dense methods that use a cost volume to search correspondences, we use self and cross attention layers in Transformer to obtain feature descriptors that are conditioned on both images. The global receptive field provided by Transformer enables our method to produce dense matches in low-texture areas, where feature detectors usually struggle to produce repeatable interest points. The experiments on indoor and outdoor datasets show that LoFTR outperforms state-of-the-art methods by a large margin. LoFTR also ranks first on two public benchmarks of visual localization among the published methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LoFTR: Detector-Free Local Feature Matching with Transformers<br>pdf: <a href="https://t.co/AFNPRCJKs6">https://t.co/AFNPRCJKs6</a><br>abs: <a href="https://t.co/GgQGINeKpE">https://t.co/GgQGINeKpE</a><br>project page: <a href="https://t.co/gwmw9BbE6g">https://t.co/gwmw9BbE6g</a> <a href="https://t.co/kWiY5IQ0QD">pic.twitter.com/kWiY5IQ0QD</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377789699566223367?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">LoFTR: Detector-Free Local Feature Matching with Transformers<a href="https://twitter.com/JiamingSuen?ref_src=twsrc%5Etfw">@JiamingSuen</a>, Zehong Shen, Yuang Wang, Hujun Bao, Xiaowei Zhou<br><br>tl;dr: dense local descriptor -&gt; linear transformer  matcher (similar to SuperGlue). Everything is in the coarse-to-fine scheme.<a href="https://t.co/O8ze2SdTUb">https://t.co/O8ze2SdTUb</a> <a href="https://t.co/yVNZv3ibzq">pic.twitter.com/yVNZv3ibzq</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1378321874095112195?ref_src=twsrc%5Etfw">April 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval

Max Bain, Arsha Nagrani, Gül Varol, Andrew Zisserman

- retweets: 1426, favorites: 170 (04/04/2021 11:47:37)

- links: [abs](https://arxiv.org/abs/2104.00650) | [pdf](https://arxiv.org/pdf/2104.00650)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Our objective in this work is video-text retrieval - in particular a joint embedding that enables efficient text-to-video retrieval. The challenges in this area include the design of the visual architecture and the nature of the training data, in that the available large scale video-text training datasets, such as HowTo100M, are noisy and hence competitive performance is achieved only at scale through large amounts of compute. We address both these challenges in this paper. We propose an end-to-end trainable model that is designed to take advantage of both large-scale image and video captioning datasets. Our model is an adaptation and extension of the recent ViT and Timesformer architectures, and consists of attention in both space and time. The model is flexible and can be trained on both image and video text datasets, either independently or in conjunction. It is trained with a curriculum learning schedule that begins by treating images as 'frozen' snapshots of video, and then gradually learns to attend to increasing temporal context when trained on video datasets. We also provide a new video-text pretraining dataset WebVid-2M, comprised of over two million videos with weak captions scraped from the internet. Despite training on datasets that are an order of magnitude smaller, we show that this approach yields state-of-the-art results on standard downstream video-retrieval benchmarks including MSR-VTT, MSVD, DiDeMo and LSMDC.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Separation of effort in the image and video retrieval communities is suboptimal - they share a lot of overlapping info! <br>Check out our NEW model for visual-text retrieval, easily trains on *both*  images and videos jointly, setting new SOTA results! <a href="https://t.co/riCQF69To1">https://t.co/riCQF69To1</a> <a href="https://t.co/HpORGtLMUs">pic.twitter.com/HpORGtLMUs</a></p>&mdash; Arsha Nagrani @🏠 (@NagraniArsha) <a href="https://twitter.com/NagraniArsha/status/1378049400375803909?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video

Jiaming Sun, Yiming Xie, Linghao Chen, Xiaowei Zhou, Hujun Bao

- retweets: 1296, favorites: 186 (04/04/2021 11:47:38)

- links: [abs](https://arxiv.org/abs/2104.00681) | [pdf](https://arxiv.org/pdf/2104.00681)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present a novel framework named NeuralRecon for real-time 3D scene reconstruction from a monocular video. Unlike previous methods that estimate single-view depth maps separately on each key-frame and fuse them later, we propose to directly reconstruct local surfaces represented as sparse TSDF volumes for each video fragment sequentially by a neural network. A learning-based TSDF fusion module based on gated recurrent units is used to guide the network to fuse features from previous fragments. This design allows the network to capture local smoothness prior and global shape prior of 3D surfaces when sequentially reconstructing the surfaces, resulting in accurate, coherent, and real-time surface reconstruction. The experiments on ScanNet and 7-Scenes datasets show that our system outperforms state-of-the-art methods in terms of both accuracy and speed. To the best of our knowledge, this is the first learning-based system that is able to reconstruct dense coherent 3D geometry in real-time.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video<br>pdf: <a href="https://t.co/KTjroX1B1s">https://t.co/KTjroX1B1s</a><br>abs: <a href="https://t.co/ahbtVraPkB">https://t.co/ahbtVraPkB</a><br>project page: <a href="https://t.co/w5K5HgFKHt">https://t.co/w5K5HgFKHt</a> <a href="https://t.co/0X48Rj6wE0">pic.twitter.com/0X48Rj6wE0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377810608964448258?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Unconstrained Scene Generation with Locally Conditioned Radiance Fields

Terrance DeVries, Miguel Angel Bautista, Nitish Srivastava, Graham W. Taylor, Joshua M. Susskind

- retweets: 1167, favorites: 217 (04/04/2021 11:47:38)

- links: [abs](https://arxiv.org/abs/2104.00670) | [pdf](https://arxiv.org/pdf/2104.00670)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We tackle the challenge of learning a distribution over complex, realistic, indoor scenes. In this paper, we introduce Generative Scene Networks (GSN), which learns to decompose scenes into a collection of many local radiance fields that can be rendered from a free moving camera. Our model can be used as a prior to generate new scenes, or to complete a scene given only sparse 2D observations. Recent work has shown that generative models of radiance fields can capture properties such as multi-view consistency and view-dependent lighting. However, these models are specialized for constrained viewing of single objects, such as cars or faces. Due to the size and complexity of realistic indoor environments, existing models lack the representational capacity to adequately capture them. Our decomposition scheme scales to larger and more complex scenes while preserving details and diversity, and the learned prior enables high-quality rendering from viewpoints that are significantly different from observed viewpoints. When compared to existing models, GSN produces quantitatively higher-quality scene renderings across several different scene datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing Generative Scene Networks (GSN), a generative model for learning radiance fields for realistic scenes. With GSN we can sample scenes from the learned prior and move through them with a freely moving camera.<br>Arxiv: <a href="https://t.co/rYiFH4uhLp">https://t.co/rYiFH4uhLp</a><br>Scenes sampled from the prior: <a href="https://t.co/aFxnve3PEd">https://t.co/aFxnve3PEd</a> <a href="https://t.co/huUT9z8a1t">pic.twitter.com/huUT9z8a1t</a></p>&mdash; Miguel A Bautista (@itsbautistam) <a href="https://twitter.com/itsbautistam/status/1378137986424967172?ref_src=twsrc%5Etfw">April 3, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis

Ajay Jain, Matthew Tancik, Pieter Abbeel

- retweets: 674, favorites: 141 (04/04/2021 11:47:38)

- links: [abs](https://arxiv.org/abs/2104.00677) | [pdf](https://arxiv.org/pdf/2104.00677)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present DietNeRF, a 3D neural scene representation estimated from a few images. Neural Radiance Fields (NeRF) learn a continuous volumetric representation of a scene through multi-view consistency, and can be rendered from novel viewpoints by ray casting. While NeRF has an impressive ability to reconstruct geometry and fine details given many images, up to 100 for challenging 360{\deg} scenes, it often finds a degenerate solution to its image reconstruction objective when only a few input views are available. To improve few-shot quality, we propose DietNeRF. We introduce an auxiliary semantic consistency loss that encourages realistic renderings at novel poses. DietNeRF is trained on individual scenes to (1) correctly render given input views from the same pose, and (2) match high-level semantic attributes across different, random poses. Our semantic loss allows us to supervise DietNeRF from arbitrary poses. We extract these semantics using a pre-trained visual encoder such as CLIP, a Vision Transformer trained on hundreds of millions of diverse single-view, 2D photographs mined from the web with natural language supervision. In experiments, DietNeRF improves the perceptual quality of few-shot view synthesis when learned from scratch, can render novel views with as few as one observed image when pre-trained on a multi-view dataset, and produces plausible completions of completely unobserved regions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis<br>pdf: <a href="https://t.co/DKAXVYozKR">https://t.co/DKAXVYozKR</a><br>abs: <a href="https://t.co/FcfuYJsX4H">https://t.co/FcfuYJsX4H</a><br>project page: <a href="https://t.co/8OAhjQMQ90">https://t.co/8OAhjQMQ90</a> <a href="https://t.co/TPKkEqeX1V">pic.twitter.com/TPKkEqeX1V</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377786960824127488?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Reconstructing 3D Human Pose by Watching Humans in the Mirror

Qi Fang, Qing Shuai, Junting Dong, Hujun Bao, Xiaowei Zhou

- retweets: 702, favorites: 104 (04/04/2021 11:47:39)

- links: [abs](https://arxiv.org/abs/2104.00340) | [pdf](https://arxiv.org/pdf/2104.00340)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we introduce the new task of reconstructing 3D human pose from a single image in which we can see the person and the person's image through a mirror. Compared to general scenarios of 3D pose estimation from a single view, the mirror reflection provides an additional view for resolving the depth ambiguity. We develop an optimization-based approach that exploits mirror symmetry constraints for accurate 3D pose reconstruction. We also provide a method to estimate the surface normal of the mirror from vanishing points in the single image. To validate the proposed approach, we collect a large-scale dataset named Mirrored-Human, which covers a large variety of human subjects, poses and backgrounds. The experiments demonstrate that, when trained on Mirrored-Human with our reconstructed 3D poses as pseudo ground-truth, the accuracy and generalizability of existing single-view 3D pose estimators can be largely improved.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reconstructing 3D Human Pose by Watching Humans in the Mirror<br>pdf: <a href="https://t.co/4AoRvYDIfP">https://t.co/4AoRvYDIfP</a><br>abs: <a href="https://t.co/L10OpNaFQE">https://t.co/L10OpNaFQE</a><br>project page: <a href="https://t.co/0NvY0M2XrD">https://t.co/0NvY0M2XrD</a> <a href="https://t.co/He65Yw5QzV">pic.twitter.com/He65Yw5QzV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377808494225723396?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Real-time Data Infrastructure at Uber

Yupeng Fu, Chinmay Soman

- retweets: 318, favorites: 97 (04/04/2021 11:47:39)

- links: [abs](https://arxiv.org/abs/2104.00087) | [pdf](https://arxiv.org/pdf/2104.00087)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent)

Uber's business is highly real-time in nature. PBs of data is continuously being collected from the end users such as Uber drivers, riders, restaurants, eaters and so on everyday. There is a lot of valuable information to be processed and many decisions must be made in seconds for a variety of use cases such as customer incentives, fraud detection, machine learning model prediction. In addition, there is an increasing need to expose this ability to different user categories, including engineers, data scientists, executives and operations personnel which adds to the complexity. In this paper, we present the overall architecture of the real-time data infrastructure and identify three scaling challenges that we need to continuously address for each component in the architecture. At Uber, we heavily rely on open source technologies for the key areas of the infrastructure. On top of those open-source software, we add significant improvements and customizations to make the open-source solutions fit in Uber's environment and bridge the gaps to meet Uber's unique scale and requirements. We then highlight several important use cases and show their real-time solutions and tradeoffs. Finally, we reflect on the lessons we learned as we built, operated and scaled these systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This pretty much summarizes my Uber tenure - coauthored with Yupeng Fu (SIGMOD21): <a href="https://t.co/xAVW0U3PdU">https://t.co/xAVW0U3PdU</a> <br>Very grateful for <a href="https://twitter.com/ApachePinot?ref_src=twsrc%5Etfw">@ApachePinot</a> <a href="https://twitter.com/apachekafka?ref_src=twsrc%5Etfw">@apachekafka</a> and <a href="https://twitter.com/ApacheFlink?ref_src=twsrc%5Etfw">@ApacheFlink</a> for providing a robust foundation to power Uber&#39;s real time analytics.</p>&mdash; Chinmay Soman (@ChinmaySoman) <a href="https://twitter.com/ChinmaySoman/status/1378044223082000389?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ar" dir="rtl"><a href="https://twitter.com/hashtag/%D8%AD%D8%A7%D8%B3%D9%88%D8%A8%D9%8A%D8%A7%D8%AA?src=hash&amp;ref_src=twsrc%5Etfw">#حاسوبيات</a><br>بحث سينشر من شركة<br>Uber<br>في احد اكبر مؤتمرات قواعد البيانات<br>تتحدث عن البنية التحتية للتعامل مع البيانات اللحظية<br>Real-time Data<a href="https://t.co/jrvvYNUp1E">https://t.co/jrvvYNUp1E</a></p>&mdash; Saif AlHarthi (@SaifAlHarthi) <a href="https://twitter.com/SaifAlHarthi/status/1377959898076569604?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Vertex Connectivity in Poly-logarithmic Max-flows

Jason Li, Danupon Nanongkai, Debmalya Panigrahi, Thatchaphol Saranurak, Sorrachai Yingchareonthawornchai

- retweets: 144, favorites: 126 (04/04/2021 11:47:39)

- links: [abs](https://arxiv.org/abs/2104.00104) | [pdf](https://arxiv.org/pdf/2104.00104)
- [cs.DS](https://arxiv.org/list/cs.DS/recent)

The vertex connectivity of an $m$-edge $n$-vertex undirected graph is the smallest number of vertices whose removal disconnects the graph, or leaves only a singleton vertex. In this paper, we give a reduction from the vertex connectivity problem to a set of maxflow instances. Using this reduction, we can solve vertex connectivity in $\tilde O(m^{\alpha})$ time for any $\alpha \ge 1$, if there is a $m^{\alpha}$-time maxflow algorithm. Using the current best maxflow algorithm that runs in $m^{4/3+o(1)}$ time (Kathuria, Liu and Sidford, FOCS 2020), this yields a $m^{4/3+o(1)}$-time vertex connectivity algorithm. This is the first improvement in the running time of the vertex connectivity problem in over 20 years, the previous best being an $\tilde O(mn)$-time algorithm due to Henzinger, Rao, and Gabow (FOCS 1996). Indeed, no algorithm with an $o(mn)$ running time was known before our work, even if we assume an $\tilde O(m)$-time maxflow algorithm. Our new technique is robust enough to also improve the best $\tilde O(mn)$-time bound for directed vertex connectivity to $mn^{1-1/12+o(1)}$ time

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How many vertices do you need to delete to disconnect an n-vertex graph?<br>  <br>Previous algorithms roughly call n max flows (or some optimized version of it) to compute this number.<br><br>Our new algorithm takes time proportional to polylog(n) max flows only!<a href="https://t.co/mNXHWfIN0a">https://t.co/mNXHWfIN0a</a></p>&mdash; Thatchaphol Saranurak (@eig) <a href="https://twitter.com/eig/status/1378120254459670529?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Why is AI hard and Physics simple?

Daniel A. Roberts

- retweets: 129, favorites: 85 (04/04/2021 11:47:39)

- links: [abs](https://arxiv.org/abs/2104.00008) | [pdf](https://arxiv.org/pdf/2104.00008)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [hep-th](https://arxiv.org/list/hep-th/recent) | [physics.hist-ph](https://arxiv.org/list/physics.hist-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We discuss why AI is hard and why physics is simple. We discuss how physical intuition and the approach of theoretical physics can be brought to bear on the field of artificial intelligence and specifically machine learning. We suggest that the underlying project of machine learning and the underlying project of physics are strongly coupled through the principle of sparsity, and we call upon theoretical physicists to work on AI as physicists. As a first step in that direction, we discuss an upcoming book on the principles of deep learning theory that attempts to realize this approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New essay &quot;Why is AI hard and Physics simple?&quot; on how the tools and language of theoretical physics might be useful for making progress in AI, and specifically for deep learning. Basically, an apologia for physicists working on ML. <a href="https://t.co/IATPFBn63h">https://t.co/IATPFBn63h</a><br><br>1/</p>&mdash; Dan Roberts (@danintheory) <a href="https://twitter.com/danintheory/status/1378020927200432131?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. PhySG: Inverse Rendering with Spherical Gaussians for Physics-based  Material Editing and Relighting

Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, Noah Snavely

- retweets: 100, favorites: 66 (04/04/2021 11:47:39)

- links: [abs](https://arxiv.org/abs/2104.00674) | [pdf](https://arxiv.org/pdf/2104.00674)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present PhySG, an end-to-end inverse rendering pipeline that includes a fully differentiable renderer and can reconstruct geometry, materials, and illumination from scratch from a set of RGB input images. Our framework represents specular BRDFs and environmental illumination using mixtures of spherical Gaussians, and represents geometry as a signed distance function parameterized as a Multi-Layer Perceptron. The use of spherical Gaussians allows us to efficiently solve for approximate light transport, and our method works on scenes with challenging non-Lambertian reflectance captured under natural, static illumination. We demonstrate, with both synthetic and real data, that our reconstructions not only enable rendering of novel viewpoints, but also physics-based appearance editing of materials and illumination.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PhySG: Inverse Rendering with Spherical Gaussians for Physics-based Material Editing and Relighting<br>pdf: <a href="https://t.co/nIKgSFadir">https://t.co/nIKgSFadir</a><br>abs: <a href="https://t.co/QFMoNo8lVr">https://t.co/QFMoNo8lVr</a><br>project page: <a href="https://t.co/YFvFZPCXN6">https://t.co/YFvFZPCXN6</a> <a href="https://t.co/JRu4SQi3Gk">pic.twitter.com/JRu4SQi3Gk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377800985100349443?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Speech Resynthesis from Discrete Disentangled Self-Supervised  Representations

Adam Polyak, Yossi Adi, Jade Copet, Eugene Kharitonov, Kushal Lakhotia, Wei-Ning Hsu, Abdelrahman Mohamed, Emmanuel Dupoux

- retweets: 72, favorites: 46 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00355) | [pdf](https://arxiv.org/pdf/2104.00355)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We propose using self-supervised discrete representations for the task of speech resynthesis. To generate disentangled representation, we separately extract low-bitrate representations for speech content, prosodic information, and speaker identity. This allows to synthesize speech in a controllable manner. We analyze various state-of-the-art, self-supervised representation learning methods and shed light on the advantages of each method while considering reconstruction quality and disentanglement properties. Specifically, we evaluate the F0 reconstruction, speaker identification performance (for both resynthesis and voice conversion), recordings' intelligibility, and overall quality using subjective human evaluation. Lastly, we demonstrate how these representations can be used for an ultra-lightweight speech codec. Using the obtained representations, we can get to a rate of 365 bits per second while providing better speech quality than the baseline methods. Audio samples can be found under the following link: \url{https://resynthesis-ssl.github.io/}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Speech Resynthesis from Discrete Disentangled Self-Supervised Representations<br>pdf: <a href="https://t.co/6uQG1W2cIX">https://t.co/6uQG1W2cIX</a><br>abs: <a href="https://t.co/kp622FKD7M">https://t.co/kp622FKD7M</a> <a href="https://t.co/OL654AlfF2">pic.twitter.com/OL654AlfF2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377797376790904833?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Sketch2Mesh: Reconstructing and Editing 3D Shapes from Sketches

Benoit Guillard, Edoardo Remelli, Pierre Yvernay, Pascal Fua

- retweets: 42, favorites: 40 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00482) | [pdf](https://arxiv.org/pdf/2104.00482)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Reconstructing 3D shape from 2D sketches has long been an open problem because the sketches only provide very sparse and ambiguous information. In this paper, we use an encoder/decoder architecture for the sketch to mesh translation. This enables us to leverage its latent parametrization to represent and refine a 3D mesh so that its projections match the external contours outlined in the sketch. We will show that this approach is easy to deploy, robust to style changes, and effective. Furthermore, it can be used for shape refinement given only single pen strokes. We compare our approach to state-of-the-art methods on sketches -- both hand-drawn and synthesized -- and demonstrate that we outperform them.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sketch2Mesh: Reconstructing and Editing 3D Shapes from Sketches<br>pdf: <a href="https://t.co/ABMuJtY6Vy">https://t.co/ABMuJtY6Vy</a><br>abs: <a href="https://t.co/atejyzpbLg">https://t.co/atejyzpbLg</a> <a href="https://t.co/Vf365HDTh1">pic.twitter.com/Vf365HDTh1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377821168967778308?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Group-Free 3D Object Detection via Transformers

Ze Liu, Zheng Zhang, Yue Cao, Han Hu, Xin Tong

- retweets: 42, favorites: 32 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00678) | [pdf](https://arxiv.org/pdf/2104.00678)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, directly detecting 3D objects from 3D point clouds has received increasing attention. To extract object representation from an irregular point cloud, existing methods usually take a point grouping step to assign the points to an object candidate so that a PointNet-like network could be used to derive object features from the grouped points. However, the inaccurate point assignments caused by the hand-crafted grouping scheme decrease the performance of 3D object detection.   In this paper, we present a simple yet effective method for directly detecting 3D objects from the 3D point cloud. Instead of grouping local points to each object candidate, our method computes the feature of an object from all the points in the point cloud with the help of an attention mechanism in the Transformers \cite{vaswani2017attention}, where the contribution of each point is automatically learned in the network training. With an improved attention stacking scheme, our method fuses object features in different stages and generates more accurate object detection results. With few bells and whistles, the proposed method achieves state-of-the-art 3D object detection performance on two widely used benchmarks, ScanNet V2 and SUN RGB-D. The code and models are publicly available at \url{https://github.com/zeliu98/Group-Free-3D}

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Group-Free 3D Object Detection via Transformers<br>pdf: <a href="https://t.co/Liq6WX16cC">https://t.co/Liq6WX16cC</a><br>abs: <a href="https://t.co/ThBRk3rPcq">https://t.co/ThBRk3rPcq</a><br>github: <a href="https://t.co/YiEihL55nk">https://t.co/YiEihL55nk</a><br><br>&quot; the proposed method achieves state-of-the-art 3D object detection performance on two widely used benchmarks, ScanNet V2 and SUN RGB-D.&quot; <a href="https://t.co/AodWkXb31a">pic.twitter.com/AodWkXb31a</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377787611817897993?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. LIFT-SLAM: a deep-learning feature-based monocular visual SLAM method

Hudson M. S. Bruno, Esther L. Colombini

- retweets: 53, favorites: 13 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00099) | [pdf](https://arxiv.org/pdf/2104.00099)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

The Simultaneous Localization and Mapping (SLAM) problem addresses the possibility of a robot to localize itself in an unknown environment and simultaneously build a consistent map of this environment. Recently, cameras have been successfully used to get the environment's features to perform SLAM, which is referred to as visual SLAM (VSLAM). However, classical VSLAM algorithms can be easily induced to fail when either the motion of the robot or the environment is too challenging. Although new approaches based on Deep Neural Networks (DNNs) have achieved promising results in VSLAM, they still are unable to outperform traditional methods. To leverage the robustness of deep learning to enhance traditional VSLAM systems, we propose to combine the potential of deep learning-based feature descriptors with the traditional geometry-based VSLAM, building a new VSLAM system called LIFT-SLAM. Experiments conducted on KITTI and Euroc datasets show that deep learning can be used to improve the performance of traditional VSLAM systems, as the proposed approach was able to achieve results comparable to the state-of-the-art while being robust to sensorial noise. We enhance the proposed VSLAM pipeline by avoiding parameter tuning for specific datasets with an adaptive approach while evaluating how transfer learning can affect the quality of the features extracted.




# 18. Mining Wikidata for Name Resources for African Languages

Jonne Sälevä, Constantine Lignos

- retweets: 46, favorites: 19 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00558) | [pdf](https://arxiv.org/pdf/2104.00558)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

This work supports further development of language technology for the languages of Africa by providing a Wikidata-derived resource of name lists corresponding to common entity types (person, location, and organization). While we are not the first to mine Wikidata for name lists, our approach emphasizes scalability and replicability and addresses data quality issues for languages that do not use Latin scripts. We produce lists containing approximately 1.9 million names across 28 African languages. We describe the data, the process used to produce it, and its limitations, and provide the software and data for public use. Finally, we discuss the ethical considerations of producing this resource and others of its kind.




# 19. Text to Image Generation with Semantic-Spatial Aware GAN

Wentong Liao, Kai Hu, Michael Ying Yang, Bodo Rosenhahn

- retweets: 12, favorites: 38 (04/04/2021 11:47:40)

- links: [abs](https://arxiv.org/abs/2104.00567) | [pdf](https://arxiv.org/pdf/2104.00567)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A text to image generation (T2I) model aims to generate photo-realistic images which are semantically consistent with the text descriptions. Built upon the recent advances in generative adversarial networks (GANs), existing T2I models have made great progress. However, a close inspection of their generated images reveals two major limitations: (1) The condition batch normalization methods are applied on the whole image feature maps equally, ignoring the local semantics; (2) The text encoder is fixed during training, which should be trained with the image generator jointly to learn better text representations for image generation. To address these limitations, we propose a novel framework Semantic-Spatial Aware GAN, which is trained in an end-to-end fashion so that the text encoder can exploit better text information. Concretely, we introduce a novel Semantic-Spatial Aware Convolution Network, which (1) learns semantic-adaptive transformation conditioned on text to effectively fuse text features and image features, and (2) learns a mask map in a weakly-supervised way that depends on the current text-image fusion process in order to guide the transformation spatially. Experiments on the challenging COCO and CUB bird datasets demonstrate the advantage of our method over the recent state-of-the-art approaches, regarding both visual fidelity and alignment with input text description.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Text to Image Generation with Semantic-Spatial Aware GAN<br>pdf: <a href="https://t.co/A6Yq8P8qtn">https://t.co/A6Yq8P8qtn</a><br>abs: <a href="https://t.co/F3ZhzibmEO">https://t.co/F3ZhzibmEO</a> <a href="https://t.co/wtOF9WxUrp">pic.twitter.com/wtOF9WxUrp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1377793968927346693?ref_src=twsrc%5Etfw">April 2, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



