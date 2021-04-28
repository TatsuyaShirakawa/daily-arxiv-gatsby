---
title: Hot Papers 2021-04-27
date: 2021-04-28T09:29:54.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-27"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-27"
socialImage: "/media/flying-marine.jpg"

---

# 1. Deep Probabilistic Graphical Modeling

Adji B. Dieng

- retweets: 7650, favorites: 904 (04/28/2021 09:29:54)

- links: [abs](https://arxiv.org/abs/2104.12053) | [pdf](https://arxiv.org/pdf/2104.12053)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

As one of the most commonly ordered imaging tests, computed tomography (CT) scan comes with inevitable radiation exposure that increases the cancer risk to patients. However, CT image quality is directly related to radiation dose, thus it is desirable to obtain high-quality CT images with as little dose as possible. CT image denoising tries to obtain high dose like high-quality CT images (domain X) from low dose low-quality CTimages (domain Y), which can be treated as an image-to-image translation task where the goal is to learn the transform between a source domain X (noisy images) and a target domain Y (clean images). In this paper, we propose a multi-cycle-consistent adversarial network (MCCAN) that builds intermediate domains and enforces both local and global cycle-consistency for edge denoising of CT images. The global cycle-consistency couples all generators together to model the whole denoising process, while the local cycle-consistency imposes effective supervision on the process between adjacent domains. Experiments show that both local and global cycle-consistency are important for the success of MCCAN, which outperformsCCADN in terms of denoising quality with slightly less computation resource consumption.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I finally found time to put my PhD thesis on arxiv. Check it out! Thanking everyone who made this thesis possible 🖤🎊<a href="https://t.co/6AAp04SJOr">https://t.co/6AAp04SJOr</a> <a href="https://t.co/EHTqEJADVP">pic.twitter.com/EHTqEJADVP</a></p>&mdash; Adji Bousso Dieng (@adjiboussodieng) <a href="https://twitter.com/adjiboussodieng/status/1386848248443310081?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. EigenGAN: Layer-Wise Eigen-Learning for GANs

Zhenliang He, Meina Kan, Shiguang Shan

- retweets: 1845, favorites: 174 (04/28/2021 09:29:54)

- links: [abs](https://arxiv.org/abs/2104.12476) | [pdf](https://arxiv.org/pdf/2104.12476)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recent studies on Generative Adversarial Network (GAN) reveal that different layers of a generative CNN hold different semantics of the synthesized images. However, few GAN models have explicit dimensions to control the semantic attributes represented in a specific layer. This paper proposes EigenGAN which is able to unsupervisedly mine interpretable and controllable dimensions from different generator layers. Specifically, EigenGAN embeds one linear subspace with orthogonal basis into each generator layer. Via the adversarial training to learn a target distribution, these layer-wise subspaces automatically discover a set of "eigen-dimensions" at each layer corresponding to a set of semantic attributes or interpretable variations. By traversing the coefficient of a specific eigen-dimension, the generator can produce samples with continuous changes corresponding to a specific semantic attribute. Taking the human face for example, EigenGAN can discover controllable dimensions for high-level concepts such as pose and gender in the subspace of deep layers, as well as low-level concepts such as hue and color in the subspace of shallow layers. Moreover, under the linear circumstance, we theoretically prove that our algorithm derives the principal components as PCA does. Codes can be found in https://github.com/LynnHo/EigenGAN-Tensorflow.

<blockquote class="twitter-tweet"><p lang="de" dir="ltr">EigenGAN: Layer-Wise Eigen-Learning for GANs<br>pdf: <a href="https://t.co/gViyi29TCb">https://t.co/gViyi29TCb</a><br>abs: <a href="https://t.co/67Sdgh1Zrz">https://t.co/67Sdgh1Zrz</a><br>github: <a href="https://t.co/7IvwikkjEi">https://t.co/7IvwikkjEi</a> <a href="https://t.co/fgnN3bW86A">pic.twitter.com/fgnN3bW86A</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386902104170708995?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding

Aishwarya Kamath, Mannat Singh, Yann LeCun, Ishan Misra, Gabriel Synnaeve, Nicolas Carion

- retweets: 1520, favorites: 151 (04/28/2021 09:29:54)

- links: [abs](https://arxiv.org/abs/2104.12763) | [pdf](https://arxiv.org/pdf/2104.12763)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Multi-modal reasoning systems rely on a pre-trained object detector to extract regions of interest from the image. However, this crucial module is typically used as a black box, trained independently of the downstream task and on a fixed vocabulary of objects and attributes. This makes it challenging for such systems to capture the long tail of visual concepts expressed in free form text. In this paper we propose MDETR, an end-to-end modulated detector that detects objects in an image conditioned on a raw text query, like a caption or a question. We use a transformer-based architecture to reason jointly over text and image by fusing the two modalities at an early stage of the model. We pre-train the network on 1.3M text-image pairs, mined from pre-existing multi-modal datasets having explicit alignment between phrases in text and objects in the image. We then fine-tune on several downstream tasks such as phrase grounding, referring expression comprehension and segmentation, achieving state-of-the-art results on popular benchmarks. We also investigate the utility of our model as an object detector on a given label set when fine-tuned in a few-shot setting. We show that our pre-training approach provides a way to handle the long tail of object categories which have very few labelled instances. Our approach can be easily extended for visual question answering, achieving competitive performance on GQA and CLEVR. The code and models are available at https://github.com/ashkamath/mdetr.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MDETR - Modulated Detection for End-to-End Multi-Modal Understanding<br>pdf: <a href="https://t.co/3wIsIudmle">https://t.co/3wIsIudmle</a><br>abs: <a href="https://t.co/tV2Ux2yxsX">https://t.co/tV2Ux2yxsX</a><br><br>MDETR, an end-to-end modulated detector that detects objects in an image conditioned on a raw text query, like a caption or a question <a href="https://t.co/A6gICVY1rr">pic.twitter.com/A6gICVY1rr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386852650709303298?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Visformer: The Vision-friendly Transformer

Zhengsu Chen, Lingxi Xie, Jianwei Niu, Xuefeng Liu, Longhui Wei, Qi Tian

- retweets: 1194, favorites: 213 (04/28/2021 09:29:54)

- links: [abs](https://arxiv.org/abs/2104.12533) | [pdf](https://arxiv.org/pdf/2104.12533)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The past year has witnessed the rapid development of applying the Transformer module to vision problems. While some researchers have demonstrated that Transformer-based models enjoy a favorable ability of fitting data, there are still growing number of evidences showing that these models suffer over-fitting especially when the training data is limited. This paper offers an empirical study by performing step-by-step operations to gradually transit a Transformer-based model to a convolution-based model. The results we obtain during the transition process deliver useful messages for improving visual recognition. Based on these observations, we propose a new architecture named Visformer, which is abbreviated from the `Vision-friendly Transformer'. With the same computational complexity, Visformer outperforms both the Transformer-based and convolution-based models in terms of ImageNet classification accuracy, and the advantage becomes more significant when the model complexity is lower or the training set is smaller. The code is available at https://github.com/danczs/Visformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Visformer: The Vision-friendly Transformer<br>pdf: <a href="https://t.co/koVlCmMnuU">https://t.co/koVlCmMnuU</a><br>abs: <a href="https://t.co/1YTeVdr2Fg">https://t.co/1YTeVdr2Fg</a><br>github: <a href="https://t.co/wHVArHwjtv">https://t.co/wHVArHwjtv</a> <a href="https://t.co/mouLZ9DT94">pic.twitter.com/mouLZ9DT94</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386851532809940997?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Improve Vision Transformers Training by Suppressing Over-smoothing

Chengyue Gong, Dilin Wang, Meng Li, Vikas Chandra, Qiang Liu

- retweets: 361, favorites: 71 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.12753) | [pdf](https://arxiv.org/pdf/2104.12753)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Introducing the transformer structure into computer vision tasks holds the promise of yielding a better speed-accuracy trade-off than traditional convolution networks. However, directly training vanilla transformers on vision tasks has been shown to yield unstable and sub-optimal results. As a result, recent works propose to modify transformer structures by incorporating convolutional layers to improve the performance on vision tasks. This work investigates how to stabilize the training of vision transformers \emph{without} special structure modification. We observe that the instability of transformer training on vision tasks can be attributed to the over-smoothing problem, that the self-attention layers tend to map the different patches from the input image into a similar latent representation, hence yielding the loss of information and degeneration of performance, especially when the number of layers is large. We then propose a number of techniques to alleviate this problem, including introducing additional loss functions to encourage diversity, prevent loss of information, and discriminate different patches by additional patch classification loss for Cutmix. We show that our proposed techniques stabilize the training and allow us to train wider and deeper vision transformers, achieving 85.0\% top-1 accuracy on ImageNet validation set without introducing extra teachers or additional convolution layers. Our code will be made publicly available at https://github.com/ChengyueGongR/PatchVisionTransformer .

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improve Vision Transformers Training by Suppressing<br>Over-smoothing<br>pdf: <a href="https://t.co/l9JsHwyT10">https://t.co/l9JsHwyT10</a><br>abs: <a href="https://t.co/cRxXv6hFqJ">https://t.co/cRxXv6hFqJ</a><br>github: <a href="https://t.co/7xBCeprDhg">https://t.co/7xBCeprDhg</a> <a href="https://t.co/OKuenxVOHU">pic.twitter.com/OKuenxVOHU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386852091839320071?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. dualFace:Two-Stage Drawing Guidance for Freehand Portrait Sketching

Zhengyu Huang, Yichen Peng, Tomohiro Hibino, Chunqi Zhao, Haoran Xie, Tsukasa Fukusato, Kazunori Miyata

- retweets: 310, favorites: 54 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.12297) | [pdf](https://arxiv.org/pdf/2104.12297)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we propose dualFace, a portrait drawing interface to assist users with different levels of drawing skills to complete recognizable and authentic face sketches. dualFace consists of two-stage drawing assistance to provide global and local visual guidance: global guidance, which helps users draw contour lines of portraits (i.e., geometric structure), and local guidance, which helps users draws details of facial parts (which conform to user-drawn contour lines), inspired by traditional artist workflows in portrait drawing. In the stage of global guidance, the user draws several contour lines, and dualFace then searches several relevant images from an internal database and displays the suggested face contour lines over the background of the canvas. In the stage of local guidance, we synthesize detailed portrait images with a deep generative model from user-drawn contour lines, but use the synthesized results as detailed drawing guidance. We conducted a user study to verify the effectiveness of dualFace, and we confirmed that dualFace significantly helps achieve a detailed portrait sketch. see http://www.jaist.ac.jp/~xie/dualface.html

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">dualFace:Two-Stage Drawing Guidance for Freehand Portrait Sketching<br>pdf: <a href="https://t.co/BNwzFvP7k0">https://t.co/BNwzFvP7k0</a><br>abs: <a href="https://t.co/odQ0gAwMFa">https://t.co/odQ0gAwMFa</a><br>github: <a href="https://t.co/FXd4V1kJYR">https://t.co/FXd4V1kJYR</a> <a href="https://t.co/ZGO7tBWj0n">pic.twitter.com/ZGO7tBWj0n</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386861725853573121?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Diverse Image Inpainting with Bidirectional and Autoregressive  Transformers

Yingchen Yu, Fangneng Zhan, Rongliang Wu, Jianxiong Pan, Kaiwen Cui, Shijian Lu, Feiying Ma, Xuansong Xie, Chunyan Miao

- retweets: 121, favorites: 47 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.12335) | [pdf](https://arxiv.org/pdf/2104.12335)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image inpainting is an underdetermined inverse problem, it naturally allows diverse contents that fill up the missing or corrupted regions reasonably and realistically. Prevalent approaches using convolutional neural networks (CNNs) can synthesize visually pleasant contents, but CNNs suffer from limited perception fields for capturing global features. With image-level attention, transformers enable to model long-range dependencies and generate diverse contents with autoregressive modeling of pixel-sequence distributions. However, the unidirectional attention in transformers is suboptimal as corrupted regions can have arbitrary shapes with contexts from arbitrary directions. We propose BAT-Fill, an image inpainting framework with a novel bidirectional autoregressive transformer (BAT) that models deep bidirectional contexts for autoregressive generation of diverse inpainting contents. BAT-Fill inherits the merits of transformers and CNNs in a two-stage manner, which allows to generate high-resolution contents without being constrained by the quadratic complexity of attention in transformers. Specifically, it first generates pluralistic image structures of low resolution by adapting transformers and then synthesizes realistic texture details of high resolutions with a CNN-based up-sampling network. Extensive experiments over multiple datasets show that BAT-Fill achieves superior diversity and fidelity in image inpainting qualitatively and quantitatively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Diverse Image Inpainting with Bidirectional and Autoregressive Transformers<br>pdf: <a href="https://t.co/hpmaerMcat">https://t.co/hpmaerMcat</a><br>abs: <a href="https://t.co/qY9zyh5m1t">https://t.co/qY9zyh5m1t</a> <a href="https://t.co/xofOQ3NGsk">pic.twitter.com/xofOQ3NGsk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387042231219982352?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Bridging observation, theory and numerical simulation of the ocean using  Machine Learning

Maike Sonnewald, Redouane Lguensat, Daniel C. Jones, Peter D. Dueben, Julien Brajard, Venkatramani Balaji

- retweets: 111, favorites: 56 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.12506) | [pdf](https://arxiv.org/pdf/2104.12506)
- [physics.ao-ph](https://arxiv.org/list/physics.ao-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We study high-dimensional Bayesian linear regression with product priors. Using the nascent theory of non-linear large deviations (Chatterjee and Dembo,2016), we derive sufficient conditions for the leading-order correctness of the naive mean-field approximation to the log-normalizing constant of the posterior distribution. Subsequently, assuming a true linear model for the observed data, we derive a limiting infinite dimensional variational formula for the log normalizing constant of the posterior. Furthermore, we establish that under an additional "separation" condition, the variational problem has a unique optimizer, and this optimizer governs the probabilistic properties of the posterior distribution. We provide intuitive sufficient conditions for the validity of this "separation" condition. Finally, we illustrate our results on concrete examples with specific design matrices.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NEW preprint on <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> in <a href="https://twitter.com/hashtag/oceanography?src=hash&amp;ref_src=twsrc%5Etfw">#oceanography</a>, led by the amazing Maike Sonnewald.  <a href="https://t.co/nTfhHpTxiT">https://t.co/nTfhHpTxiT</a><br><br>It has *not* yet been through peer review; comments and suggestions from the community are very welcome! <a href="https://t.co/R73PjM91ZW">pic.twitter.com/R73PjM91ZW</a></p>&mdash; Dr Dan Jones 🇦🇶🌊👩‍💻 (@DanJonesOcean) <a href="https://twitter.com/DanJonesOcean/status/1386984198712578049?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. baller2vec++: A Look-Ahead Multi-Entity Transformer For Modeling  Coordinated Agents

Michael A. Alcorn, Anh Nguyen

- retweets: 90, favorites: 59 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.11980) | [pdf](https://arxiv.org/pdf/2104.11980)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

Grid maps are widely established for the representation of static objects in robotics and automotive applications. Though, incorporating velocity information is still widely examined because of the increased complexity of dynamic grids concerning both velocity measurement models for radar sensors and the representation of velocity in a grid framework. In this paper, both issues are addressed: sensor models and an efficient grid framework, which are required to ensure efficient and robust environment perception with radar. To that, we introduce new inverse radar sensor models covering radar sensor artifacts such as measurement ambiguities to integrate automotive radar sensors for improved velocity estimation.   Furthermore, we introduce UNIFY, a multiple belief Bayesian grid map framework for static occupancy and velocity estimation with independent layers. The proposed UNIFY framework utilizes a grid-cell-based layer to provide occupancy information and a particle-based velocity layer for motion state estimation in an autonomous vehicle's environment. Each UNIFY layer allows individual execution as well as simultaneous execution of both layers for optimal adaption to varying environments in autonomous driving applications.   UNIFY was tested and evaluated in terms of plausibility and efficiency on a large real-world radar data-set in challenging traffic scenarios covering different densities in urban and rural sceneries.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">baller2vec++: A Look-Ahead Multi-Entity Transformer For Modeling Coordinated Agents<br>pdf: <a href="https://t.co/jZhHm8LvRM">https://t.co/jZhHm8LvRM</a><br>abs: <a href="https://t.co/cVMD7AbdQv">https://t.co/cVMD7AbdQv</a><br>github: <a href="https://t.co/IWqe9TuUHl">https://t.co/IWqe9TuUHl</a> <a href="https://t.co/lsqMOhlXEO">pic.twitter.com/lsqMOhlXEO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1387039809089118210?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. PanGu-$α$: Large-scale Autoregressive Pretrained Chinese Language  Models with Auto-parallel Computation

Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, Chen Li, Ziyan Gong, Yifan Yao, Xinjing Huang, Jun Wang, Jianfeng Yu, Qi Guo, Yue Yu, Yan Zhang, Jin Wang, Hengtao Tao, Dasen Yan, Zexuan Yi, Fang Peng, Fangqing Jiang, Han Zhang, Lingfeng Deng, Yehong Zhang, Zhe Lin, Chao Zhang, Shaojie Zhang, Mingyue Guo, Shanzhi Gu, Gaojun Fan, Yaowei Wang, Xuefeng Jin, Qun Liu, Yonghong Tian

- retweets: 98, favorites: 49 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.12369) | [pdf](https://arxiv.org/pdf/2104.12369)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Large-scale Pretrained Language Models (PLMs) have become the new paradigm for Natural Language Processing (NLP). PLMs with hundreds of billions parameters such as GPT-3 have demonstrated strong performances on natural language understanding and generation with \textit{few-shot in-context} learning. In this work, we present our practice on training large-scale autoregressive language models named PanGu-$\alpha$, with up to 200 billion parameters. PanGu-$\alpha$ is developed under the MindSpore and trained on a cluster of 2048 Ascend 910 AI processors. The training parallelism strategy is implemented based on MindSpore Auto-parallel, which composes five parallelism dimensions to scale the training task to 2048 processors efficiently, including data parallelism, op-level model parallelism, pipeline model parallelism, optimizer model parallelism and rematerialization. To enhance the generalization ability of PanGu-$\alpha$, we collect 1.1TB high-quality Chinese data from a wide range of domains to pretrain the model. We empirically test the generation ability of PanGu-$\alpha$ in various scenarios including text summarization, question answering, dialogue generation, etc. Moreover, we investigate the effect of model scales on the few-shot performances across a broad range of Chinese NLP tasks. The experimental results demonstrate the superior capabilities of PanGu-$\alpha$ in performing various tasks under few-shot or zero-shot settings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Thanks for the attention. We have just released the tech report on arxiv: <a href="https://t.co/UryZRnHCy3">https://t.co/UryZRnHCy3</a>. We believe the models are still under-trained, and have space for further improvement. <a href="https://t.co/AgpaYoM0j9">https://t.co/AgpaYoM0j9</a></p>&mdash; Xin Jiang (@jxfeb) <a href="https://twitter.com/jxfeb/status/1386883469649735680?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Playing Lottery Tickets with Vision and Language

Zhe Gan, Yen-Chun Chen, Linjie Li, Tianlong Chen, Yu Cheng, Shuohang Wang, Jingjing Liu

- retweets: 90, favorites: 23 (04/28/2021 09:29:55)

- links: [abs](https://arxiv.org/abs/2104.11832) | [pdf](https://arxiv.org/pdf/2104.11832)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A concept of drone launched short range rockets (DLSRR) is presented. A drone or an aircraft rises DLSRR to a release altitude of up to 20 $km$. At the release altitude, the drone or an aircraft is moving at a velocity of up to 700 $m/s$ and a steep angle of up to 68$^o$ to the horizontal. After DLSRRs are released, their motors start firing. DLSRRs use slow burning motors to gain altitude and velocity. At the apogee of their flight DLSRRs release projectiles which fly to the target and strike it at high impact velocity. The projectiles reach a target at ranges of up to 442 $km$ and impact velocities up to 1.88 $km/s$. We show that a rocket launched at high altitude and high initial velocity does not need expensive thermal protection to survive ascent. Delivery of munitions to target by DLSRRs should be much less expensive than delivery by a conventional rocket. %% Even though delivery of munitions by bomber aircraft is even less expensive, a bomber needs to fly close to the target, while a DLSRR carrier releases the rockets from a distance of at least 200 $km$ from the target. %% All parameters of DLSRRs, and their trajectories are calculated based on theoretical (mechanical and thermodynamical) analysis and on several MatLab programs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Playing Lottery Tickets with Vision and Language<br>pdf: <a href="https://t.co/Osa6Mb5mkD">https://t.co/Osa6Mb5mkD</a><br>abs: <a href="https://t.co/yXpBUsx6SU">https://t.co/yXpBUsx6SU</a> <a href="https://t.co/Cy8mtod2Yd">pic.twitter.com/Cy8mtod2Yd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386848052347117568?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Unikraft: Fast, Specialized Unikernels the Easy Way

Simon Kuenzer, Vlad-Andrei Bădoiu, Hugo Lefeuvre, Sharan Santhanam, Alexander Jung, Gaulthier Gain, Cyril Soldani, Costin Lupu, Ştefan Teodorescu, Costi Răducanu, Cristian Banu, Laurent Mathy, Răzvan Deaconescu, Costin Raiciu, Felipe Huici

- retweets: 81, favorites: 31 (04/28/2021 09:29:56)

- links: [abs](https://arxiv.org/abs/2104.12721) | [pdf](https://arxiv.org/pdf/2104.12721)
- [cs.OS](https://arxiv.org/list/cs.OS/recent)

Unikernels are famous for providing excellent performance in terms of boot times, throughput and memory consumption, to name a few metrics. However, they are infamous for making it hard and extremely time consuming to extract such performance, and for needing significant engineering effort in order to port applications to them. We introduce Unikraft, a novel micro-library OS that (1) fully modularizes OS primitives so that it is easy to customize the unikernel and include only relevant components and (2) exposes a set of composable, performance-oriented APIs in order to make it easy for developers to obtain high performance.   Our evaluation using off-the-shelf applications such as nginx, SQLite, and Redis shows that running them on Unikraft results in a 1.7x-2.7x performance improvement compared to Linux guests. In addition, Unikraft images for these apps are around 1MB, require less than 10MB of RAM to run, and boot in around 1ms on top of the VMM time (total boot time 3ms-40ms). Unikraft is a Linux Foundation open source project and can be found at www.unikraft.org.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unikraft – Fast, Specialized Unikernels <a href="https://t.co/3hDrVguZRW">https://t.co/3hDrVguZRW</a></p>&mdash; Hacker News (@newsycombinator) <a href="https://twitter.com/newsycombinator/status/1387014158047223812?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object  Detection with Transformers

Tianrui Guan, Jun Wang, Shiyi Lan, Rohan Chandra, Zuxuan Wu, Larry Davis, Dinesh Manocha

- retweets: 81, favorites: 29 (04/28/2021 09:29:56)

- links: [abs](https://arxiv.org/abs/2104.11896) | [pdf](https://arxiv.org/pdf/2104.11896)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent theoretical works on over-parameterized neural nets have focused on two aspects: optimization and generalization. Many existing works that study optimization and generalization together are based on neural tangent kernel and require a very large width. In this work, we are interested in the following question: for a binary classification problem with two-layer mildly over-parameterized ReLU network, can we find a point with small test error in polynomial time? We first show that the landscape of loss functions with explicit regularization has the following property: all local minima and certain other points which are only stationary in certain directions achieve small test error. We then prove that for convolutional neural nets, there is an algorithm which finds one of these points in polynomial time (in the input dimension and the number of data points). In addition, we prove that for a fully connected neural net, with an additional assumption on the data distribution, there is a polynomial time algorithm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">M3DeTR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers<br>pdf: <a href="https://t.co/unjK4oj4cJ">https://t.co/unjK4oj4cJ</a><br>abs: <a href="https://t.co/JeCnRO7xO9">https://t.co/JeCnRO7xO9</a> <a href="https://t.co/GkaoYuCSpQ">pic.twitter.com/GkaoYuCSpQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386848962792001536?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Towards Good Practices for Efficiently Annotating Large-Scale Image  Classification Datasets

Yuan-Hong Liao, Amlan Kar, Sanja Fidler

- retweets: 49, favorites: 29 (04/28/2021 09:29:56)

- links: [abs](https://arxiv.org/abs/2104.12690) | [pdf](https://arxiv.org/pdf/2104.12690)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Data is the engine of modern computer vision, which necessitates collecting large-scale datasets. This is expensive, and guaranteeing the quality of the labels is a major challenge. In this paper, we investigate efficient annotation strategies for collecting multi-class classification labels for a large collection of images. While methods that exploit learnt models for labeling exist, a surprisingly prevalent approach is to query humans for a fixed number of labels per datum and aggregate them, which is expensive. Building on prior work on online joint probabilistic modeling of human annotations and machine-generated beliefs, we propose modifications and best practices aimed at minimizing human labeling effort. Specifically, we make use of advances in self-supervised learning, view annotation as a semi-supervised learning problem, identify and mitigate pitfalls and ablate several key design choices to propose effective guidelines for labeling. Our analysis is done in a more realistic simulation that involves querying human labelers, which uncovers issues with evaluation using existing worker simulation methods. Simulated experiments on a 125k image subset of the ImageNet100 show that it can be annotated to 80% top-1 accuracy with 0.35 annotations per image on average, a 2.7x and 6.7x improvement over prior work and manual annotation, respectively. Project page: https://fidler-lab.github.io/efficient-annotation-cookbook

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Good Practices for Efficiently Annotating Large-Scale Image Classification Datasets<br>pdf: <a href="https://t.co/RDhOb9Yq9W">https://t.co/RDhOb9Yq9W</a><br>abs: <a href="https://t.co/24NzW4O2HP">https://t.co/24NzW4O2HP</a><br>project page: <a href="https://t.co/EzW6Xumtfl">https://t.co/EzW6Xumtfl</a> <a href="https://t.co/BjPCiocTuS">pic.twitter.com/BjPCiocTuS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1386874983926927369?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Focused Attention Improves Document-Grounded Generation

Shrimai Prabhumoye, Kazuma Hashimoto, Yingbo Zhou, Alan W Black, Ruslan Salakhutdinov

- retweets: 30, favorites: 46 (04/28/2021 09:29:56)

- links: [abs](https://arxiv.org/abs/2104.12714) | [pdf](https://arxiv.org/pdf/2104.12714)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Document grounded generation is the task of using the information provided in a document to improve text generation. This work focuses on two different document grounded generation tasks: Wikipedia Update Generation task and Dialogue response generation. Our work introduces two novel adaptations of large scale pre-trained encoder-decoder models focusing on building context driven representation of the document and enabling specific attention to the information in the document. Additionally, we provide a stronger BART baseline for these tasks. Our proposed techniques outperform existing methods on both automated (at least 48% increase in BLEU-4 points) and human evaluation for closeness to reference and relevance to the document. Furthermore, we perform comprehensive manual inspection of the generated output and categorize errors to provide insights into future directions in modeling these tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m very excited to announce that our work &quot;Focused Attention Improves Document-Grounded Generation&quot; w/ Kazuma Hashimoto <a href="https://twitter.com/SFResearch?ref_src=twsrc%5Etfw">@SFResearch</a>, <a href="https://twitter.com/yingbozhou_ai?ref_src=twsrc%5Etfw">@yingbozhou_ai</a>, Alan W Black, <a href="https://twitter.com/rsalakhu?ref_src=twsrc%5Etfw">@rsalakhu</a> has been accepted <a href="https://twitter.com/NAACLHLT?ref_src=twsrc%5Etfw">@NAACLHLT</a>. <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> <br>Paper: <a href="https://t.co/oxazBO0vJ5">https://t.co/oxazBO0vJ5</a><br>Code: <a href="https://t.co/h0lMJloVTs">https://t.co/h0lMJloVTs</a> <a href="https://t.co/ZyJRnFLdtJ">pic.twitter.com/ZyJRnFLdtJ</a></p>&mdash; Shrimai (@shrimai_) <a href="https://twitter.com/shrimai_/status/1387044308658769927?ref_src=twsrc%5Etfw">April 27, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



