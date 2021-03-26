---
title: Hot Papers 2021-03-25
date: 2021-03-26T11:41:31.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. Finetuning Pretrained Transformers into RNNs

Jungo Kasai, Hao Peng, Yizhe Zhang, Dani Yogatama, Gabriel Ilharco, Nikolaos Pappas, Yi Mao, Weizhu Chen, Noah A. Smith

- retweets: 3889, favorites: 450 (03/26/2021 11:41:31)

- links: [abs](https://arxiv.org/abs/2103.13076) | [pdf](https://arxiv.org/pdf/2103.13076)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Transformers have outperformed recurrent neural networks (RNNs) in natural language generation. This comes with a significant computational overhead, as the attention mechanism scales with a quadratic complexity in sequence length. Efficient transformer variants have received increasing interest from recent works. Among them, a linear-complexity recurrent variant has proven well suited for autoregressive generation. It approximates the softmax attention with randomized or heuristic feature maps, but can be difficult to train or yield suboptimal accuracy. This work aims to convert a pretrained transformer into its efficient recurrent counterpart, improving the efficiency while retaining the accuracy. Specifically, we propose a swap-then-finetune procedure: in an off-the-shelf pretrained transformer, we replace the softmax attention with its linear-complexity recurrent alternative and then finetune. With a learned feature map, our approach provides an improved tradeoff between efficiency and accuracy over the standard transformer and other recurrent variants. We also show that the finetuning process needs lower training cost than training these recurrent variants from scratch. As many recent models for natural language tasks are increasingly dependent on large-scale pretrained transformers, this work presents a viable approach to improving inference efficiency without repeating the expensive pretraining process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Finetuning Pretrained Transformers into RNNs<br><br>Successfully converts a pretrained transformer into its efficient linear-complexity recurrent counterpart with a learned feature map to improve the efficiency while retaining the accuracy.<a href="https://t.co/yRfENT2ch2">https://t.co/yRfENT2ch2</a> <a href="https://t.co/k4gwCljy7m">pic.twitter.com/k4gwCljy7m</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1374890328621125638?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Finetuning Pretrained Transformers into RNNs<br>pdf: <a href="https://t.co/EcFo5w5JYV">https://t.co/EcFo5w5JYV</a><br>abs: <a href="https://t.co/XIVw5xTYUG">https://t.co/XIVw5xTYUG</a> <a href="https://t.co/KKN1Tz3m3O">pic.twitter.com/KKN1Tz3m3O</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374886338902818817?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. FastMoE: A Fast Mixture-of-Expert Training System

Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, Jie Tang

- retweets: 913, favorites: 191 (03/26/2021 11:41:31)

- links: [abs](https://arxiv.org/abs/2103.13262) | [pdf](https://arxiv.org/pdf/2103.13262)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Mixture-of-Expert (MoE) presents a strong potential in enlarging the size of language model to trillions of parameters. However, training trillion-scale MoE requires algorithm and system co-design for a well-tuned high performance distributed training system. Unfortunately, the only existing platform that meets the requirements strongly depends on Google's hardware (TPU) and software (Mesh Tensorflow) stack, and is not open and available to the public, especially GPU and PyTorch communities.   In this paper, we present FastMoE, a distributed MoE training system based on PyTorch with common accelerators. The system provides a hierarchical interface for both flexible model design and easy adaption to different applications, such as Transformer-XL and Megatron-LM. Different from direct implementation of MoE models using PyTorch, the training speed is highly optimized in FastMoE by sophisticated high-performance acceleration skills. The system supports placing different experts on multiple GPUs across multiple nodes, enabling enlarging the number of experts linearly against the number of GPUs. The source of FastMoE is available at https://github.com/laekov/fastmoe under Apache-2 license.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FastMoE: A Fast Mixture-of-Expert Training System<br><br>Presents FastMoE, a distributed MoE training system based on PyTorch that works with GPUs unlike the existing MoE.<br><br>abs: <a href="https://t.co/TPpwrT4QOt">https://t.co/TPpwrT4QOt</a><br>code: <a href="https://t.co/4GNIpJssFo">https://t.co/4GNIpJssFo</a> <a href="https://t.co/KqEErUSZhQ">pic.twitter.com/KqEErUSZhQ</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1374885622570082309?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FastMoE: A Fast Mixture-of-Expert Training System<br>pdf: <a href="https://t.co/8x3dr431BS">https://t.co/8x3dr431BS</a><br>abs: <a href="https://t.co/YqZ3orckPr">https://t.co/YqZ3orckPr</a><br>github: <a href="https://t.co/DcTY8k5xln">https://t.co/DcTY8k5xln</a> <a href="https://t.co/yfcnElbTpt">pic.twitter.com/yfcnElbTpt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374886803648475138?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Can Vision Transformers Learn without Natural Images?

Kodai Nakashima, Hirokatsu Kataoka, Asato Matsumoto, Kenji Iwata, Nakamasa Inoue

- retweets: 816, favorites: 254 (03/26/2021 11:41:32)

- links: [abs](https://arxiv.org/abs/2103.13023) | [pdf](https://arxiv.org/pdf/2103.13023)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Can we complete pre-training of Vision Transformers (ViT) without natural images and human-annotated labels? Although a pre-trained ViT seems to heavily rely on a large-scale dataset and human-annotated labels, recent large-scale datasets contain several problems in terms of privacy violations, inadequate fairness protection, and labor-intensive annotation. In the present paper, we pre-train ViT without any image collections and annotation labor. We experimentally verify that our proposed framework partially outperforms sophisticated Self-Supervised Learning (SSL) methods like SimCLRv2 and MoCov2 without using any natural images in the pre-training phase. Moreover, although the ViT pre-trained without natural images produces some different visualizations from ImageNet pre-trained ViT, it can interpret natural image datasets to a large extent. For example, the performance rates on the CIFAR-10 dataset are as follows: our proposal 97.6 vs. SimCLRv2 97.4 vs. ImageNet 98.0.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can Vision Transformers Learn without Natural Images?<br>pdf: <a href="https://t.co/aLjEYIjZhu">https://t.co/aLjEYIjZhu</a><br>abs: <a href="https://t.co/7BSqg3sttI">https://t.co/7BSqg3sttI</a><br>project page: <a href="https://t.co/T14KziiPlM">https://t.co/T14KziiPlM</a> <a href="https://t.co/DgUH45eqnM">pic.twitter.com/DgUH45eqnM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374885886610001920?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can Vision Transformers Learn without Natural Images?<br><br>Partially outperforms strong SSL baselines such as SimCLRv2 and MoCov2 w/o using any natural images in the pre-training phase.<br><br>abs: <a href="https://t.co/SYrWTjpwqk">https://t.co/SYrWTjpwqk</a><br>project: <a href="https://t.co/2pUrWTk5w9">https://t.co/2pUrWTk5w9</a> <a href="https://t.co/KDgLHP8iQu">pic.twitter.com/KDgLHP8iQu</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1374887227285639170?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">自然界にはフラクタル性があると言われていますが、フラクタル画像使い倒すと最早画像データセット要らなくない？と言うのを友人の <a href="https://twitter.com/HirokatuKataoka?ref_src=twsrc%5Etfw">@HirokatuKataoka</a> さんたちがやっていてACCVでHonorable MentionされてたやつのTransformer版が出てる。<a href="https://t.co/xkckHfSBiy">https://t.co/xkckHfSBiy</a></p>&mdash; Yoshitaka Ushiku (@losnuevetoros) <a href="https://twitter.com/losnuevetoros/status/1374936120144715780?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">我々の論文 “Can Vision Transformers Learn without Natural Images?” をarXivに掲載しました！<br><br>PDF: <a href="https://t.co/sfjEkzwd8p">https://t.co/sfjEkzwd8p</a><br>Project page: <a href="https://t.co/3e4vpC2Eae">https://t.co/3e4vpC2Eae</a> <a href="https://t.co/HdbdMXGOlU">pic.twitter.com/HdbdMXGOlU</a></p>&mdash; cvpaper.challenge | AI/CV研究コミュニティ (@CVpaperChalleng) <a href="https://twitter.com/CVpaperChalleng/status/1374900874787520512?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. AutoMix: Unveiling the Power of Mixup

Zicheng Liu, Siyuan Li, Di Wu, Zhiyuan Chen, Lirong Wu, Jianzhu Guo, Stan Z. Li

- retweets: 441, favorites: 108 (03/26/2021 11:41:32)

- links: [abs](https://arxiv.org/abs/2103.13027) | [pdf](https://arxiv.org/pdf/2103.13027)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Mixup-based data augmentation has achieved great success as regularizer for deep neural networks. However, existing mixup methods require explicitly designed mixup policies. In this paper, we present a flexible, general Automatic Mixup (AutoMix) framework which utilizes discriminative features to learn a sample mixing policy adaptively. We regard mixup as a pretext task and split it into two sub-problems: mixed samples generation and mixup classification. To this end, we design a lightweight mix block to generate synthetic samples based on feature maps and mix labels. Since the two sub-problems are in the nature of Expectation-Maximization (EM), we also propose a momentum training pipeline to optimize the mixup process and mixup classification process alternatively in an end-to-end fashion. Extensive experiments on six popular classification benchmarks show that AutoMix consistently outperforms other leading mixup methods and improves generalization abilities to downstream tasks. We hope AutoMix will motivate the community to rethink the role of mixup in representation learning. The code will be released soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AutoMix: Unveiling the Power of Mixup<a href="https://t.co/rgXpFRzNjE">https://t.co/rgXpFRzNjE</a> <a href="https://t.co/rkhMa10RhJ">pic.twitter.com/rkhMa10RhJ</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1374938738858422273?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Multi-view 3D Reconstruction with Transformer

Dan Wang, Xinrui Cui, Xun Chen, Zhengxia Zou, Tianyang Shi, Septimiu Salcudean, Z. Jane Wang, Rabab Ward

- retweets: 357, favorites: 100 (03/26/2021 11:41:32)

- links: [abs](https://arxiv.org/abs/2103.12957) | [pdf](https://arxiv.org/pdf/2103.12957)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep CNN-based methods have so far achieved the state of the art results in multi-view 3D object reconstruction. Despite the considerable progress, the two core modules of these methods - multi-view feature extraction and fusion, are usually investigated separately, and the object relations in different views are rarely explored. In this paper, inspired by the recent great success in self-attention-based Transformer models, we reformulate the multi-view 3D reconstruction as a sequence-to-sequence prediction problem and propose a new framework named 3D Volume Transformer (VolT) for such a task. Unlike previous CNN-based methods using a separate design, we unify the feature extraction and view fusion in a single Transformer network. A natural advantage of our design lies in the exploration of view-to-view relationships using self-attention among multiple unordered inputs. On ShapeNet - a large-scale 3D reconstruction benchmark dataset, our method achieves a new state-of-the-art accuracy in multi-view reconstruction with fewer parameters ($70\%$ less) than other CNN-based methods. Experimental results also suggest the strong scaling capability of our method. Our code will be made publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-view 3D Reconstruction with Transformer<br>pdf: <a href="https://t.co/BsedtkzixG">https://t.co/BsedtkzixG</a><br>abs: <a href="https://t.co/BgmFtsxyUH">https://t.co/BgmFtsxyUH</a> <a href="https://t.co/Os8fUpKB5z">pic.twitter.com/Os8fUpKB5z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374887344373960705?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. One-Shot GAN: Learning to Generate Samples from Single Images and Videos

Vadim Sushko, Juergen Gall, Anna Khoreva

- retweets: 162, favorites: 95 (03/26/2021 11:41:32)

- links: [abs](https://arxiv.org/abs/2103.13389) | [pdf](https://arxiv.org/pdf/2103.13389)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Given a large number of training samples, GANs can achieve remarkable performance for the image synthesis task. However, training GANs in extremely low-data regimes remains a challenge, as overfitting often occurs, leading to memorization or training divergence. In this work, we introduce One-Shot GAN, an unconditional generative model that can learn to generate samples from a single training image or a single video clip. We propose a two-branch discriminator architecture, with content and layout branches designed to judge internal content and scene layout realism separately from each other. This allows synthesis of visually plausible, novel compositions of a scene, with varying content and layout, while preserving the context of the original sample. Compared to previous single-image GAN models, One-Shot GAN generates more diverse, higher quality images, while also not being restricted to a single image setting. We show that our model successfully deals with other one-shot regimes, and introduce a new task of learning generative models from a single video.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">One-Shot GAN: Learning to Generate Samples from Single Images and Videos<br>pdf: <a href="https://t.co/GikSikQxPx">https://t.co/GikSikQxPx</a><br>abs: <a href="https://t.co/FLTxC6PoAM">https://t.co/FLTxC6PoAM</a> <a href="https://t.co/plefaU1a5n">pic.twitter.com/plefaU1a5n</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374889538322112516?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Koo: The new King? Characterizing India's Emerging Social Network

Asmit Kumar Singh, Chirag Jain, Jivitesh Jain, Rishi Raj Jain, Shradha Sehgal, Ponnurangam Kumaraguru

- retweets: 58, favorites: 58 (03/26/2021 11:41:32)

- links: [abs](https://arxiv.org/abs/2103.13239) | [pdf](https://arxiv.org/pdf/2103.13239)
- [cs.SI](https://arxiv.org/list/cs.SI/recent)

Social media has grown exponentially in a short period, coming to the forefront of communications and online interactions. Despite their rapid growth, social media platforms have been unable to scale to different languages globally and remain inaccessible to many. In this report, we characterize Koo, a multilingual micro-blogging site that rose in popularity in 2021, as an Indian alternative to Twitter. We collected a dataset of 4.07 million users, 163.12 million follower-following relationships, and their content and activity across 12 languages. The prominent presence of Indian languages in the discourse on Koo indicates the platform's success in promoting regional languages. We observe Koo's follower-following network to be much denser than Twitter's, comprising of closely-knit linguistic communities. This initial characterization heralds a deeper study of the dynamics of the multilingual social network and its diverse Indian user base.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Is <a href="https://twitter.com/hashtag/kooapp?src=hash&amp;ref_src=twsrc%5Etfw">#kooapp</a>, the new social network, India’s king?<br>With 4M users &amp; 163M follower relations, we find out!<br>Linguistic communities; <a href="https://twitter.com/hashtag/Hindi?src=hash&amp;ref_src=twsrc%5Etfw">#Hindi</a> <a href="https://twitter.com/hashtag/Bengaluru?src=hash&amp;ref_src=twsrc%5Etfw">#Bengaluru</a> prominent; <br>Video: <a href="https://t.co/YxaTJUXdk1">https://t.co/YxaTJUXdk1</a> <br>Full report: <a href="https://t.co/XSFEQJjz2t">https://t.co/XSFEQJjz2t</a> <br>\c <a href="https://twitter.com/aprameya?ref_src=twsrc%5Etfw">@aprameya</a> <a href="https://twitter.com/mayankbidawatka?ref_src=twsrc%5Etfw">@mayankbidawatka</a> <a href="https://twitter.com/rsprasad?ref_src=twsrc%5Etfw">@rsprasad</a> <a href="https://twitter.com/kooindia?ref_src=twsrc%5Etfw">@kooindia</a></p>&mdash; Ponnurangam Kumaraguru “PK” (@ponguru) <a href="https://twitter.com/ponguru/status/1374956889620967426?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. AcinoSet: A 3D Pose Estimation Dataset and Baseline Models for Cheetahs  in the Wild

Daniel Joska, Liam Clark, Naoya Muramatsu, Ricardo Jericevich, Fred Nicolls, Alexander Mathis, Mackenzie W. Mathis, Amir Patel

- retweets: 30, favorites: 41 (03/26/2021 11:41:33)

- links: [abs](https://arxiv.org/abs/2103.13282) | [pdf](https://arxiv.org/pdf/2103.13282)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

Animals are capable of extreme agility, yet understanding their complex dynamics, which have ecological, biomechanical and evolutionary implications, remains challenging. Being able to study this incredible agility will be critical for the development of next-generation autonomous legged robots. In particular, the cheetah (acinonyx jubatus) is supremely fast and maneuverable, yet quantifying its whole-body 3D kinematic data during locomotion in the wild remains a challenge, even with new deep learning-based methods. In this work we present an extensive dataset of free-running cheetahs in the wild, called AcinoSet, that contains 119,490 frames of multi-view synchronized high-speed video footage, camera calibration files and 7,588 human-annotated frames. We utilize markerless animal pose estimation to provide 2D keypoints. Then, we use three methods that serve as strong baselines for 3D pose estimation tool development: traditional sparse bundle adjustment, an Extended Kalman Filter, and a trajectory optimization-based method we call Full Trajectory Estimation. The resulting 3D trajectories, human-checked 3D ground truth, and an interactive tool to inspect the data is also provided. We believe this dataset will be useful for a diverse range of fields such as ecology, neuroscience, robotics, biomechanics as well as computer vision.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🐆 Care about 3D animal pose &amp; bio-inspired robotics?<br><br>🔥 In collaboration w/<a href="https://twitter.com/UnitAfrican?ref_src=twsrc%5Etfw">@UnitAfrican</a>:<br><br>𝘈𝘤𝘪𝘯𝘰𝘚𝘦𝘵: 𝘈 3𝘋 𝘗𝘰𝘴𝘦 𝘌𝘴𝘵𝘪𝘮𝘢𝘵𝘪𝘰𝘯 𝘋𝘢𝘵𝘢𝘴𝘦𝘵 &amp; 𝘉𝘢𝘴𝘦𝘭𝘪𝘯𝘦 𝘔𝘰𝘥𝘦𝘭𝘴 𝘧𝘰𝘳 𝘊𝘩𝘦𝘦𝘵𝘢𝘩𝘴 𝘪𝘯 𝘵𝘩𝘦 𝘞𝘪𝘭𝘥<br><br>🥳 at <a href="https://twitter.com/hashtag/ICRA2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICRA2021</a><a href="https://t.co/actdU2tHLY">https://t.co/actdU2tHLY</a> <a href="https://t.co/v9lHQbz9zJ">pic.twitter.com/v9lHQbz9zJ</a></p>&mdash; Mackenzie Mathis (@TrackingActions) <a href="https://twitter.com/TrackingActions/status/1375150637487390738?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Weakly Supervised Instance Segmentation for Videos with Temporal Mask  Consistency

Qing Liu, Vignesh Ramanathan, Dhruv Mahajan, Alan Yuille, Zhenheng Yang

- retweets: 25, favorites: 25 (03/26/2021 11:41:33)

- links: [abs](https://arxiv.org/abs/2103.12886) | [pdf](https://arxiv.org/pdf/2103.12886)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Weakly supervised instance segmentation reduces the cost of annotations required to train models. However, existing approaches which rely only on image-level class labels predominantly suffer from errors due to (a) partial segmentation of objects and (b) missing object predictions. We show that these issues can be better addressed by training with weakly labeled videos instead of images. In videos, motion and temporal consistency of predictions across frames provide complementary signals which can help segmentation. We are the first to explore the use of these video signals to tackle weakly supervised instance segmentation. We propose two ways to leverage this information in our model. First, we adapt inter-pixel relation network (IRN) to effectively incorporate motion information during training. Second, we introduce a new MaskConsist module, which addresses the problem of missing object instances by transferring stable predictions between neighboring frames during training. We demonstrate that both approaches together improve the instance segmentation metric $AP_{50}$ on video frames of two datasets: Youtube-VIS and Cityscapes by $5\%$ and $3\%$ respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Weakly Supervised Instance Segmentation for Videos with Temporal Mask Consistency<br>pdf: <a href="https://t.co/05IA9xzsR9">https://t.co/05IA9xzsR9</a><br>abs: <a href="https://t.co/BaNhx6BRtR">https://t.co/BaNhx6BRtR</a> <a href="https://t.co/ZUoXwrWrPm">pic.twitter.com/ZUoXwrWrPm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1374903796506169347?ref_src=twsrc%5Etfw">March 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



