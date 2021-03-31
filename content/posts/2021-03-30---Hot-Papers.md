---
title: Hot Papers 2021-03-30
date: 2021-03-31T10:26:50.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-30"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-30"
socialImage: "/media/flying-marine.jpg"

---

# 1. ViViT: A Video Vision Transformer

Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid

- retweets: 2739, favorites: 330 (03/31/2021 10:26:50)

- links: [abs](https://arxiv.org/abs/2103.15691) | [pdf](https://arxiv.org/pdf/2103.15691)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification. Our model extracts spatio-temporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show how we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks. To facilitate further research, we will release code and models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ViViT: A Video Vision Transformer<br><br>Achieves SotA on various video classification benchmarks including Kinetics 400, outperforming prior methods based on 3D CNN.<a href="https://t.co/d8nR8cU6Ac">https://t.co/d8nR8cU6Ac</a> <a href="https://t.co/dSkB7b7lac">pic.twitter.com/dSkB7b7lac</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1376707154522955776?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Pervasive Label Errors in Test Sets Destabilize Machine Learning  Benchmarks

Curtis G. Northcutt, Anish Athalye, Jonas Mueller

- retweets: 1522, favorites: 193 (03/31/2021 10:26:51)

- links: [abs](https://arxiv.org/abs/2103.14749) | [pdf](https://arxiv.org/pdf/2103.14749)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We identify label errors in the test sets of 10 of the most commonly-used computer vision, natural language, and audio datasets, and subsequently study the potential for these label errors to affect benchmark results. Errors in test sets are numerous and widespread: we estimate an average of 3.4% errors across the 10 datasets, where for example 2916 label errors comprise 6% of the ImageNet validation set. Putative label errors are identified using confident learning algorithms and then human-validated via crowdsourcing (54% of the algorithmically-flagged candidates are indeed erroneously labeled). Traditionally, machine learning practitioners choose which model to deploy based on test accuracy - our findings advise caution here, proposing that judging models over correctly labeled test sets may be more useful, especially for noisy real-world datasets. Surprisingly, we find that lower capacity models may be practically more useful than higher capacity models in real-world datasets with high proportions of erroneously labeled data. For example, on ImageNet with corrected labels: ResNet-18 outperforms ResNet50 if the prevalence of originally mislabeled test examples increases by just 6%. On CIFAR-10 with corrected labels: VGG-11 outperforms VGG-19 if the prevalence of originally mislabeled test examples increases by just 5%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We all know ML datasets are filled with errors. <br><br>But now there&#39;s actually a list.<a href="https://t.co/AiDPwCyoWl">https://t.co/AiDPwCyoWl</a><br><br>MIT researchers found 3.4% errors across 10 widely-cited datasets, incl. ImageNet &amp; Amazon Reviews: <a href="https://t.co/pciP0jmiX5">https://t.co/pciP0jmiX5</a><br><br>Paper: <a href="https://t.co/slGhBt6KlO">https://t.co/slGhBt6KlO</a> <a href="https://t.co/cqjHgkYQox">pic.twitter.com/cqjHgkYQox</a></p>&mdash; MIT CSAIL (@MIT_CSAIL) <a href="https://twitter.com/MIT_CSAIL/status/1376967812191825922?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Quantum Self-Supervised Learning

Ben Jaderberg, Lewis W. Anderson, Weidi Xie, Samuel Albanie, Martin Kiffner, Dieter Jaksch

- retweets: 1211, favorites: 194 (03/31/2021 10:26:51)

- links: [abs](https://arxiv.org/abs/2103.14653) | [pdf](https://arxiv.org/pdf/2103.14653)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The popularisation of neural networks has seen incredible advances in pattern recognition, driven by the supervised learning of human annotations. However, this approach is unsustainable in relation to the dramatically increasing size of real-world datasets. This has led to a resurgence in self-supervised learning, a paradigm whereby the model generates its own supervisory signal from the data. Here we propose a hybrid quantum-classical neural network architecture for contrastive self-supervised learning and test its effectiveness in proof-of-principle experiments. Interestingly, we observe a numerical advantage for the learning of visual representations using small-scale quantum neural networks over equivalently structured classical networks, even when the quantum circuits are sampled with only 100 shots. Furthermore, we apply our best quantum model to classify unseen images on the ibmq_paris quantum computer and find that current noisy devices can already achieve equal accuracy to the equivalent classical model on downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Super cool: self-supervised learning on images with a Quantum Neural Network. The quantum net learns better representations than the equivalent classical nets for use on downstream tasks. And is actually evaluated on a real quantum computer! <a href="https://t.co/NbkEmkoXug">https://t.co/NbkEmkoXug</a> <a href="https://twitter.com/benjaderberg?ref_src=twsrc%5Etfw">@benjaderberg</a> <a href="https://t.co/9KWEM6y3B4">pic.twitter.com/9KWEM6y3B4</a></p>&mdash; Max Jaderberg (@maxjaderberg) <a href="https://twitter.com/maxjaderberg/status/1376886882995425285?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Labels4Free: Unsupervised Segmentation using StyleGAN

Rameen Abdal, Peihao Zhu, Niloy Mitra, Peter Wonka

- retweets: 1122, favorites: 128 (03/31/2021 10:26:52)

- links: [abs](https://arxiv.org/abs/2103.14968) | [pdf](https://arxiv.org/pdf/2103.14968)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose an unsupervised segmentation framework for StyleGAN generated objects. We build on two main observations. First, the features generated by StyleGAN hold valuable information that can be utilized towards training segmentation networks. Second, the foreground and background can often be treated to be largely independent and be composited in different ways. For our solution, we propose to augment the StyleGAN2 generator architecture with a segmentation branch and to split the generator into a foreground and background network. This enables us to generate soft segmentation masks for the foreground object in an unsupervised fashion. On multiple object classes, we report comparable results against state-of-the-art supervised segmentation networks, while against the best unsupervised segmentation approach we demonstrate a clear improvement, both in qualitative and quantitative metrics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Labels4Free: Unsupervised Segmentation using StyleGAN<br>pdf: <a href="https://t.co/BWuxmauugW">https://t.co/BWuxmauugW</a><br>abs: <a href="https://t.co/pxRiOaDLsY">https://t.co/pxRiOaDLsY</a><br>project page: <a href="https://t.co/EUhEHaAGeT">https://t.co/EUhEHaAGeT</a> <a href="https://t.co/siZScaGFaf">pic.twitter.com/siZScaGFaf</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376716793268551680?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. CvT: Introducing Convolutions to Vision Transformers

Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang

- retweets: 889, favorites: 194 (03/31/2021 10:26:52)

- links: [abs](https://arxiv.org/abs/2103.15808) | [pdf](https://arxiv.org/pdf/2103.15808)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection. These changes introduce desirable properties of convolutional neural networks (CNNs) to the ViT architecture (\ie shift, scale, and distortion invariance) while maintaining the merits of Transformers (\ie dynamic attention, global context, and better generalization). We validate CvT by conducting extensive experiments, showing that this approach achieves state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs. In addition, performance gains are maintained when pretrained on larger datasets (\eg ImageNet-22k) and fine-tuned to downstream tasks. Pre-trained on ImageNet-22k, our CvT-W24 obtains a top-1 accuracy of 87.7\% on the ImageNet-1k val set. Finally, our results show that the positional encoding, a crucial component in existing Vision Transformers, can be safely removed in our model, simplifying the design for higher resolution vision tasks. Code will be released at \url{https://github.com/leoxiaobin/CvT}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CvT: Introducing Convolutions to Vision Transformers<br><br>Achieves SotA over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs by adding a hierarchy of Transformers.<br><br>abs: <a href="https://t.co/EjCHoVTIqR">https://t.co/EjCHoVTIqR</a><br>code: <a href="https://t.co/nHtw6zR6bJ">https://t.co/nHtw6zR6bJ</a> <a href="https://t.co/a2FjbIejw3">pic.twitter.com/a2FjbIejw3</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1376705831438192641?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">CvT: Introducing Convolutions to Vision Transformers<br>pdf: <a href="https://t.co/59l2nRLrLe">https://t.co/59l2nRLrLe</a><br>abs: <a href="https://t.co/xU9s1p2N45">https://t.co/xU9s1p2N45</a> <a href="https://t.co/CEoaWE7peo">pic.twitter.com/CEoaWE7peo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376712436921860098?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Few-shot Semantic Image Synthesis Using StyleGAN Prior

Yuki Endo, Yoshihiro Kanamori

- retweets: 484, favorites: 115 (03/31/2021 10:26:53)

- links: [abs](https://arxiv.org/abs/2103.14877) | [pdf](https://arxiv.org/pdf/2103.14877)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

This paper tackles a challenging problem of generating photorealistic images from semantic layouts in few-shot scenarios where annotated training pairs are hardly available but pixel-wise annotation is quite costly. We present a training strategy that performs pseudo labeling of semantic masks using the StyleGAN prior. Our key idea is to construct a simple mapping between the StyleGAN feature and each semantic class from a few examples of semantic masks. With such mappings, we can generate an unlimited number of pseudo semantic masks from random noise to train an encoder for controlling a pre-trained StyleGAN generator. Although the pseudo semantic masks might be too coarse for previous approaches that require pixel-aligned masks, our framework can synthesize high-quality images from not only dense semantic masks but also sparse inputs such as landmarks and scribbles. Qualitative and quantitative results with various datasets demonstrate improvement over previous approaches with respect to layout fidelity and visual quality in as few as one- or five-shot settings.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-shot Semantic Image Synthesis Using StyleGAN Prior<br>pdf: <a href="https://t.co/RztH9fL6qC">https://t.co/RztH9fL6qC</a><br>abs: <a href="https://t.co/kTDZrhvUnl">https://t.co/kTDZrhvUnl</a> <a href="https://t.co/UNKuRt8flC">pic.twitter.com/UNKuRt8flC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376716146490114053?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Autonomous Overtaking in Gran Turismo Sport Using Curriculum  Reinforcement Learning

Yunlong Song, HaoChih Lin, Elia Kaufmann, Peter Duerr, Davide Scaramuzza

- retweets: 380, favorites: 105 (03/31/2021 10:26:53)

- links: [abs](https://arxiv.org/abs/2103.14666) | [pdf](https://arxiv.org/pdf/2103.14666)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Professional race car drivers can execute extreme overtaking maneuvers. However, conventional systems for autonomous overtaking rely on either simplified assumptions about the vehicle dynamics or solving expensive trajectory optimization problems online. When the vehicle is approaching its physical limits, existing model-based controllers struggled to handle highly nonlinear dynamics and cannot leverage the large volume of data generated by simulation or real-world driving. To circumvent these limitations, this work proposes a new learning-based method to tackle the autonomous overtaking problem. We evaluate our approach using Gran Turismo Sport -- a world-leading car racing simulator known for its detailed dynamic modeling of various cars and tracks. By leveraging curriculum learning, our approach leads to faster convergence as well as increased performance compared to vanilla reinforcement learning. As a result, the trained controller outperforms the built-in model-based game AI and achieves comparable overtaking performance with an experienced human driver.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Autonomous Overtaking in Gran Turismo Sport Using Curriculum Reinforcement Learning<br>pdf: <a href="https://t.co/p75amhpRjy">https://t.co/p75amhpRjy</a><br>abs: <a href="https://t.co/wb7G4Fn8WK">https://t.co/wb7G4Fn8WK</a> <a href="https://t.co/XlvbsxuKA3">pic.twitter.com/XlvbsxuKA3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376748949571309571?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image  Generative Models

Niv Granot, Assaf Shocher, Ben Feinstein, Shai Bagon, Michal Irani

- retweets: 301, favorites: 141 (03/31/2021 10:26:53)

- links: [abs](https://arxiv.org/abs/2103.15545) | [pdf](https://arxiv.org/pdf/2103.15545)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Single image generative models perform synthesis and manipulation tasks by capturing the distribution of patches within a single image. The classical (pre Deep Learning) prevailing approaches for these tasks are based on an optimization process that maximizes patch similarity between the input and generated output. Recently, however, Single Image GANs were introduced both as a superior solution for such manipulation tasks, but also for remarkable novel generative tasks. Despite their impressiveness, single image GANs require long training time (usually hours) for each image and each task. They often suffer from artifacts and are prone to optimization issues such as mode collapse. In this paper, we show that all of these tasks can be performed without any training, within several seconds, in a unified, surprisingly simple framework. We revisit and cast the "good-old" patch-based methods into a novel optimization-free framework. We start with an initial coarse guess, and then simply refine the details coarse-to-fine using patch-nearest-neighbor search. This allows generating random novel images better and much faster than GANs. We further demonstrate a wide range of applications, such as image editing and reshuffling, retargeting to different sizes, structural analogies, image collage and a newly introduced task of conditional inpainting. Not only is our method faster ($\times 10^3$-$\times 10^4$ than a GAN), it produces superior results (confirmed by quantitative and qualitative evaluation), less artifacts and more realistic global structure than any of the previous approaches (whether GAN-based or classical patch-based).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Models<br>pdf: <a href="https://t.co/7nZabHmvF8">https://t.co/7nZabHmvF8</a><br>abs: <a href="https://t.co/qC8qeoTq3k">https://t.co/qC8qeoTq3k</a><br>project page: <a href="https://t.co/Z6looXvoDb">https://t.co/Z6looXvoDb</a> <a href="https://t.co/9McfHBi9NO">pic.twitter.com/9McfHBi9NO</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376718554872045568?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Drop the GAN:<br>In Defense of Patches Nearest Neighbors as Single Image Generative Models<br><br>Niv Granot, <a href="https://twitter.com/AssafShocher?ref_src=twsrc%5Etfw">@AssafShocher</a>,  Ben Feinstein, Shai Bagon, Michal Irani  <a href="https://t.co/ZdPSAepGen">https://t.co/ZdPSAepGen</a><br><br>Patch-NN vs GAN for image generation.<br> <br>1/3 <a href="https://t.co/quHihj7X16">pic.twitter.com/quHihj7X16</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1376846458389946368?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Multi-Scale Vision Longformer: A New Vision Transformer for  High-Resolution Image Encoding

Pengchuan Zhang, Xiyang Dai, Jianwei Yang, Bin Xiao, Lu Yuan, Lei Zhang, Jianfeng Gao

- retweets: 324, favorites: 86 (03/31/2021 10:26:53)

- links: [abs](https://arxiv.org/abs/2103.15358) | [pdf](https://arxiv.org/pdf/2103.15358)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

This paper presents a new Vision Transformer (ViT) architecture Multi-Scale Vision Longformer, which significantly enhances the ViT of \cite{dosovitskiy2020image} for encoding high-resolution images using two techniques. The first is the multi-scale model structure, which provides image encodings at multiple scales with manageable computational cost. The second is the attention mechanism of vision Longformer, which is a variant of Longformer \cite{beltagy2020longformer}, originally developed for natural language processing, and achieves a linear complexity w.r.t. the number of input tokens. A comprehensive empirical study shows that the new ViT significantly outperforms several strong baselines, including the existing ViT models and their ResNet counterparts, and the Pyramid Vision Transformer from a concurrent work \cite{wang2021pyramid}, on a range of vision tasks, including image classification, object detection, and segmentation. The models and source code used in this study will be released to public soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding<br>pdf: <a href="https://t.co/pIDqRwr1CT">https://t.co/pIDqRwr1CT</a><br>abs: <a href="https://t.co/MvQvsFfmQZ">https://t.co/MvQvsFfmQZ</a> <a href="https://t.co/cZe5ZjEqKw">pic.twitter.com/cZe5ZjEqKw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376707384043782147?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS

Ye Jia, Heiga Zen, Jonathan Shen, Yu Zhang, Yonghui Wu

- retweets: 312, favorites: 91 (03/31/2021 10:26:54)

- links: [abs](https://arxiv.org/abs/2103.15060) | [pdf](https://arxiv.org/pdf/2103.15060)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper introduces PnG BERT, a new encoder model for neural TTS. This model is augmented from the original BERT model, by taking both phoneme and grapheme representations of text as input, as well as the word-level alignment between them. It can be pre-trained on a large text corpus in a self-supervised manner, and fine-tuned in a TTS task. Experimental results show that a neural TTS model using a pre-trained PnG BERT as its encoder yields more natural prosody and more accurate pronunciation than a baseline model using only phoneme input with no pre-training. Subjective side-by-side preference evaluations show that raters have no statistically significant preference between the speech synthesized using a PnG BERT and ground truth recordings from professional speakers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper from our team: <br><br>Ye Jia, Heiga Zen, Jonathan Shen, Yu Zhang, Yonghui Wu<br>&quot;PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS&quot;<br><br>Arxiv: <a href="https://t.co/1qrbkkqDyB">https://t.co/1qrbkkqDyB</a><br>Samples: <a href="https://t.co/RlUhmfEWKE">https://t.co/RlUhmfEWKE</a></p>&mdash; Heiga Zen (全 炳河) (@heiga_zen) <a href="https://twitter.com/heiga_zen/status/1376901591870017536?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. High-Fidelity and Arbitrary Face Editing

Yue Gao, Fangyun Wei, Jianmin Bao, Shuyang Gu, Dong Chen, Fang Wen, Zhouhui Lian

- retweets: 110, favorites: 68 (03/31/2021 10:26:54)

- links: [abs](https://arxiv.org/abs/2103.15814) | [pdf](https://arxiv.org/pdf/2103.15814)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Cycle consistency is widely used for face editing. However, we observe that the generator tends to find a tricky way to hide information from the original image to satisfy the constraint of cycle consistency, making it impossible to maintain the rich details (e.g., wrinkles and moles) of non-editing areas. In this work, we propose a simple yet effective method named HifaFace to address the above-mentioned problem from two perspectives. First, we relieve the pressure of the generator to synthesize rich details by directly feeding the high-frequency information of the input image into the end of the generator. Second, we adopt an additional discriminator to encourage the generator to synthesize rich details. Specifically, we apply wavelet transformation to transform the image into multi-frequency domains, among which the high-frequency parts can be used to recover the rich details. We also notice that a fine-grained and wider-range control for the attribute is of great importance for face editing. To achieve this goal, we propose a novel attribute regression loss. Powered by the proposed framework, we achieve high-fidelity and arbitrary face editing, outperforming other state-of-the-art approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">High-Fidelity and Arbitrary Face Editing<br>pdf: <a href="https://t.co/JhqJlPYgFt">https://t.co/JhqJlPYgFt</a><br>abs: <a href="https://t.co/Q8WbgHhqpC">https://t.co/Q8WbgHhqpC</a> <a href="https://t.co/zOBE5IUlPH">pic.twitter.com/zOBE5IUlPH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376763308586459138?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. NeMI: Unifying Neural Radiance Fields with Multiplane Images for Novel  View Synthesis

Jiaxin Li, Zijian Feng, Qi She, Henghui Ding, Changhu Wang, Gim Hee Lee

- retweets: 90, favorites: 85 (03/31/2021 10:26:54)

- links: [abs](https://arxiv.org/abs/2103.14910) | [pdf](https://arxiv.org/pdf/2103.14910)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper, we propose an approach to perform novel view synthesis and depth estimation via dense 3D reconstruction from a single image. Our NeMI unifies Neural radiance fields (NeRF) with Multiplane Images (MPI). Specifically, our NeMI is a general two-dimensional and image-conditioned extension of NeRF, and a continuous depth generalization of MPI. Given a single image as input, our method predicts a 4-channel image (RGB and volume density) at arbitrary depth values to jointly reconstruct the camera frustum and fill in occluded contents. The reconstructed and inpainted frustum can then be easily rendered into novel RGB or depth views using differentiable rendering. Extensive experiments on RealEstate10K, KITTI and Flowers Light Fields show that our NeMI outperforms state-of-the-art by a large margin in novel view synthesis. We also achieve competitive results in depth estimation on iBims-1 and NYU-v2 without annotated depth supervision. Project page available at https://vincentfung13.github.io/projects/nemi/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeMI: Unifying Neural Radiance Fields with Multiplane Images for Novel View Synthesis<br>pdf: <a href="https://t.co/jkRd5e7PY2">https://t.co/jkRd5e7PY2</a><br>abs: <a href="https://t.co/Dnebx7p10d">https://t.co/Dnebx7p10d</a> <a href="https://t.co/PDgEenJdI3">pic.twitter.com/PDgEenJdI3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376714334294962176?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. On the Adversarial Robustness of Visual Transformers

Rulin Shao, Zhouxing Shi, Jinfeng Yi, Pin-Yu Chen, Cho-Jui Hsieh

- retweets: 72, favorites: 89 (03/31/2021 10:26:54)

- links: [abs](https://arxiv.org/abs/2103.15670) | [pdf](https://arxiv.org/pdf/2103.15670)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Following the success in advancing natural language processing and understanding, transformers are expected to bring revolutionary changes to computer vision. This work provides the first and comprehensive study on the robustness of vision transformers (ViTs) against adversarial perturbations. Tested on various white-box and transfer attack settings, we find that ViTs possess better adversarial robustness when compared with convolutional neural networks (CNNs). We summarize the following main observations contributing to the improved robustness of ViTs:   1) Features learned by ViTs contain less low-level information and are more generalizable, which contributes to superior robustness against adversarial perturbations.   2) Introducing convolutional or tokens-to-token blocks for learning low-level features in ViTs can improve classification accuracy but at the cost of adversarial robustness.   3) Increasing the proportion of transformers in the model structure (when the model consists of both transformer and CNN blocks) leads to better robustness. But for a pure transformer model, simply increasing the size or adding layers cannot guarantee a similar effect.   4) Pre-training on larger datasets does not significantly improve adversarial robustness though it is critical for training ViTs.   5) Adversarial training is also applicable to ViT for training robust models.   Furthermore, feature visualization and frequency analysis are conducted for explanation. The results show that ViTs are less sensitive to high-frequency perturbations than CNNs and there is a high correlation between how well the model learns low-level features and its robustness against different frequency-based perturbations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Visual transformers (ViTs) are getting more attention, but how about their robustness against adversarial perturbations? Check out our findings and summaries.<br><br>And yes, ViTs are empirically more robust than CNNs; adversarial training works too!<br><br>Paper: <a href="https://t.co/eRWqy7MuwR">https://t.co/eRWqy7MuwR</a> <a href="https://t.co/gHxmItoWmh">pic.twitter.com/gHxmItoWmh</a></p>&mdash; Pin-Yu Chen (@pinyuchenTW) <a href="https://twitter.com/pinyuchenTW/status/1376705998325448709?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Generic Attention-model Explainability for Interpreting Bi-Modal and  Encoder-Decoder Transformers

Hila Chefer, Shir Gur, Lior Wolf

- retweets: 110, favorites: 38 (03/31/2021 10:26:54)

- links: [abs](https://arxiv.org/abs/2103.15679) | [pdf](https://arxiv.org/pdf/2103.15679)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers are increasingly dominating multi-modal reasoning tasks, such as visual question answering, achieving state-of-the-art results thanks to their ability to contextualize information using the self-attention and co-attention mechanisms. These attention modules also play a role in other computer vision tasks including object detection and image segmentation. Unlike Transformers that only use self-attention, Transformers with co-attention require to consider multiple attention maps in parallel in order to highlight the information that is relevant to the prediction in the model's input. In this work, we propose the first method to explain prediction by any Transformer-based architecture, including bi-modal Transformers and Transformers with co-attentions. We provide generic solutions and apply these to the three most commonly used of these architectures: (i) pure self-attention, (ii) self-attention combined with co-attention, and (iii) encoder-decoder attention. We show that our method is superior to all existing methods which are adapted from single modality explainability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers<br>pdf: <a href="https://t.co/B3KiLhQMph">https://t.co/B3KiLhQMph</a><br>abs: <a href="https://t.co/o4222aRQ5L">https://t.co/o4222aRQ5L</a><br>github: <a href="https://t.co/GGNlygHq9d">https://t.co/GGNlygHq9d</a> <a href="https://t.co/gUtnqdMWjE">pic.twitter.com/gUtnqdMWjE</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376711810095714308?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Alignment of Language Agents

Zachary Kenton, Tom Everitt, Laura Weidinger, Iason Gabriel, Vladimir Mikulik, Geoffrey Irving

- retweets: 100, favorites: 34 (03/31/2021 10:26:55)

- links: [abs](https://arxiv.org/abs/2103.14659) | [pdf](https://arxiv.org/pdf/2103.14659)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

For artificial intelligence to be beneficial to humans the behaviour of AI agents needs to be aligned with what humans want. In this paper we discuss some behavioural issues for language agents, arising from accidental misspecification by the system designer. We highlight some ways that misspecification can occur and discuss some behavioural issues that could arise from misspecification, including deceptive or manipulative language, and review some approaches for avoiding these issues.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Some discussion from DeepMind researchers about potential for language models to train on data containing outputs of other LMs. From this paper: <a href="https://t.co/XLMl6AL6lS">https://t.co/XLMl6AL6lS</a> <a href="https://t.co/jYvF2txueZ">https://t.co/jYvF2txueZ</a> <a href="https://t.co/m2CEOVUG7R">pic.twitter.com/m2CEOVUG7R</a></p>&mdash; Jack Clark (@jackclarkSF) <a href="https://twitter.com/jackclarkSF/status/1376992868657590272?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image  Classification

Chun-Fu Chen, Quanfu Fan, Rameswar Panda

- retweets: 90, favorites: 44 (03/31/2021 10:26:55)

- links: [abs](https://arxiv.org/abs/2103.14899) | [pdf](https://arxiv.org/pdf/2103.14899)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The recently developed vision transformer (ViT) has achieved promising results on image classification compared to convolutional neural networks. Inspired by this, in this paper, we study how to learn multi-scale feature representations in transformer models for image classification. To this end, we propose a dual-branch transformer to combine image patches (i.e., tokens in a transformer) of different sizes to produce stronger image features. Our approach processes small-patch and large-patch tokens with two separate branches of different computational complexity and these tokens are then fused purely by attention multiple times to complement each other. Furthermore, to reduce computation, we develop a simple yet effective token fusion module based on cross attention, which uses a single token for each branch as a query to exchange information with other branches. Our proposed cross-attention only requires linear time for both computational and memory complexity instead of quadratic time otherwise. Extensive experiments demonstrate that the proposed approach performs better than or on par with several concurrent works on vision transformer, in addition to efficient CNN models. For example, on the ImageNet1K dataset, with some architectural changes, our approach outperforms the recent DeiT by a large margin of 2\%

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification<br>pdf: <a href="https://t.co/XFx9Azl9Wx">https://t.co/XFx9Azl9Wx</a><br>abs: <a href="https://t.co/Tcx5VPI9o8">https://t.co/Tcx5VPI9o8</a> <a href="https://t.co/S63pQcg6Ir">pic.twitter.com/S63pQcg6Ir</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376709276144648195?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Hand tracking for immersive virtual reality: opportunities and  challenges

Gavin Buckingham

- retweets: 110, favorites: 18 (03/31/2021 10:26:55)

- links: [abs](https://arxiv.org/abs/2103.14853) | [pdf](https://arxiv.org/pdf/2103.14853)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Hand tracking has become an integral feature of recent generations of immersive virtual reality head-mounted displays. With the widespread adoption of this feature, hardware engineers and software developers are faced with an exciting array of opportunities and a number of challenges, mostly in relation to the human user. In this article, I outline what I see as the main possibilities for hand tracking to add value to immersive virtual reality as well as some of the potential challenges in the context of the psychology and neuroscience of the human user. It is hoped that this paper serves as a roadmap for the development of best practices in the field for the development of subsequent generations of hand tracking and virtual reality technologies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Just in time for <a href="https://twitter.com/IEEEVR?ref_src=twsrc%5Etfw">@IEEEVR</a>, I&#39;ve written a new paper called: Hand tracking for immersive virtual reality: opportunities and challenges<br><br>You can find this on arXiv<a href="https://t.co/9uqm2rIXwf">https://t.co/9uqm2rIXwf</a><br><br>[thread]</p>&mdash; Gavin Buckingham (@DrGBuckingham) <a href="https://twitter.com/DrGBuckingham/status/1376818540787941376?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. OLED: One-Class Learned Encoder-Decoder Network with Adversarial Context  Masking for Novelty Detection

John Taylor Jewell, Vahid Reza Khazaie, Yalda Mohsenzadeh

- retweets: 81, favorites: 34 (03/31/2021 10:26:55)

- links: [abs](https://arxiv.org/abs/2103.14953) | [pdf](https://arxiv.org/pdf/2103.14953)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Novelty detection is the task of recognizing samples that do not belong to the distribution of the target class. During training, the novelty class is absent, preventing the use of traditional classification approaches. Deep autoencoders have been widely used as a base of many unsupervised novelty detection methods. In particular, context autoencoders have been successful in the novelty detection task because of the more effective representations they learn by reconstructing original images from randomly masked images. However, a significant drawback of context autoencoders is that random masking fails to consistently cover important structures of the input image, leading to suboptimal representations - especially for the novelty detection task. In this paper, to optimize input masking, we have designed a framework consisting of two competing networks, a Mask Module and a Reconstructor. The Mask Module is a convolutional autoencoder that learns to generate optimal masks that cover the most important parts of images. Alternatively, the Reconstructor is a convolutional encoder-decoder that aims to reconstruct unperturbed images from masked images. The networks are trained in an adversarial manner in which the Mask Module generates masks that are applied to images given to the Reconstructor. In this way, the Mask Module seeks to maximize the reconstruction error that the Reconstructor is minimizing. When applied to novelty detection, the proposed approach learns semantically richer representations compared to context autoencoders and enhances novelty detection at test time through more optimal masking. Novelty detection experiments on the MNIST and CIFAR-10 image datasets demonstrate the proposed approach's superiority over cutting-edge methods. In a further experiment on the UCSD video dataset for novelty detection, the proposed approach achieves state-of-the-art results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our new preprint with my graduate students John Jewell and <a href="https://twitter.com/vrkhazaie?ref_src=twsrc%5Etfw">@vrkhazaie</a> on novelty/anomaly detection: <a href="https://t.co/Woc2qWFeQ7">https://t.co/Woc2qWFeQ7</a></p>&mdash; Yalda Mohsenzadeh (@Yalda_Mhz) <a href="https://twitter.com/Yalda_Mhz/status/1376747324362387464?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. GNeRF: GAN-based Neural Radiance Field without Posed Camera

Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, Jingyi Yu

- retweets: 56, favorites: 46 (03/31/2021 10:26:55)

- links: [abs](https://arxiv.org/abs/2103.15606) | [pdf](https://arxiv.org/pdf/2103.15606)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce GNeRF, a framework to marry Generative Adversarial Networks (GAN) with Neural Radiance Field reconstruction for the complex scenarios with unknown and even randomly initialized camera poses. Recent NeRF-based advances have gained popularity for remarkable realistic novel view synthesis. However, most of them heavily rely on accurate camera poses estimation, while few recent methods can only optimize the unknown camera poses in roughly forward-facing scenes with relatively short camera trajectories and require rough camera poses initialization. Differently, our GNeRF only utilizes randomly initialized poses for complex outside-in scenarios. We propose a novel two-phases end-to-end framework. The first phase takes the use of GANs into the new realm for coarse camera poses and radiance fields jointly optimization, while the second phase refines them with additional photometric loss. We overcome local minima using a hybrid and iterative optimization scheme. Extensive experiments on a variety of synthetic and natural scenes demonstrate the effectiveness of GNeRF. More impressively, our approach outperforms the baselines favorably in those scenes with repeated patterns or even low textures that are regarded as extremely challenging before.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GNeRF: GAN-based Neural Radiance Field without Posed Camera<br>pdf: <a href="https://t.co/bR5nCwoeKw">https://t.co/bR5nCwoeKw</a><br>abs: <a href="https://t.co/Xd7rTKgkWc">https://t.co/Xd7rTKgkWc</a> <a href="https://t.co/KJJL1J8w13">pic.twitter.com/KJJL1J8w13</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376715132324225026?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. PixelTransformer: Sample Conditioned Signal Generation

Shubham Tulsiani, Abhinav Gupta

- retweets: 30, favorites: 67 (03/31/2021 10:26:56)

- links: [abs](https://arxiv.org/abs/2103.15813) | [pdf](https://arxiv.org/pdf/2103.15813)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a generative model that can infer a distribution for the underlying spatial signal conditioned on sparse samples e.g. plausible images given a few observed pixels. In contrast to sequential autoregressive generative models, our model allows conditioning on arbitrary samples and can answer distributional queries for any location. We empirically validate our approach across three image datasets and show that we learn to generate diverse and meaningful samples, with the distribution variance reducing given more observed pixels. We also show that our approach is applicable beyond images and can allow generating other types of spatial outputs e.g. polynomials, 3D shapes, and videos.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PixelTransformer: Sample Conditioned Signal Generation<br>pdf: <a href="https://t.co/ncExK3yiid">https://t.co/ncExK3yiid</a><br>abs: <a href="https://t.co/asdtHwlc8T">https://t.co/asdtHwlc8T</a><br>project page: <a href="https://t.co/0FNfQCy0yA">https://t.co/0FNfQCy0yA</a> <a href="https://t.co/JTNNLapVWn">pic.twitter.com/JTNNLapVWn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376708340168921088?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. Towards High Fidelity Monocular Face Reconstruction with Rich  Reflectance using Self-supervised Learning and Ray Tracing

Abdallah Dib, Cedric Thebault, Junghyun Ahn, Philippe-Henri Gosselin, Christian Theobalt, Louis Chevallier

- retweets: 42, favorites: 32 (03/31/2021 10:26:56)

- links: [abs](https://arxiv.org/abs/2103.15432) | [pdf](https://arxiv.org/pdf/2103.15432)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Robust face reconstruction from monocular image in general lighting conditions is challenging. Methods combining deep neural network encoders with differentiable rendering have opened up the path for very fast monocular reconstruction of geometry, lighting and reflectance. They can also be trained in self-supervised manner for increased robustness and better generalization. However, their differentiable rasterization based image formation models, as well as underlying scene parameterization, limit them to Lambertian face reflectance and to poor shape details. More recently, ray tracing was introduced for monocular face reconstruction within a classic optimization-based framework and enables state-of-the art results. However optimization-based approaches are inherently slow and lack robustness. In this paper, we build our work on the aforementioned approaches and propose a new method that greatly improves reconstruction quality and robustness in general scenes. We achieve this by combining a CNN encoder with a differentiable ray tracer, which enables us to base the reconstruction on much more advanced personalized diffuse and specular albedos, a more sophisticated illumination model and a plausible representation of self-shadows. This enables to take a big leap forward in reconstruction quality of shape, appearance and lighting even in scenes with difficult illumination. With consistent face attributes reconstruction, our method leads to practical applications such as relighting and self-shadows removal. Compared to state-of-the-art methods, our results show improved accuracy and validity of the approach.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing<br>pdf: <a href="https://t.co/wa2cJc8MdF">https://t.co/wa2cJc8MdF</a><br>abs: <a href="https://t.co/aTjNp5NCce">https://t.co/aTjNp5NCce</a> <a href="https://t.co/tJBmz5u3Fq">pic.twitter.com/tJBmz5u3Fq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376723033075879937?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. LayoutParser: A Unified Toolkit for Deep Learning Based Document Image  Analysis

Zejiang Shen, Ruochen Zhang, Melissa Dell, Benjamin Charles Germain Lee, Jacob Carlson, Weining Li

- retweets: 56, favorites: 15 (03/31/2021 10:26:56)

- links: [abs](https://arxiv.org/abs/2103.15348) | [pdf](https://arxiv.org/pdf/2103.15348)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Recent advances in document image analysis (DIA) have been primarily driven by the application of neural networks. Ideally, research outcomes could be easily deployed in production and extended for further investigation. However, various factors like loosely organized codebases and sophisticated model configurations complicate the easy reuse of important innovations by a wide audience. Though there have been on-going efforts to improve reusability and simplify deep learning (DL) model development in disciplines like natural language processing and computer vision, none of them are optimized for challenges in the domain of DIA. This represents a major gap in the existing toolkit, as DIA is central to academic research across a wide range of disciplines in the social sciences and humanities. This paper introduces layoutparser, an open-source library for streamlining the usage of DL in DIA research and applications. The core layoutparser library comes with a set of simple and intuitive interfaces for applying and customizing DL models for layout detection, character recognition, and many other document processing tasks. To promote extensibility, layoutparser also incorporates a community platform for sharing both pre-trained models and full document digitization pipelines. We demonstrate that layoutparser is helpful for both lightweight and large-scale digitization pipelines in real-word use cases. The library is publicly available at https://layout-parser.github.io/.




# 23. Learning Generative Models of Textured 3D Meshes from Real-World Images

Dario Pavllo, Jonas Kohler, Thomas Hofmann, Aurelien Lucchi

- retweets: 30, favorites: 36 (03/31/2021 10:26:56)

- links: [abs](https://arxiv.org/abs/2103.15627) | [pdf](https://arxiv.org/pdf/2103.15627)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent advances in differentiable rendering have sparked an interest in learning generative models of textured 3D meshes from image collections. These models natively disentangle pose and appearance, enable downstream applications in computer graphics, and improve the ability of generative models to understand the concept of image formation. Although there has been prior work on learning such models from collections of 2D images, these approaches require a delicate pose estimation step that exploits annotated keypoints, thereby restricting their applicability to a few specific datasets. In this work, we propose a GAN framework for generating textured triangle meshes without relying on such annotations. We show that the performance of our approach is on par with prior work that relies on ground-truth keypoints, and more importantly, we demonstrate the generality of our method by setting new baselines on a larger set of categories from ImageNet - for which keypoints are not available - without any class-specific hyperparameter tuning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Generative Models of Textured 3D Meshes from Real-World Images<br>pdf: <a href="https://t.co/1sB3e87V3x">https://t.co/1sB3e87V3x</a><br>abs: <a href="https://t.co/9hrZ68k896">https://t.co/9hrZ68k896</a> <a href="https://t.co/Dyejy1Vy2t">pic.twitter.com/Dyejy1Vy2t</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376719224199127040?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. Checkerboard Context Model for Efficient Learned Image Compression

Dailan He, Yaoyan Zheng, Baocheng Sun, Yan Wang, Hongwei Qin

- retweets: 36, favorites: 25 (03/31/2021 10:26:56)

- links: [abs](https://arxiv.org/abs/2103.15306) | [pdf](https://arxiv.org/pdf/2103.15306)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

For learned image compression, the autoregressive context model is proved effective in improving the rate-distortion (RD) performance. Because it helps remove spatial redundancies among latent representations. However, the decoding process must be done in a strict scan order, which breaks the parallelization. We propose a parallelizable checkerboard context model (CCM) to solve the problem. Our two-pass checkerboard context calculation eliminates such limitations on spatial locations by re-organizing the decoding order. Speeding up the decoding process more than 40 times in our experiments, it achieves significantly improved computational efficiency with almost the same rate-distortion performance. To the best of our knowledge, this is the first exploration on parallelization-friendly spatial context model for learned image compression.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Checkerboard Context Model for Efficient Learned Image Compression<br>pdf: <a href="https://t.co/ag4NG9pbom">https://t.co/ag4NG9pbom</a><br>abs: <a href="https://t.co/TyVNCvELnN">https://t.co/TyVNCvELnN</a> <a href="https://t.co/xznfEWFvdQ">pic.twitter.com/xznfEWFvdQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376713577072037891?ref_src=twsrc%5Etfw">March 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



