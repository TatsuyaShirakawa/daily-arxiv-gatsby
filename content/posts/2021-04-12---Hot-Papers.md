---
title: Hot Papers 2021-04-12
date: 2021-04-13T10:00:52.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. CutPaste: Self-Supervised Learning for Anomaly Detection and  Localization

Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister

- retweets: 4026, favorites: 306 (04/13/2021 10:00:52)

- links: [abs](https://arxiv.org/abs/2104.04015) | [pdf](https://arxiv.org/pdf/2104.04015)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We aim at constructing a high performance model for defect detection that detects unknown anomalous patterns of an image without anomalous data. To this end, we propose a two-stage framework for building anomaly detectors using normal training data only. We first learn self-supervised deep representations and then build a generative one-class classifier on learned representations. We learn representations by classifying normal data from the CutPaste, a simple data augmentation strategy that cuts an image patch and pastes at a random location of a large image. Our empirical study on MVTec anomaly detection dataset demonstrates the proposed algorithm is general to be able to detect various types of real-world defects. We bring the improvement upon previous arts by 3.1 AUCs when learning representations from scratch. By transfer learning on pretrained representations on ImageNet, we achieve a new state-of-theart 96.6 AUC. Lastly, we extend the framework to learn and extract representations from patches to allow localizing defective areas without annotations during training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CutPaste: Self-Supervised Learning for Anomaly Detection and Localization<br>pdf: <a href="https://t.co/JpUeDATWKh">https://t.co/JpUeDATWKh</a><br>abs: <a href="https://t.co/h6hFClmYOn">https://t.co/h6hFClmYOn</a><br>&quot;By transfer learning on pretrained representations on ImageNet, we achieve a new state-of-theart 96.6 AUC&quot; <a href="https://t.co/WRnuwhxZhR">pic.twitter.com/WRnuwhxZhR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381412371474874370?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Efficient Large-Scale Language Model Training on GPU Clusters

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia

- retweets: 2997, favorites: 424 (04/13/2021 10:00:52)

- links: [abs](https://arxiv.org/abs/2104.04473) | [pdf](https://arxiv.org/pdf/2104.04473)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Large language models have led to state-of-the-art accuracies across a range of tasks. However, training these large models efficiently is challenging for two reasons: a) GPU memory capacity is limited, making it impossible to fit large models on a single GPU or even on a multi-GPU server; and b) the number of compute operations required to train these models can result in unrealistically long training times. New methods of model parallelism such as tensor and pipeline parallelism have been proposed to address these challenges; unfortunately, naive usage leads to fundamental scaling issues at thousands of GPUs due to various reasons, e.g., expensive cross-node communication or idle periods waiting on other devices.   In this work, we show how to compose different types of parallelism methods (tensor, pipeline, and data paralleism) to scale to thousands of GPUs, achieving a two-order-of-magnitude increase in the sizes of models we can efficiently train compared to existing systems. We discuss various implementations of pipeline parallelism and propose a novel schedule that can improve throughput by more than 10% with comparable memory footprint compared to previously-proposed approaches. We quantitatively study the trade-offs between tensor, pipeline, and data parallelism, and provide intuition as to how to configure distributed training of a large model. The composition of these techniques allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs with achieved per-GPU throughput of 52% of peak; previous efforts to train similar-sized models achieve much lower throughput (36% of theoretical peak). Our code has been open-sourced at https://github.com/nvidia/megatron-lm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient Large-Scale Language Model Training on GPU Clusters<br><br>- Efficiently composes different types of parallelism methods to scale to thousands of GPUs. <br><br>- Achieves 502 PFLOP/s on 3072 GPUs at training a model w/ 1 trillion params. <br><br>abs: <a href="https://t.co/zbWBdGP799">https://t.co/zbWBdGP799</a> <a href="https://t.co/Ysr4lzDHrQ">pic.twitter.com/Ysr4lzDHrQ</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1381407690753142787?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Efficient Large-Scale Language Model Training on GPU Clusters<br>pdf: <a href="https://t.co/9eAtZ6YICl">https://t.co/9eAtZ6YICl</a><br>abs: <a href="https://t.co/aiX9JWRJBK">https://t.co/aiX9JWRJBK</a> <a href="https://t.co/uq4NH6AYwn">pic.twitter.com/uq4NH6AYwn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381409344336637955?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Protein sequence design with deep generative models

Zachary Wu, Kadina E. Johnston, Frances H. Arnold, Kevin K. Yang

- retweets: 1979, favorites: 246 (04/13/2021 10:00:53)

- links: [abs](https://arxiv.org/abs/2104.04457) | [pdf](https://arxiv.org/pdf/2104.04457)
- [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Protein engineering seeks to identify protein sequences with optimized properties. When guided by machine learning, protein sequence generation methods can draw on prior knowledge and experimental efforts to improve this process. In this review, we highlight recent applications of machine learning to generate protein sequences, focusing on the emerging field of deep generative methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A review of deep generative models for protein sequence engineering, by <a href="https://twitter.com/ZvxyWu?ref_src=twsrc%5Etfw">@ZvxyWu</a> <a href="https://twitter.com/kadinaj?ref_src=twsrc%5Etfw">@kadinaj</a> <a href="https://twitter.com/francesarnold?ref_src=twsrc%5Etfw">@francesarnold</a> and me! <a href="https://t.co/8ZigkmCEmf">https://t.co/8ZigkmCEmf</a> <a href="https://t.co/fxKnMpGrX4">pic.twitter.com/fxKnMpGrX4</a></p>&mdash; Kevin Yang 楊凱筌 (@KevinKaichuang) <a href="https://twitter.com/KevinKaichuang/status/1381583628241604611?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excellent new review out from <a href="https://twitter.com/ZvxyWu?ref_src=twsrc%5Etfw">@zvxywu</a> <a href="https://twitter.com/kadinaj?ref_src=twsrc%5Etfw">@kadinaj</a> <a href="https://twitter.com/francesarnold?ref_src=twsrc%5Etfw">@francesarnold</a> and <a href="https://twitter.com/KevinKaichuang?ref_src=twsrc%5Etfw">@KevinKaichuang</a> <br><br>Protein sequence design with deep generative models<a href="https://t.co/R4UenH7OeI">https://t.co/R4UenH7OeI</a></p>&mdash; Michael Retchin (@MichaelRetchin) <a href="https://twitter.com/MichaelRetchin/status/1381428735178645510?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. AlephBERT:A Hebrew Large Pre-Trained Language Model to Start-off your  Hebrew NLP Application With

Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Refael Shaked Greenfeld, Reut Tsarfaty

- retweets: 575, favorites: 140 (04/13/2021 10:00:53)

- links: [abs](https://arxiv.org/abs/2104.04052) | [pdf](https://arxiv.org/pdf/2104.04052)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Large Pre-trained Language Models (PLMs) have become ubiquitous in the development of language understanding technology and lie at the heart of many artificial intelligence advances. While advances reported for English using PLMs are unprecedented, reported advances using PLMs in Hebrew are few and far between. The problem is twofold. First, Hebrew resources available for training NLP models are not at the same order of magnitude as their English counterparts. Second, there are no accepted tasks and benchmarks to evaluate the progress of Hebrew PLMs on. In this work we aim to remedy both aspects. First, we present AlephBERT, a large pre-trained language model for Modern Hebrew, which is trained on larger vocabulary and a larger dataset than any Hebrew PLM before. Second, using AlephBERT we present new state-of-the-art results on multiple Hebrew tasks and benchmarks, including: Segmentation, Part-of-Speech Tagging, full Morphological Tagging, Named-Entity Recognition and Sentiment Analysis. We make our AlephBERT model publicly available, providing a single point of entry for the development of Hebrew NLP applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce *AlephBERT*, a Hebrew pretrained language model to start off your Hebrew NLP Application with.<br><br>*report*<a href="https://t.co/GNhIDsVqRl">https://t.co/GNhIDsVqRl</a><br><br>*demo*<a href="https://t.co/1XVPpWPVJd">https://t.co/1XVPpWPVJd</a><br><br>*model*<a href="https://t.co/PZN6TBDpkK">https://t.co/PZN6TBDpkK</a><br><br>The credit goes to my amazing research team at <a href="https://twitter.com/biunlp?ref_src=twsrc%5Etfw">@biunlp</a> <a href="https://twitter.com/OnlpLab?ref_src=twsrc%5Etfw">@OnlpLab</a></p>&mdash; Reut Tsarfaty (@rtsarfaty) <a href="https://twitter.com/rtsarfaty/status/1381562132723085315?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Appearance-Driven Automatic 3D Model Simplification

Jon Hasselgren, Jacob Munkberg, Jaakko Lehtinen, Miika Aittala, Samuli Laine

- retweets: 182, favorites: 70 (04/13/2021 10:00:53)

- links: [abs](https://arxiv.org/abs/2104.03989) | [pdf](https://arxiv.org/pdf/2104.03989)
- [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a suite of techniques for jointly optimizing triangle meshes and shading models to match the appearance of reference scenes. This capability has a number of uses, including appearance-preserving simplification of extremely complex assets, conversion between rendering systems, and even conversion between geometric scene representations.   We follow and extend the classic analysis-by-synthesis family of techniques: enabled by a highly efficient differentiable renderer and modern nonlinear optimization algorithms, our results are driven to minimize the image-space difference to the target scene when rendered in similar viewing and lighting conditions. As the only signals driving the optimization are differences in rendered images, the approach is highly general and versatile: it easily supports many different forward rendering models such as normal mapping, spatially-varying BRDFs, displacement mapping, etc. Supervision through images only is also key to the ability to easily convert between rendering systems and scene representations.   We output triangle meshes with textured materials to ensure that the models render efficiently on modern graphics hardware and benefit from, e.g., hardware-accelerated rasterization, ray tracing, and filtered texture lookups. Our system is integrated in a small Python code base, and can be applied at high resolutions and on large models. We describe several use cases, including mesh decimation, level of detail generation, seamless mesh filtering and approximations of aggregate geometry.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Appearance-Driven Automatic 3D Model Simplification<br>pdf: <a href="https://t.co/v8tXzTZ7DF">https://t.co/v8tXzTZ7DF</a><br>abs: <a href="https://t.co/K4WRZruDTp">https://t.co/K4WRZruDTp</a> <a href="https://t.co/dfI2CEcItW">pic.twitter.com/dfI2CEcItW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381429197248299008?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. SI-Score: An image dataset for fine-grained analysis of robustness to  object location, rotation and size

Jessica Yung, Rob Romijnders, Alexander Kolesnikov, Lucas Beyer, Josip Djolonga, Neil Houlsby, Sylvain Gelly, Mario Lucic, Xiaohua Zhai

- retweets: 100, favorites: 52 (04/13/2021 10:00:53)

- links: [abs](https://arxiv.org/abs/2104.04191) | [pdf](https://arxiv.org/pdf/2104.04191)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Before deploying machine learning models it is critical to assess their robustness. In the context of deep neural networks for image understanding, changing the object location, rotation and size may affect the predictions in non-trivial ways. In this work we perform a fine-grained analysis of robustness with respect to these factors of variation using SI-Score, a synthetic dataset. In particular, we investigate ResNets, Vision Transformers and CLIP, and identify interesting qualitative differences between these.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SI-Score: An image dataset for fine-grained analysis of robustness to object location, rotation and size<br>pdf: <a href="https://t.co/7rEmfRQSGc">https://t.co/7rEmfRQSGc</a><br>abs: <a href="https://t.co/dJWuWmL4Hi">https://t.co/dJWuWmL4Hi</a><br>github: <a href="https://t.co/akJjNh7rGP">https://t.co/akJjNh7rGP</a> <a href="https://t.co/JGggiQhXQx">pic.twitter.com/JGggiQhXQx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381407371302334465?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Replay in Deep Learning: Current Approaches and Missing Biological  Elements

Tyler L. Hayes, Giri P. Krishnan, Maxim Bazhenov, Hava T. Siegelmann, Terrence J. Sejnowski, Christopher Kanan

- retweets: 97, favorites: 47 (04/13/2021 10:00:54)

- links: [abs](https://arxiv.org/abs/2104.04132) | [pdf](https://arxiv.org/pdf/2104.04132)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Replay is the reactivation of one or more neural patterns, which are similar to the activation patterns experienced during past waking experiences. Replay was first observed in biological neural networks during sleep, and it is now thought to play a critical role in memory formation, retrieval, and consolidation. Replay-like mechanisms have been incorporated into deep artificial neural networks that learn over time to avoid catastrophic forgetting of previous knowledge. Replay algorithms have been successfully used in a wide range of deep learning methods within supervised, unsupervised, and reinforcement learning paradigms. In this paper, we provide the first comprehensive comparison between replay in the mammalian brain and replay in artificial neural networks. We identify multiple aspects of biological replay that are missing in deep learning systems and hypothesize how they could be utilized to improve artificial neural networks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Replay in Deep Learning: Current Approaches and Missing Biological Elements<br>pdf: <a href="https://t.co/sZeaLnxwST">https://t.co/sZeaLnxwST</a><br>abs: <a href="https://t.co/NGPElGyBzK">https://t.co/NGPElGyBzK</a><br>&quot;In this paper, we provide the first comprehensive comparison between replay in the mammalian brain and replay in artificial neural networks&quot; <a href="https://t.co/sFUr7L4epp">pic.twitter.com/sFUr7L4epp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381419974133219334?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. An Empirical Comparison of Instance Attribution Methods for NLP

Pouya Pezeshkpour, Sarthak Jain, Byron C. Wallace, Sameer Singh

- retweets: 62, favorites: 47 (04/13/2021 10:00:54)

- links: [abs](https://arxiv.org/abs/2104.04128) | [pdf](https://arxiv.org/pdf/2104.04128)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Widespread adoption of deep models has motivated a pressing need for approaches to interpret network outputs and to facilitate model debugging. Instance attribution methods constitute one means of accomplishing these goals by retrieving training instances that (may have) led to a particular prediction. Influence functions (IF; Koh and Liang 2017) provide machinery for doing this by quantifying the effect that perturbing individual train instances would have on a specific test prediction. However, even approximating the IF is computationally expensive, to the degree that may be prohibitive in many cases. Might simpler approaches (e.g., retrieving train examples most similar to a given test point) perform comparably? In this work, we evaluate the degree to which different potential instance attribution agree with respect to the importance of training samples. We find that simple retrieval methods yield training instances that differ from those identified via gradient-based methods (such as IFs), but that nonetheless exhibit desirable characteristics similar to more complex attribution methods. Code for all methods and experiments in this paper is available at: https://github.com/successar/instance_attributions_NLP.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are you interested in instance attribution methods for NLP tasks? How feasible are these methods for large pre-trained language models such as BERT? Can these methods actually outperform their simpler counterparts? See our <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> paper. (1/n) <a href="https://t.co/bBBX6dHFwS">https://t.co/bBBX6dHFwS</a></p>&mdash; Pouya Pezeshkpour (@PPezeshkpour) <a href="https://twitter.com/PPezeshkpour/status/1381679854773399552?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. It's All About The Cards: Sharing on Social Media Probably Encouraged  HTML Metadata Growth

Shawn M. Jones, Valentina Neblitt-Jones, Michele C. Weigle, Martin Klein, Michael L. Nelson

- retweets: 90, favorites: 8 (04/13/2021 10:00:54)

- links: [abs](https://arxiv.org/abs/2104.04116) | [pdf](https://arxiv.org/pdf/2104.04116)
- [cs.DL](https://arxiv.org/list/cs.DL/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

In a perfect world, all articles consistently contain sufficient metadata to describe the resource. We know this is not the reality, so we are motivated to investigate the evolution of the metadata that is present when authors and publishers supply their own. Because applying metadata takes time, we recognize that each news article author has a limited metadata budget with which to spend their time and effort. How are they spending this budget? What are the top metadata categories in use? How did they grow over time? What purpose do they serve? We also recognize that not all metadata fields are used equally. What is the growth of individual fields over time? Which fields experienced the fastest adoption? In this paper, we review 227,726 HTML news articles from 29 outlets captured by the Internet Archive between 1998 and 2016. Upon reviewing the metadata fields in each article, we discovered that 2010 began a metadata renaissance as publishers embraced metadata for improved search engine ranking, search engine tracking, social media tracking, and social media sharing. When analyzing individual fields, we find that one application of metadata stands out above all others: social cards -- the cards generated by platforms like Twitter when one shares a URL. Once a metadata standard was established for cards in 2010, its fields were adopted by 20% of articles in the first year and reached more than 95% adoption by 2016. This rate of adoption surpasses efforts like Schema.org and Dublin Core by a fair margin. When confronted with these results on how news publishers spend their metadata budget, we must conclude that it is all about the cards.




# 10. Plug-and-Blend: A Framework for Controllable Story Generation with  Blended Control Codes

Zhiyu Lin, Mark Riedl

- retweets: 28, favorites: 49 (04/13/2021 10:00:54)

- links: [abs](https://arxiv.org/abs/2104.04039) | [pdf](https://arxiv.org/pdf/2104.04039)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

We describe a Plug-and-Play controllable language generation framework, Plug-and-Blend, that allows a human user to input multiple control codes (topics). In the context of automated story generation, this allows a human user lose or fine grained control of the topics that will appear in the generated story, and can even allow for overlapping, blended topics. We show that our framework, working with different generation models, controls the generation towards given continuous-weighted control codes while keeping the generated sentences fluent, demonstrating strong blending capability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Plug-and-Blend: A Framework for Controllable Story Generation with Blended Control Codes<br>pdf: <a href="https://t.co/pvlpBMUwOY">https://t.co/pvlpBMUwOY</a><br>abs: <a href="https://t.co/2zxsmLvCLg">https://t.co/2zxsmLvCLg</a> <a href="https://t.co/QjaZQXzczH">pic.twitter.com/QjaZQXzczH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381445428772802560?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Have you dreamed of enhancing your existing text generation models with controlling knobs on topics and styles, like walking in a latent space, WITHOUT any fine-tuning/added layer/structure changes?<br>Our new work is now available on Arxiv! <a href="https://t.co/P8iNwnjXGJ">https://t.co/P8iNwnjXGJ</a> <a href="https://twitter.com/mark_riedl?ref_src=twsrc%5Etfw">@mark_riedl</a> <a href="https://t.co/mZPBUKdaz3">pic.twitter.com/mZPBUKdaz3</a></p>&mdash; Zhiyu Lin (@xxbidiao) <a href="https://twitter.com/xxbidiao/status/1381676069208530945?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Auxiliary Tasks and Exploration Enable ObjectNav

Joel Ye, Dhruv Batra, Abhishek Das, Erik Wijmans

- retweets: 36, favorites: 26 (04/13/2021 10:00:54)

- links: [abs](https://arxiv.org/abs/2104.04112) | [pdf](https://arxiv.org/pdf/2104.04112)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

ObjectGoal Navigation (ObjectNav) is an embodied task wherein agents are to navigate to an object instance in an unseen environment. Prior works have shown that end-to-end ObjectNav agents that use vanilla visual and recurrent modules, e.g. a CNN+RNN, perform poorly due to overfitting and sample inefficiency. This has motivated current state-of-the-art methods to mix analytic and learned components and operate on explicit spatial maps of the environment. We instead re-enable a generic learned agent by adding auxiliary learning tasks and an exploration reward. Our agents achieve 24.5% success and 8.1% SPL, a 37% and 8% relative improvement over prior state-of-the-art, respectively, on the Habitat ObjectNav Challenge. From our analysis, we propose that agents will act to simplify their visual inputs so as to smooth their RNN dynamics, and that auxiliary tasks reduce overfitting by minimizing effective RNN dimensionality; i.e. a performant ObjectNav agent that must maintain coherent plans over long horizons does so by learning smooth, low-dimensional recurrent dynamics. Site: https://joel99.github.io/objectnav/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Auxiliary Tasks and Exploration Enable ObjectGoal Navigation<br>pdf: <a href="https://t.co/KA3fH37oDz">https://t.co/KA3fH37oDz</a><br>abs: <a href="https://t.co/DC73KEhW4T">https://t.co/DC73KEhW4T</a><br>project page: <a href="https://t.co/Feac5JOibb">https://t.co/Feac5JOibb</a><br>github: <a href="https://t.co/dHSy7G1a46">https://t.co/dHSy7G1a46</a> <a href="https://t.co/NRddmiKnnZ">pic.twitter.com/NRddmiKnnZ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1381477260264402944?ref_src=twsrc%5Etfw">April 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



