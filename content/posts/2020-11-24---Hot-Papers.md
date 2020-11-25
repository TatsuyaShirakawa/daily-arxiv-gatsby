---
title: Hot Papers 2020-11-24
date: 2020-11-25T12:34:19.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. Enriching ImageNet with Human Similarity Judgments and Psychological  Embeddings

Brett D. Roads, Bradley C. Love

- retweets: 4864, favorites: 230 (11/25/2020 12:34:19)

- links: [abs](https://arxiv.org/abs/2011.11015) | [pdf](https://arxiv.org/pdf/2011.11015)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Advances in object recognition flourished in part because of the availability of high-quality datasets and associated benchmarks. However, these benchmarks---such as ILSVRC---are relatively task-specific, focusing predominately on predicting class labels. We introduce a publicly-available dataset that embodies the task-general capabilities of human perception and reasoning. The Human Similarity Judgments extension to ImageNet (ImageNet-HSJ) is composed of human similarity judgments that supplement the ILSVRC validation set. The new dataset supports a range of task and performance metrics, including the evaluation of unsupervised learning algorithms. We demonstrate two methods of assessment: using the similarity judgments directly and using a psychological embedding trained on the similarity judgments. This embedding space contains an order of magnitude more points (i.e., images) than previous efforts based on human judgments. Scaling to the full 50,000 image set was made possible through a selective sampling process that used variational Bayesian inference and model ensembles to sample aspects of the embedding space that were most uncertain. This methodological innovation not only enables scaling, but should also improve the quality of solutions by focusing sampling where it is needed. To demonstrate the utility of ImageNet-HSJ, we used the similarity ratings and the embedding space to evaluate how well several popular models conform to human similarity judgments. One finding is that more complex models that perform better on task-specific benchmarks do not better conform to human semantic judgments. In addition to the human similarity judgments, pre-trained psychological embeddings and code for inferring variational embeddings are made publicly available. Collectively, ImageNet-HSJ assets support the appraisal of internal representations and the development of more human-like models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing the embedding space you didn&#39;t know you needed: Human similarity judgments for the entire ImageNet (50k images) validation set. Perfect for evaluating representations, including unsupervised models. It&#39;s already bearing fruit, w <a href="https://twitter.com/BDRoads?ref_src=twsrc%5Etfw">@BDRoads</a> <a href="https://t.co/Cl8iTdcHqj">https://t.co/Cl8iTdcHqj</a> (1/3) <a href="https://t.co/0zfYNjhuom">pic.twitter.com/0zfYNjhuom</a></p>&mdash; Bradley Love (@ProfData) <a href="https://twitter.com/ProfData/status/1331196418917347328?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. An Effective Anti-Aliasing Approach for Residual Networks

Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Nicolas Le Roux, Ross Goroshin

- retweets: 462, favorites: 115 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.10675) | [pdf](https://arxiv.org/pdf/2011.10675)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image pre-processing in the frequency domain has traditionally played a vital role in computer vision and was even part of the standard pipeline in the early days of deep learning. However, with the advent of large datasets, many practitioners concluded that this was unnecessary due to the belief that these priors can be learned from the data itself. Frequency aliasing is a phenomenon that may occur when sub-sampling any signal, such as an image or feature map, causing distortion in the sub-sampled output. We show that we can mitigate this effect by placing non-trainable blur filters and using smooth activation functions at key locations, particularly where networks lack the capacity to learn them. These simple architectural changes lead to substantial improvements in out-of-distribution generalization on both image classification under natural corruptions on ImageNet-C [10] and few-shot learning on Meta-Dataset [17], without introducing additional trainable parameters and using the default hyper-parameters of open source codebases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Nicolas Le Roux, Ross Goroshin, An Effective Anti-Aliasing Approach for Residual Networks, arXiv, 2020.<a href="https://t.co/ByjG4trqgf">https://t.co/ByjG4trqgf</a> <a href="https://t.co/QTJJrGHnTK">pic.twitter.com/QTJJrGHnTK</a></p>&mdash; Kosta Derpanis (@CSProfKGD) <a href="https://twitter.com/CSProfKGD/status/1331131867370811392?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. CoMatch: Semi-supervised Learning with Contrastive Graph Regularization

Junnan Li, Caiming Xiong, Steven Hoi

- retweets: 342, favorites: 96 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.11183) | [pdf](https://arxiv.org/pdf/2011.11183)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Semi-supervised learning has been an effective paradigm for leveraging unlabeled data to reduce the reliance on labeled data. We propose CoMatch, a new semi-supervised learning method that unifies dominant approaches and addresses their limitations. CoMatch jointly learns two representations of the training data, their class probabilities and low-dimensional embeddings. The two representations interact with each other to jointly evolve. The embeddings impose a smoothness constraint on the class probabilities to improve the pseudo-labels, whereas the pseudo-labels regularize the structure of the embeddings through graph-based contrastive learning. CoMatch achieves state-of-the-art performance on multiple datasets. It achieves ~20% accuracy improvement on the label-scarce CIFAR-10 and STL-10. On ImageNet with 1% labels, CoMatch achieves a top-1 accuracy of 66.0%, outperforming FixMatch by 12.6%. The accuracy further increases to 67.1% with self-supervised pre-training. Furthermore, CoMatch achieves better representation learning performance on downstream tasks, outperforming both supervised learning and self-supervised learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to introduce CoMatch, our new semi-supervised learning method! CoMatch jointly learns class probability and image representation with graph-based contrastive learning. <a href="https://twitter.com/CaimingXiong?ref_src=twsrc%5Etfw">@CaimingXiong</a> <a href="https://twitter.com/stevenhoi?ref_src=twsrc%5Etfw">@stevenhoi</a> <br>Blog: <a href="https://t.co/jiHVtIaeB7">https://t.co/jiHVtIaeB7</a><br>Paper: <a href="https://t.co/XehgJEEmS9">https://t.co/XehgJEEmS9</a></p>&mdash; Li Junnan (@LiJunnan0409) <a href="https://twitter.com/LiJunnan0409/status/1331102851637022720?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Gradualizing the Calculus of Inductive Constructions

Meven Lennon-Bertrand, Kenji Maillard, Nicolas Tabareau, Éric Tanter

- retweets: 240, favorites: 61 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.10618) | [pdf](https://arxiv.org/pdf/2011.10618)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

Acknowledging the ordeal of a fully formal development in a proof assistant such as Coq, we investigate gradual variations on the Calculus of Inductive Construction (CIC) for swifter prototyping with imprecise types and terms. We observe, with a no-go theorem, a crucial tradeoff between graduality and the key properties of normalization and closure of universes under dependent product that CIC enjoys. Beyond this Fire Triangle of Graduality, we explore the gradualization of CIC with three different compromises, each relaxing one edge of the Fire Triangle. We develop a parametrized presentation of Gradual CIC that encompasses all three variations, and develop their metatheory. We first present a bidirectional elaboration of Gradual CIC to a dependently-typed cast calculus, which elucidates the interrelation between typing, conversion, and the gradual guarantees. We use a syntactic model into CIC to inform the design of a safe, confluent reduction, and establish, when applicable, normalization. We also study the stronger notion of graduality as embedding-projection pairs formulated by New and Ahmed, using appropriate semantic model constructions. This work informs and paves the way towards the development of malleable proof assistants and dependently-typed programming languages.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">If you&#39;re interested in gradual typing and type theory, you might have wondered what it means to gradualize the Calculus of Inductive Constructions... Check out <a href="https://t.co/Q90X6eiH9K">https://t.co/Q90X6eiH9K</a></p>&mdash; Éric Tanter (@etanter) <a href="https://twitter.com/etanter/status/1331199981336879104?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Pose2Pose: 3D Positional Pose-Guided 3D Rotational Pose Prediction for  Expressive 3D Human Pose and Mesh Estimation

Gyeongsik Moon, Kyoung Mu Lee

- retweets: 198, favorites: 82 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.11534) | [pdf](https://arxiv.org/pdf/2011.11534)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Previous 3D human pose and mesh estimation methods mostly rely on only global image feature to predict 3D rotations of human joints (i.e., 3D rotational pose) from an input image. However, local features on the position of human joints (i.e., positional pose) can provide joint-specific information, which is essential to understand human articulation. To effectively utilize both local and global features, we present Pose2Pose, a 3D positional pose-guided 3D rotational pose prediction network, along with a positional pose-guided pooling and joint-specific graph convolution. The positional pose-guided pooling extracts useful joint-specific local and global features. Also, the joint-specific graph convolution effectively processes the joint-specific features by learning joint-specific characteristics and different relationships between different joints. We use Pose2Pose for expressive 3D human pose and mesh estimation and show that it outperforms all previous part-specific and expressive methods by a large margin. The codes will be publicly available.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pose2Pose: 3D Positional Pose-Guided 3D Rotational Pose Prediction for Expressive 3D Human Pose and Mesh Estimation<br>pdf: <a href="https://t.co/3Knl31BVOC">https://t.co/3Knl31BVOC</a><br>abs: <a href="https://t.co/f4tmRDEaNp">https://t.co/f4tmRDEaNp</a> <a href="https://t.co/VeuxiXKOEw">pic.twitter.com/VeuxiXKOEw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331106137488363520?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. HDR Environment Map Estimation for Real-Time Augmented Reality

Gowri Somanath, Daniel Kurz

- retweets: 110, favorites: 40 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.10687) | [pdf](https://arxiv.org/pdf/2011.10687)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a method to estimate an HDR environment map from a narrow field-of-view LDR camera image in real-time. This enables perceptually appealing reflections and shading on virtual objects of any material finish, from mirror to diffuse, rendered into a real physical environment using augmented reality. Our method is based on our efficient convolutional neural network architecture, EnvMapNet, trained end-to-end with two novel losses, ProjectionLoss for the generated image, and ClusterLoss for adversarial training. Through qualitative and quantitative comparison to state-of-the-art methods, we demonstrate that our algorithm reduces the directional error of estimated light sources by more than 50%, and achieves 3.7 times lower Frechet Inception Distance (FID). We further showcase a mobile application that is able to run our neural network model in under 9 ms on an iPhone XS, and render in real-time, visually coherent virtual objects in previously unseen real-world environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HDR Environment Map Estimation for Real-Time Augmented Reality<br>pdf: <a href="https://t.co/MgCwsQOz30">https://t.co/MgCwsQOz30</a><br>abs: <a href="https://t.co/Mzt1ZAwiMY">https://t.co/Mzt1ZAwiMY</a> <a href="https://t.co/raJrhczW4h">pic.twitter.com/raJrhczW4h</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331065567818739715?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Rethinking Transformer-based Set Prediction for Object Detection

Zhiqing Sun, Shengcao Cao, Yiming Yang, Kris Kitani

- retweets: 90, favorites: 42 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.10881) | [pdf](https://arxiv.org/pdf/2011.10881)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

DETR is a recently proposed Transformer-based method which views object detection as a set prediction problem and achieves state-of-the-art performance but demands extra-long training time to converge. In this paper, we investigate the causes of the optimization difficulty in the training of DETR. Our examinations reveal several factors contributing to the slow convergence of DETR, primarily the issues with the Hungarian loss and the Transformer cross attention mechanism. To overcome these issues we propose two solutions, namely, TSP-FCOS (Transformer-based Set Prediction with FCOS) and TSP-RCNN (Transformer-based Set Prediction with RCNN). Experimental results show that the proposed methods not only converge much faster than the original DETR, but also significantly outperform DETR and other baselines in terms of detection accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking Transformer-based Set Prediction for Object Detection<br>pdf: <a href="https://t.co/2vAR7AEzu1">https://t.co/2vAR7AEzu1</a><br>abs: <a href="https://t.co/b8ucapBLFw">https://t.co/b8ucapBLFw</a> <a href="https://t.co/iNfd2OFP7C">pic.twitter.com/iNfd2OFP7C</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331063974054473731?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. PLOP: Learning without Forgetting for Continual Semantic Segmentation

Arthur Douillard, Yifu Chen, Arnaud Dapogny, Matthieu Cord

- retweets: 66, favorites: 22 (11/25/2020 12:34:20)

- links: [abs](https://arxiv.org/abs/2011.11390) | [pdf](https://arxiv.org/pdf/2011.11390)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep learning approaches are nowadays ubiquitously used to tackle computer vision tasks such as semantic segmentation, requiring large datasets and substantial computational power. Continual learning for semantic segmentation (CSS) is an emerging trend that consists in updating an old model by sequentially adding new classes. However, continual learning methods are usually prone to catastrophic forgetting. This issue is further aggravated in CSS where, at each step, old classes from previous iterations are collapsed into the background. In this paper, we propose Local POD, a multi-scale pooling distillation scheme that preserves long- and short-range spatial relationships at feature level. Furthermore, we design an entropy-based pseudo-labelling of the background w.r.t. classes predicted by the old model to deal with background shift and avoid catastrophic forgetting of the old classes. Our approach, called PLOP, significantly outperforms state-of-the-art methods in existing CSS scenarios, as well as in newly proposed challenging benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work from Y.Chen, A.Dapogny, <a href="https://twitter.com/quobbe?ref_src=twsrc%5Etfw">@quobbe</a>, and myself.<br><br>We tackle Continual Semantic Segmentation by introducing a novel distillation loss exploiting local &amp; global details, and an uncertainty-based pseudo-labeling handling background shift<br><br>(We are PLOP)<a href="https://t.co/p5yqUYLH22">https://t.co/p5yqUYLH22</a> <a href="https://t.co/2eAlFELYTV">pic.twitter.com/2eAlFELYTV</a></p>&mdash; Arthur Douillard (@Ar_Douillard) <a href="https://twitter.com/Ar_Douillard/status/1331222775881674752?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. RVCoreP-32IC: A high-performance RISC-V soft processor with an efficient  fetch unit supporting the compressed instructions

Takuto Kanamori, Hiromu Miyazaki, Kenji Kise

- retweets: 56, favorites: 25 (11/25/2020 12:34:21)

- links: [abs](https://arxiv.org/abs/2011.11246) | [pdf](https://arxiv.org/pdf/2011.11246)
- [cs.AR](https://arxiv.org/list/cs.AR/recent)

In this paper, we propose a high-performance RISC-V soft processor with an efficient fetch unit supporting the compressed instructions targeting on FPGA. The compressed instruction extension in RISC-V can reduce the program size by about 25%. But it needs a complicated logic for the instruction fetch unit and has a significant impact on performance. We propose an instruction fetch unit that supports the compressed instructions while exhibiting high performance. Furthermore, we propose a RISC-V soft processor using this unit. We implement this proposed processor in Verilog HDL and verify the behavior using Verilog simulation and an actual Xilinx Atrix-7 FPGA board. We compare the results of some benchmarks and the amount of hardware with related works. DMIPS, CoreMark value, and Embench value of the proposed processor achieved 42.5%, 41.1% and 21.3% higher performance than the related work, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have proposed a high-performance <a href="https://twitter.com/hashtag/RISCV?src=hash&amp;ref_src=twsrc%5Etfw">#RISCV</a> soft processor with an efficient fetch unit supporting the compressed instructions targeting on FPGA. <a href="https://t.co/AZIG8t4nuK">https://t.co/AZIG8t4nuK</a> <a href="https://t.co/uX0gzW0BiZ">pic.twitter.com/uX0gzW0BiZ</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1331215707313491968?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. SCGAN: Saliency Map-guided Colorization with Generative Adversarial  Network

Yuzhi Zhao, Lai-Man Po, Kwok-Wai Cheung, Wing-Yin Yu, Yasar Abbas Ur Rehman

- retweets: 42, favorites: 33 (11/25/2020 12:34:21)

- links: [abs](https://arxiv.org/abs/2011.11377) | [pdf](https://arxiv.org/pdf/2011.11377)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Given a grayscale photograph, the colorization system estimates a visually plausible colorful image. Conventional methods often use semantics to colorize grayscale images. However, in these methods, only classification semantic information is embedded, resulting in semantic confusion and color bleeding in the final colorized image. To address these issues, we propose a fully automatic Saliency Map-guided Colorization with Generative Adversarial Network (SCGAN) framework. It jointly predicts the colorization and saliency map to minimize semantic confusion and color bleeding in the colorized image. Since the global features from pre-trained VGG-16-Gray network are embedded to the colorization encoder, the proposed SCGAN can be trained with much less data than state-of-the-art methods to achieve perceptually reasonable colorization. In addition, we propose a novel saliency map-based guidance method. Branches of the colorization decoder are used to predict the saliency map as a proxy target. Moreover, two hierarchical discriminators are utilized for the generated colorization and saliency map, respectively, in order to strengthen visual perception performance. The proposed system is evaluated on ImageNet validation set. Experimental results show that SCGAN can generate more reasonable colorized images than state-of-the-art techniques.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network<br>pdf: <a href="https://t.co/2vDFMadJ1d">https://t.co/2vDFMadJ1d</a><br>abs: <a href="https://t.co/tXZD9AxV4N">https://t.co/tXZD9AxV4N</a><br>github: <a href="https://t.co/2xiQd2RJAa">https://t.co/2xiQd2RJAa</a> <a href="https://t.co/M2EkjhVMo2">pic.twitter.com/M2EkjhVMo2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331074787179958275?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. An Empirical Study of Representation Learning for Reinforcement Learning  in Healthcare

Taylor W. Killian, Haoran Zhang, Jayakumar Subramanian, Mehdi Fatemi, Marzyeh Ghassemi

- retweets: 30, favorites: 40 (11/25/2020 12:34:21)

- links: [abs](https://arxiv.org/abs/2011.11235) | [pdf](https://arxiv.org/pdf/2011.11235)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Reinforcement Learning (RL) has recently been applied to sequential estimation and prediction problems identifying and developing hypothetical treatment strategies for septic patients, with a particular focus on offline learning with observational data. In practice, successful RL relies on informative latent states derived from sequential observations to develop optimal treatment strategies. To date, how best to construct such states in a healthcare setting is an open question. In this paper, we perform an empirical study of several information encoding architectures using data from septic patients in the MIMIC-III dataset to form representations of a patient state. We evaluate the impact of representation dimension, correlations with established acuity scores, and the treatment policies derived from them. We find that sequentially formed state representations facilitate effective policy learning in batch settings, validating a more thoughtful approach to representation learning that remains faithful to the sequential and partial nature of healthcare data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper investigating the quality of representations learned from healthcare data for use in RL will be published in the proceedings of the coming <a href="https://twitter.com/hashtag/ML4H?src=hash&amp;ref_src=twsrc%5Etfw">#ML4H</a> workshop <a href="https://twitter.com/NeurIPSConf?ref_src=twsrc%5Etfw">@NeurIPSConf</a>!<a href="https://t.co/ASCSE9gFDN">https://t.co/ASCSE9gFDN</a><br><br>w/ H. Zhang, J. Subramanian, <a href="https://twitter.com/mefatemi?ref_src=twsrc%5Etfw">@mefatemi</a> and <a href="https://twitter.com/MarzyehGhassemi?ref_src=twsrc%5Etfw">@MarzyehGhassemi</a> <br><br>Short 🧵 (1/X) <a href="https://t.co/nwOJCwzebS">pic.twitter.com/nwOJCwzebS</a></p>&mdash; Taylor W. Killian (@tw_killian) <a href="https://twitter.com/tw_killian/status/1331309842733146119?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Iterative Text-based Editing of Talking-heads Using Neural Retargeting

Xinwei Yao, Ohad Fried, Kayvon Fatahalian, Maneesh Agrawala

- retweets: 30, favorites: 31 (11/25/2020 12:34:21)

- links: [abs](https://arxiv.org/abs/2011.10688) | [pdf](https://arxiv.org/pdf/2011.10688)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

We present a text-based tool for editing talking-head video that enables an iterative editing workflow. On each iteration users can edit the wording of the speech, further refine mouth motions if necessary to reduce artifacts and manipulate non-verbal aspects of the performance by inserting mouth gestures (e.g. a smile) or changing the overall performance style (e.g. energetic, mumble). Our tool requires only 2-3 minutes of the target actor video and it synthesizes the video for each iteration in about 40 seconds, allowing users to quickly explore many editing possibilities as they iterate. Our approach is based on two key ideas. (1) We develop a fast phoneme search algorithm that can quickly identify phoneme-level subsequences of the source repository video that best match a desired edit. This enables our fast iteration loop. (2) We leverage a large repository of video of a source actor and develop a new self-supervised neural retargeting technique for transferring the mouth motions of the source actor to the target actor. This allows us to work with relatively short target actor videos, making our approach applicable in many real-world editing scenarios. Finally, our refinement and performance controls give users the ability to further fine-tune the synthesized results.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Iterative Text-based Editing of Talking-heads Using Neural Retargeting<br>pdf: <a href="https://t.co/cvY27b7s1q">https://t.co/cvY27b7s1q</a><br>abs: <a href="https://t.co/XbJ7RDCP2C">https://t.co/XbJ7RDCP2C</a><br>project page: <a href="https://t.co/NgYxCxvnFj">https://t.co/NgYxCxvnFj</a> <a href="https://t.co/1nPORqJTWg">pic.twitter.com/1nPORqJTWg</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1331069364271984643?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. A Closed-Form Solution to Local Non-Rigid Structure-from-Motion

Shaifali Parashar, Yuxuan Long, Mathieu Salzmann, Pascal Fua

- retweets: 25, favorites: 27 (11/25/2020 12:34:21)

- links: [abs](https://arxiv.org/abs/2011.11567) | [pdf](https://arxiv.org/pdf/2011.11567)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

A recent trend in Non-Rigid Structure-from-Motion (NRSfM) is to express local, differential constraints between pairs of images, from which the surface normal at any point can be obtained by solving a system of polynomial equations. The systems of equations derived in previous work, however, are of high degree, having up to five real solutions, thus requiring a computationally expensive strategy to select a unique solution. Furthermore, they suffer from degeneracies that make the resulting estimates unreliable, without any mechanism to identify this situation.   In this paper, we show that, under widely applicable assumptions, we can derive a new system of equation in terms of the surface normals whose two solutions can be obtained in closed-form and can easily be disambiguated locally. Our formalism further allows us to assess how reliable the estimated local normals are and, hence, to discard them if they are not. Our experiments show that our reconstructions, obtained from two or more views, are significantly more accurate than those of state-of-the-art methods, while also being faster.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">On rare occasions, we still work on shallow problems without any deep learning, only lots of equations. It&#39;s kind of refreshing actually ;-)  <a href="https://t.co/saNsxOM0In">https://t.co/saNsxOM0In</a> See <a href="https://t.co/9gSLi9rrld">https://t.co/9gSLi9rrld</a> for details. <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a></p>&mdash; Pascal Fua (@FuaPv) <a href="https://twitter.com/FuaPv/status/1331228191252504576?ref_src=twsrc%5Etfw">November 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



