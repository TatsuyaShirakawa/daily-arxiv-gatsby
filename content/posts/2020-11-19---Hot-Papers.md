---
title: Hot Papers 2020-11-19
date: 2020-11-20T10:39:18.Z
template: "post"
draft: false
slug: "hot-papers-2020-11-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-11-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. Liquid Warping GAN with Attention: A Unified Framework for Human Image  Synthesis

Wen Liu, Zhixin Piao, Zhi Tu, Wenhan Luo, Lin Ma, Shenghua Gao

- retweets: 2450, favorites: 272 (11/20/2020 10:39:18)

- links: [abs](https://arxiv.org/abs/2011.09055) | [pdf](https://arxiv.org/pdf/2011.09055)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We tackle human image synthesis, including human motion imitation, appearance transfer, and novel view synthesis, within a unified framework. It means that the model, once being trained, can be used to handle all these tasks. The existing task-specific methods mainly use 2D keypoints to estimate the human body structure. However, they only express the position information with no abilities to characterize the personalized shape of the person and model the limb rotations. In this paper, we propose to use a 3D body mesh recovery module to disentangle the pose and shape. It can not only model the joint location and rotation but also characterize the personalized body shape. To preserve the source information, such as texture, style, color, and face identity, we propose an Attentional Liquid Warping GAN with Attentional Liquid Warping Block (AttLWB) that propagates the source information in both image and feature spaces to the synthesized reference. Specifically, the source features are extracted by a denoising convolutional auto-encoder for characterizing the source identity well. Furthermore, our proposed method can support a more flexible warping from multiple sources. To further improve the generalization ability of the unseen source images, a one/few-shot adversarial learning is applied. In detail, it firstly trains a model in an extensive training set. Then, it finetunes the model by one/few-shot unseen image(s) in a self-supervised way to generate high-resolution (512 x 512 and 1024 x 1024) results. Also, we build a new dataset, namely iPER dataset, for the evaluation of human motion imitation, appearance transfer, and novel view synthesis. Extensive experiments demonstrate the effectiveness of our methods in terms of preserving face identity, shape consistency, and clothes details. All codes and dataset are available on https://impersonator.org/work/impersonator-plus-plus.html.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis<br>pdf: <a href="https://t.co/ALlprn3gBM">https://t.co/ALlprn3gBM</a><br>abs: <a href="https://t.co/fCjP1h6kps">https://t.co/fCjP1h6kps</a><br>project page: <a href="https://t.co/KuOJh5aO8p">https://t.co/KuOJh5aO8p</a> <a href="https://t.co/7wX4YKCUPu">pic.twitter.com/7wX4YKCUPu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1329245144147775488?ref_src=twsrc%5Etfw">November 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. UP-DETR: Unsupervised Pre-training for Object Detection with  Transformers

Zhigang Dai, Bolun Cai, Yugeng Lin, Junying Chen

- retweets: 1122, favorites: 165 (11/20/2020 10:39:18)

- links: [abs](https://arxiv.org/abs/2011.09094) | [pdf](https://arxiv.org/pdf/2011.09094)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Object detection with transformers (DETR) reaches competitive performance with Faster R-CNN via a transformer encoder-decoder architecture. Inspired by the great success of pre-training transformers in natural language processing, we propose a pretext task named random query patch detection to unsupervisedly pre-train DETR (UP-DETR) for object detection. Specifically, we randomly crop patches from the given image and then feed them as queries to the decoder. The model is pre-trained to detect these query patches from the original image. During the pre-training, we address two critical issues: multi-task learning and multi-query localization. (1) To trade-off multi-task learning of classification and localization in the pretext task, we freeze the CNN backbone and propose a patch feature reconstruction branch which is jointly optimized with patch detection. (2) To perform multi-query localization, we introduce UP-DETR from single-query patch and extend it to multi-query patches with object query shuffle and attention mask. In our experiments, UP-DETR significantly boosts the performance of DETR with faster convergence and higher precision on PASCAL VOC and COCO datasets. The code will be available soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">UP-DETR: Unsupervised Pre-training for Object Detection with Transformers<br>pdf: <a href="https://t.co/BBahLvDzzi">https://t.co/BBahLvDzzi</a><br>abs: <a href="https://t.co/TSMVnKSKBm">https://t.co/TSMVnKSKBm</a> <a href="https://t.co/08wkPcz4pt">pic.twitter.com/08wkPcz4pt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1329247865554800643?ref_src=twsrc%5Etfw">November 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. C-Learning: Learning to Achieve Goals via Recursive Classification

Benjamin Eysenbach, Ruslan Salakhutdinov, Sergey Levine

- retweets: 196, favorites: 82 (11/20/2020 10:39:19)

- links: [abs](https://arxiv.org/abs/2011.08909) | [pdf](https://arxiv.org/pdf/2011.08909)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We study the problem of predicting and controlling the future state distribution of an autonomous agent. This problem, which can be viewed as a reframing of goal-conditioned reinforcement learning (RL), is centered around learning a conditional probability density function over future states. Instead of directly estimating this density function, we indirectly estimate this density function by training a classifier to predict whether an observation comes from the future. Via Bayes' rule, predictions from our classifier can be transformed into predictions over future states. Importantly, an off-policy variant of our algorithm allows us to predict the future state distribution of a new policy, without collecting new experience. This variant allows us to optimize functionals of a policy's future state distribution, such as the density of reaching a particular goal state. While conceptually similar to Q-learning, our work lays a principled foundation for goal-conditioned RL as density estimation, providing justification for goal-conditioned methods used in prior work. This foundation makes hypotheses about Q-learning, including the optimal goal-sampling ratio, which we confirm experimentally. Moreover, our proposed method is competitive with prior goal-conditioned RL methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">C-Learning learns goal-conditioned policies using classifiers, without any hand-designed rewards. The theoretical framework in C-Learning helps to explain connections between prediction and Q-functions, relabeling, and others.<a href="https://t.co/tGEGEF52Ga">https://t.co/tGEGEF52Ga</a><br>w/ Eysenbach, <a href="https://twitter.com/rsalakhu?ref_src=twsrc%5Etfw">@rsalakhu</a> <a href="https://t.co/z3co465aBc">pic.twitter.com/z3co465aBc</a></p>&mdash; Sergey Levine (@svlevine) <a href="https://twitter.com/svlevine/status/1329515152769572864?ref_src=twsrc%5Etfw">November 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. A large-scale comparison of social media coverage and mentions captured  by the two altmetric aggregators- Altmetric.com and PlumX

Mousumi Karmakar, Sumit Kumar Banshal, Vivek Kumar Singh

- retweets: 156, favorites: 17 (11/20/2020 10:39:19)

- links: [abs](https://arxiv.org/abs/2011.09069) | [pdf](https://arxiv.org/pdf/2011.09069)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

The increased social media attention to scholarly articles has resulted in efforts to create platforms & services to track and measure the social media transactions around scholarly articles in different social platforms (such as Twitter, Blog, Facebook) and academic social networks (such as Mendeley, Academia and ResearchGate). Altmetric.com and PlumX are two popular aggregators that track social media activity around scholarly articles from a variety of social platforms and provide the coverage and transaction data to researchers for various purposes. However, some previous studies have shown that the social media data captured by the two aggregators have differences in terms of coverage and magnitude of mentions. This paper aims to revisit the question by doing a large-scale analysis of social media mentions of a data sample of 1,785,149 publication records (drawn from multiple disciplines, demographies, publishers). Results obtained show that PlumX tracks more wide sources and more articles as compared to Altmetric.com. However, the coverage and average mentions of the two aggregators vary across different social media platforms, with Altmetric.com recording higher mentions in Twitter and Blog, and PlumX recording higher mentions in Facebook and Mendeley, for the same set of articles. The coverage and average mentions captured by the two aggregators across different document types, disciplines and publishers is also analyzed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A large-scale comparison of <a href="https://t.co/DwOMocOMbD">https://t.co/DwOMocOMbD</a> and PlumX available as pre-print at- <a href="https://t.co/VK1X2bxVfX">https://t.co/VK1X2bxVfX</a> <a href="https://twitter.com/altmetric?ref_src=twsrc%5Etfw">@altmetric</a> <a href="https://twitter.com/PlumAnalytics?ref_src=twsrc%5Etfw">@PlumAnalytics</a> <a href="https://twitter.com/skonkiel?ref_src=twsrc%5Etfw">@skonkiel</a> <a href="https://twitter.com/digitalsci?ref_src=twsrc%5Etfw">@digitalsci</a> <a href="https://twitter.com/SKBanshal?ref_src=twsrc%5Etfw">@SKBanshal</a> <a href="https://twitter.com/Mousumi2212?ref_src=twsrc%5Etfw">@Mousumi2212</a> <a href="https://twitter.com/Philipp_Mayr?ref_src=twsrc%5Etfw">@Philipp_Mayr</a> <a href="https://twitter.com/JLOrtegaPriego?ref_src=twsrc%5Etfw">@JLOrtegaPriego</a> <a href="https://t.co/BSd0hghDRe">pic.twitter.com/BSd0hghDRe</a></p>&mdash; Vivek Singh (@vivekks12) <a href="https://twitter.com/vivekks12/status/1329286377142247425?ref_src=twsrc%5Etfw">November 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. FLaaS: Federated Learning as a Service

Nicolas Kourtellis, Kleomenis Katevas, Diego Perino

- retweets: 42, favorites: 12 (11/20/2020 10:39:19)

- links: [abs](https://arxiv.org/abs/2011.09359) | [pdf](https://arxiv.org/pdf/2011.09359)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

Federated Learning (FL) is emerging as a promising technology to build machine learning models in a decentralized, privacy-preserving fashion. Indeed, FL enables local training on user devices, avoiding user data to be transferred to centralized servers, and can be enhanced with differential privacy mechanisms. Although FL has been recently deployed in real systems, the possibility of collaborative modeling across different 3rd-party applications has not yet been explored. In this paper, we tackle this problem and present Federated Learning as a Service (FLaaS), a system enabling different scenarios of 3rd-party application collaborative model building and addressing the consequent challenges of permission and privacy management, usability, and hierarchical model training. FLaaS can be deployed in different operational environments. As a proof of concept, we implement it on a mobile phone setting and discuss practical implications of results on simulated and real devices with respect to on-device training CPU cost, memory footprint and power consumed per FL model round. Therefore, we demonstrate FLaaS's feasibility in building unique or joint FL models across applications for image object detection in a few hours, across 100 devices.




# 6. Do 'altmetric mentions' follow Power Laws? Evidence from social media  mention data in Altmetric.com

Sumit Kumar Banshal, Aparna Basu, Vivek Kumar Singh, Solanki Gupta, Pranab K. Muhuri

- retweets: 42, favorites: 11 (11/20/2020 10:39:19)

- links: [abs](https://arxiv.org/abs/2011.09079) | [pdf](https://arxiv.org/pdf/2011.09079)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

Power laws are a characteristic distribution that are ubiquitous, in that they are found almost everywhere, in both natural as well as in man-made systems. They tend to emerge in large, connected and self-organizing systems, for example, scholarly publications. Citations to scientific papers have been found to follow a power law, i.e., the number of papers having a certain level of citation x are proportional to x raised to some negative power. The distributional character of altmetrics has not been studied yet as altmetrics are among the newest indicators related to scholarly publications. Here we select a data sample from the altmetrics aggregator Altmetrics.com containing records from the platforms Facebook, Twitter, News, Blogs, etc., and the composite variable Alt-score for the period 2016. The individual and the composite data series of 'mentions' on the various platforms are fit to a power law distribution, and the parameters and goodness of fit determined using least squares regression. The log-log plot of the data, 'mentions' vs. number of papers, falls on an approximately linear line, suggesting the plausibility of a power law distribution. The fit is not very good in all cases due to large fluctuations in the tail. We show that fit to the power law can be improved by truncating the data series to eliminate large fluctuations in the tail. We conclude that altmetric distributions also follow power laws with a fairly good fit over a wide range of values. More rigorous methods of determination may not be necessary at present.




# 7. FROST: Faster and more Robust One-shot Semi-supervised Training

Helena E. Liu, Leslie N. Smith

- retweets: 16, favorites: 35 (11/20/2020 10:39:19)

- links: [abs](https://arxiv.org/abs/2011.09471) | [pdf](https://arxiv.org/pdf/2011.09471)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recent advances in one-shot semi-supervised learning have lowered the barrier for deep learning of new applications. However, the state-of-the-art for semi-supervised learning is slow to train and the performance is sensitive to the choices of the labeled data and hyper-parameter values. In this paper, we present a one-shot semi-supervised learning method that trains up to an order of magnitude faster and is more robust than state-of-the-art methods. Specifically, we show that by combining semi-supervised learning with a one-stage, single network version of self-training, our FROST methodology trains faster and is more robust to choices for the labeled samples and changes in hyper-parameters. Our experiments demonstrate FROST's capability to perform well when the composition of the unlabeled data is unknown; that is when the unlabeled data contain unequal numbers of each class and can contain out-of-distribution examples that don't belong to any of the training classes. High performance, speed of training, and insensitivity to hyper-parameters make FROST the most practical method for one-shot semi-supervised training.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out my new paper &quot;FROST: Faster and more Robust One-shot Semi-supervised Training&quot; at <a href="https://t.co/lhfEfeCcKB">https://t.co/lhfEfeCcKB</a>.  FROST is cool and it is the most practical method for one-shot semi-supervised training.</p>&mdash; Leslie Smith (@lnsmith613) <a href="https://twitter.com/lnsmith613/status/1329547654293053442?ref_src=twsrc%5Etfw">November 19, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



