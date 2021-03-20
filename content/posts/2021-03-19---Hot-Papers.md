---
title: Hot Papers 2021-03-19
date: 2021-03-20T16:34:36.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. Requirement Engineering Challenges for AI-intense Systems Development

Hans-Martin Heyn, Eric Knauss, Amna Pir Muhammad, Olof Erikssonz, Jennifer Linder, Padmini Subbiah, Shameer Kumar Pradhan, Sagar Tungal

- retweets: 5907, favorites: 322 (03/20/2021 16:34:36)

- links: [abs](https://arxiv.org/abs/2103.10270) | [pdf](https://arxiv.org/pdf/2103.10270)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Availability of powerful computation and communication technology as well as advances in artificial intelligence enable a new generation of complex, AI-intense systems and applications. Such systems and applications promise exciting improvements on a societal level, yet they also bring with them new challenges for their development. In this paper we argue that significant challenges relate to defining and ensuring behaviour and quality attributes of such systems and applications. We specifically derive four challenge areas from relevant use cases of complex, AI-intense systems and applications related to industry, transportation, and home automation: understanding, determining, and specifying (i) contextual definitions and requirements, (ii) data attributes and requirements, (iii) performance definition and monitoring, and (iv) the impact of human factors on system acceptance and success. Solving these challenges will imply process support that integrates new requirements engineering methods into development approaches for complex, AI-intense systems and applications. We present these challenges in detail and propose a research roadmap.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚀 A great read for machine learning engineers. It focuses on the engineering challenges of AI-intense systems development. Topics range from data requirements to performance definition and monitoring. Lots of practical tips across different use cases. <a href="https://t.co/U9QjYTLRvJ">https://t.co/U9QjYTLRvJ</a> <a href="https://t.co/BngDDMnkcV">pic.twitter.com/BngDDMnkcV</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1372896507242946563?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Learning to Resize Images for Computer Vision Tasks

Hossein Talebi, Peyman Milanfar

- retweets: 5046, favorites: 291 (03/20/2021 16:34:36)

- links: [abs](https://arxiv.org/abs/2103.09950) | [pdf](https://arxiv.org/pdf/2103.09950)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

For all the ways convolutional neural nets have revolutionized computer vision in recent years, one important aspect has received surprisingly little attention: the effect of image size on the accuracy of tasks being trained for. Typically, to be efficient, the input images are resized to a relatively small spatial resolution (e.g. 224x224), and both training and inference are carried out at this resolution. The actual mechanism for this re-scaling has been an afterthought: Namely, off-the-shelf image resizers such as bilinear and bicubic are commonly used in most machine learning software frameworks. But do these resizers limit the on task performance of the trained networks? The answer is yes. Indeed, we show that the typical linear resizer can be replaced with learned resizers that can substantially improve performance. Importantly, while the classical resizers typically result in better perceptual quality of the downscaled images, our proposed learned resizers do not necessarily give better visual quality, but instead improve task performance. Our learned image resizer is jointly trained with a baseline vision model. This learned CNN-based resizer creates machine friendly visual manipulations that lead to a consistent improvement of the end task metric over the baseline model. Specifically, here we focus on the classification task with the ImageNet dataset, and experiment with four different models to learn resizers adapted to each model. Moreover, we show that the proposed resizer can also be useful for fine-tuning the classification baselines for other vision tasks. To this end, we experiment with three different baselines to develop image quality assessment (IQA) models on the AVA dataset.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Front-end resizers in deep networks are simple filters. They’re an afterthought — but they shouldn’t be<br><br>Deep computer vision models can benefit greatly from replacing these fixed linear resizers with well-designed, learned, nonlinear resizers.<br><br>A thread<a href="https://t.co/aShHBFMdCl">https://t.co/aShHBFMdCl</a> <a href="https://t.co/oSGFsJxyWD">pic.twitter.com/oSGFsJxyWD</a></p>&mdash; Peyman Milanfar (@docmilanfar) <a href="https://twitter.com/docmilanfar/status/1372781113886662661?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. FastNeRF: High-Fidelity Neural Rendering at 200FPS

Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin

- retweets: 2248, favorites: 367 (03/20/2021 16:34:36)

- links: [abs](https://arxiv.org/abs/2103.10380) | [pdf](https://arxiv.org/pdf/2103.10380)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent work on Neural Radiance Fields (NeRF) showed how neural networks can be used to encode complex 3D environments that can be rendered photorealistically from novel viewpoints. Rendering these images is very computationally demanding and recent improvements are still a long way from enabling interactive rates, even on high-end hardware. Motivated by scenarios on mobile and mixed reality devices, we propose FastNeRF, the first NeRF-based system capable of rendering high fidelity photorealistic images at 200Hz on a high-end consumer GPU. The core of our method is a graphics-inspired factorization that allows for (i) compactly caching a deep radiance map at each position in space, (ii) efficiently querying that map using ray directions to estimate the pixel values in the rendered image. Extensive experiments show that the proposed method is 3000 times faster than the original NeRF algorithm and at least an order of magnitude faster than existing work on accelerating NeRF, while maintaining visual quality and extensibility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FastNeRF: High-Fidelity Neural Rendering at 200FPS<br>pdf: <a href="https://t.co/EZquUukJLH">https://t.co/EZquUukJLH</a><br>abs: <a href="https://t.co/f2h0NHEZWh">https://t.co/f2h0NHEZWh</a> <a href="https://t.co/opJkZ8ujEs">pic.twitter.com/opJkZ8ujEs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372712928416296970?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stephan J. Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, Julien Valentin , FastNeRF: High-Fidelity Neural Rendering at 200FPS, arXiv, 2021.<a href="https://t.co/4wlAPpJuFd">https://t.co/4wlAPpJuFd</a> <a href="https://t.co/7temkke1yL">pic.twitter.com/7temkke1yL</a></p>&mdash; Kosta Derpanis (@CSProfKGD) <a href="https://twitter.com/CSProfKGD/status/1372734088625451013?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Vanishing size of critical mass for tipping points in social convention

Iacopo Iacopini, Giovanni Petri, Andrea Baronchelli, Alain Barrat

- retweets: 2352, favorites: 141 (03/20/2021 16:34:37)

- links: [abs](https://arxiv.org/abs/2103.10411) | [pdf](https://arxiv.org/pdf/2103.10411)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

How can minorities of regular individuals overturn social conventions? Theoretical and empirical studies have proposed that when a committed minority reaches a critical group size-ranging from 10% of the population up to 40%-a cascade of behaviour change rapidly increases the acceptance of the minority view and apparently stable social norms can be overturned. However, several observations suggest that much smaller groups may be sufficient to bring the system to a tipping point. Here, we generalise a model previously used for both theoretical and empirical investigations of tipping points in social convention and find that the critical mass necessary to trigger behaviour change is dramatically reduced if individuals are less prone to change their views, i.e., are more resistant to social influence. We show that groups smaller than 3% of the population are effective on different kinds of social networks, both when pairwise or group interactions are considered, and in a broad region of the parameter space. In some cases, even groups as small as 0.3% may overturn the current social norm. Our findings reconcile the numerous observational accounts of rapid change in social convention triggered by committed minorities with the apparent difficulty of establishing such large minorities in the first place. We anticipate that they will be of interest for both researchers and practitioners interested in understanding the phenomenon of norm change, and in designing interventions aimed at contrasting such global challenges as climate change and vaccine hesitancy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How can minorities of regular individuals overturn social conventions?<br><br>With <a href="https://twitter.com/lordgrilo?ref_src=twsrc%5Etfw">@lordgrilo</a>, <a href="https://twitter.com/a_baronca?ref_src=twsrc%5Etfw">@a_baronca</a> &amp; <a href="https://twitter.com/alainbarrat?ref_src=twsrc%5Etfw">@alainbarrat</a> we introduce (<a href="https://t.co/ssW5xDgqEl">https://t.co/ssW5xDgqEl</a>) a higher-order naming game showing:<br><br>- vanishing size of critical mass under imperfect social influence<br>- strong group effects <a href="https://t.co/SHr7T37xo3">pic.twitter.com/SHr7T37xo3</a></p>&mdash; Iacopo Iacopini (@iacopoiacopini) <a href="https://twitter.com/iacopoiacopini/status/1372793412173193216?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. All NLP Tasks Are Generation Tasks: A General Pretraining Framework

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang

- retweets: 1752, favorites: 289 (03/20/2021 16:34:37)

- links: [abs](https://arxiv.org/abs/2103.10360) | [pdf](https://arxiv.org/pdf/2103.10360)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

There have been various types of pretraining architectures including autoregressive models (e.g., GPT), autoencoding models (e.g., BERT), and encoder-decoder models (e.g., T5). On the other hand, NLP tasks are different in nature, with three main categories being classification, unconditional generation, and conditional generation. However, none of the pretraining frameworks performs the best for all tasks, which introduces inconvenience for model development and selection. We propose a novel pretraining framework GLM (General Language Model) to address this challenge. Compared to previous work, our architecture has three major benefits: (1) it performs well on classification, unconditional generation, and conditional generation tasks with one single pretrained model; (2) it outperforms BERT-like models on classification due to improved pretrain-finetune consistency; (3) it naturally handles variable-length blank filling which is crucial for many downstream tasks. Empirically, GLM substantially outperforms BERT on the SuperGLUE natural language understanding benchmark with the same amount of pre-training data. Moreover, GLM with 1.25x parameters of BERT-Large achieves the best performance in NLU, conditional and unconditional generation at the same time, which demonstrates its generalizability to different downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">All NLP Tasks Are Generation Tasks: A General Pretraining Framework<br><br>Proposes GLM, that performs well on classification, unconditional generation, and conditional generation with single pretrained model.<br><br>abs: <a href="https://t.co/3AdA1I7Y0f">https://t.co/3AdA1I7Y0f</a><br>code: <a href="https://t.co/8Xpnaa67Il">https://t.co/8Xpnaa67Il</a> <a href="https://t.co/JQAYtI3upc">pic.twitter.com/JQAYtI3upc</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1372710196196171776?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">All NLP Tasks Are Generation Tasks: A General Pretraining Framework<br>pdf: <a href="https://t.co/Zhdx33AUgL">https://t.co/Zhdx33AUgL</a><br>abs: <a href="https://t.co/ibJdjVbIsD">https://t.co/ibJdjVbIsD</a> <a href="https://t.co/8OPdrHVCPn">pic.twitter.com/8OPdrHVCPn</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372709807145254922?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. GPT Understands, Too

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, Jie Tang

- retweets: 1398, favorites: 224 (03/20/2021 16:34:37)

- links: [abs](https://arxiv.org/abs/2103.10385) | [pdf](https://arxiv.org/pdf/2103.10385)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

While GPTs with traditional fine-tuning fail to achieve strong results on natural language understanding (NLU), we show that GPTs can be better than or comparable to similar-sized BERTs on NLU tasks with a novel method P-tuning -- which employs trainable continuous prompt embeddings. On the knowledge probing (LAMA) benchmark, the best GPT recovers 64\% (P@1) of world knowledge without any additional text provided during test time, which substantially improves the previous best by 20+ percentage points. On the SuperGlue benchmark, GPTs achieve comparable and sometimes better performance to similar-sized BERTs in supervised learning. Importantly, we find that P-tuning also improves BERTs' performance in both few-shot and supervised settings while largely reducing the need for prompt engineering. Consequently, P-tuning outperforms the state-of-the-art approaches on the few-shot SuperGlue benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GPT Understands, Too<br><br>Proposes P-tuning, a novel fine-tuning method that allows GPTs to achieve comparable and sometimes better performance to similar-sized BERTs in supervised<br>learning.<br><br>abs: <a href="https://t.co/7zDcgfrhts">https://t.co/7zDcgfrhts</a><br>code: <a href="https://t.co/Dy0exDNEiz">https://t.co/Dy0exDNEiz</a> <a href="https://t.co/kd6rrJzmFF">pic.twitter.com/kd6rrJzmFF</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1372710844639760385?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Large Scale Image Completion via Co-Modulated Generative Adversarial  Networks

Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, Yan Xu

- retweets: 840, favorites: 127 (03/20/2021 16:34:38)

- links: [abs](https://arxiv.org/abs/2103.10428) | [pdf](https://arxiv.org/pdf/2103.10428)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Numerous task-specific variants of conditional generative adversarial networks have been developed for image completion. Yet, a serious limitation remains that all existing algorithms tend to fail when handling large-scale missing regions. To overcome this challenge, we propose a generic new approach that bridges the gap between image-conditional and recent modulated unconditional generative architectures via co-modulation of both conditional and stochastic style representations. Also, due to the lack of good quantitative metrics for image completion, we propose the new Paired/Unpaired Inception Discriminative Score (P-IDS/U-IDS), which robustly measures the perceptual fidelity of inpainted images compared to real images via linear separability in a feature space. Experiments demonstrate superior performance in terms of both quality and diversity over state-of-the-art methods in free-form image completion and easy generalization to image-to-image translation. Code is available at https://github.com/zsyzzsoft/co-mod-gan.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large Scale Image Completion via Co-Modulated Generative Adversarial Networks<br>pdf: <a href="https://t.co/hBEfWq7rqp">https://t.co/hBEfWq7rqp</a><br>abs: <a href="https://t.co/2BX4Ykrwmt">https://t.co/2BX4Ykrwmt</a><br>github: <a href="https://t.co/lIABQnV3aV">https://t.co/lIABQnV3aV</a> <a href="https://t.co/A53Nkae26K">pic.twitter.com/A53Nkae26K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372718074110078984?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Using latent space regression to analyze and leverage compositionality  in GANs

Lucy Chai, Jonas Wulff, Phillip Isola

- retweets: 708, favorites: 147 (03/20/2021 16:34:38)

- links: [abs](https://arxiv.org/abs/2103.10426) | [pdf](https://arxiv.org/pdf/2103.10426)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In recent years, Generative Adversarial Networks have become ubiquitous in both research and public perception, but how GANs convert an unstructured latent code to a high quality output is still an open question. In this work, we investigate regression into the latent space as a probe to understand the compositional properties of GANs. We find that combining the regressor and a pretrained generator provides a strong image prior, allowing us to create composite images from a collage of random image parts at inference time while maintaining global consistency. To compare compositional properties across different generators, we measure the trade-offs between reconstruction of the unrealistic input and image quality of the regenerated samples. We find that the regression approach enables more localized editing of individual image parts compared to direct editing in the latent space, and we conduct experiments to quantify this independence effect. Our method is agnostic to the semantics of edits, and does not require labels or predefined concepts during training. Beyond image composition, our method extends to a number of related applications, such as image inpainting or example-based image editing, which we demonstrate on several GANs and datasets, and because it uses only a single forward pass, it can operate in real-time. Code is available on our project page: https://chail.github.io/latent-composition/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Using latent space regression to analyze and leverage compositionality in GANs<br>pdf: <a href="https://t.co/rPXAG7Z9sO">https://t.co/rPXAG7Z9sO</a><br>abs: <a href="https://t.co/vwiI4op8qE">https://t.co/vwiI4op8qE</a><br>project page: <a href="https://t.co/s8ZYe07Yjf">https://t.co/s8ZYe07Yjf</a><br>github: <a href="https://t.co/uTln0UwyjT">https://t.co/uTln0UwyjT</a><br>colab: <a href="https://t.co/nO1Sc5MlQO">https://t.co/nO1Sc5MlQO</a> <a href="https://t.co/G6hyERQss7">pic.twitter.com/G6hyERQss7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372717212901986309?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Situated Language Learning via Interactive Narratives

Prithviraj Ammanabrolu, Mark O. Riedl

- retweets: 276, favorites: 87 (03/20/2021 16:34:38)

- links: [abs](https://arxiv.org/abs/2103.09977) | [pdf](https://arxiv.org/pdf/2103.09977)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

This paper provides a roadmap that explores the question of how to imbue learning agents with the ability to understand and generate contextually relevant natural language in service of achieving a goal. We hypothesize that two key components in creating such agents are interactivity and environment grounding, shown to be vital parts of language learning in humans, and posit that interactive narratives should be the environments of choice for such training these agents. These games are simulations in which an agent interacts with the world through natural language -- "perceiving", "acting upon", and "talking to" the world using textual descriptions, commands, and dialogue -- and as such exist at the intersection of natural language processing, storytelling, and sequential decision making. We discuss the unique challenges a text games' puzzle-like structure combined with natural language state-and-action spaces provides: knowledge representation, commonsense reasoning, and exploration. Beyond the challenges described so far, progress in the realm of interactive narratives can be applied in adjacent problem domains. These applications provide interesting challenges of their own as well as extensions to those discussed so far. We describe three of them in detail: (1) evaluating AI system's commonsense understanding by automatically creating interactive narratives; (2) adapting abstract text-based policies to include other modalities such as vision; and (3) enabling multi-agent and human-AI collaboration in shared, situated worlds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Situated Language Learning via Interactive Narratives<a href="https://t.co/QQPsZFk0Hw">https://t.co/QQPsZFk0Hw</a><a href="https://twitter.com/rajammanabrolu?ref_src=twsrc%5Etfw">@rajammanabrolu</a> explains why you should totally be working on AI for interactive narratives and text adventure games. For science! <a href="https://t.co/pw8PKm0Wbr">pic.twitter.com/pw8PKm0Wbr</a></p>&mdash; Mark O. Riedl (@mark_riedl) <a href="https://twitter.com/mark_riedl/status/1372715298655236096?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Your new excuse to play games at work! <a href="https://twitter.com/mark_riedl?ref_src=twsrc%5Etfw">@mark_riedl</a> and I talk abt the importance of *interactive+situated* <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> and why interactive narratives like <a href="https://twitter.com/hashtag/Zork?src=hash&amp;ref_src=twsrc%5Etfw">#Zork</a> (and beyond!) are perfect training envs for language based <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> agents. <a href="https://t.co/tSHNaM5ULS">https://t.co/tSHNaM5ULS</a> 1/4 <a href="https://t.co/5bfSv6Wer6">pic.twitter.com/5bfSv6Wer6</a></p>&mdash; Prithviraj Ammanabrolu (@rajammanabrolu) <a href="https://twitter.com/rajammanabrolu/status/1372720204975849474?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Neural tensor contractions and the expressive power of deep neural  quantum states

Or Sharir, Amnon Shashua, Giuseppe Carleo

- retweets: 104, favorites: 106 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10293) | [pdf](https://arxiv.org/pdf/2103.10293)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

We establish a direct connection between general tensor networks and deep feed-forward artificial neural networks. The core of our results is the construction of neural-network layers that efficiently perform tensor contractions, and that use commonly adopted non-linear activation functions. The resulting deep networks feature a number of edges that closely matches the contraction complexity of the tensor networks to be approximated. In the context of many-body quantum states, this result establishes that neural-network states have strictly the same or higher expressive power than practically usable variational tensor networks. As an example, we show that all matrix product states can be efficiently written as neural-network states with a number of edges polynomial in the bond dimension and depth logarithmic in the system size. The opposite instead does not hold true, and our results imply that there exist quantum states that are not efficiently expressible in terms of matrix product states or practically usable PEPS, but that are instead efficiently expressible with neural network states.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient &quot;conversions&quot; between tensor networks and common deep networks are possible. Useful to prove representation results. Example: all Matrix Product States can be efficiently written as &quot;quasi-shallow&quot; deep Neural Quantum States. <a href="https://twitter.com/cqs_lab?ref_src=twsrc%5Etfw">@cqs_lab</a> <a href="https://twitter.com/HebrewU?ref_src=twsrc%5Etfw">@HebrewU</a> <a href="https://t.co/sJGzCY3CbY">https://t.co/sJGzCY3CbY</a> <a href="https://t.co/ErawrBGmrQ">https://t.co/ErawrBGmrQ</a></p>&mdash; Giuseppe Carleo (@gppcarleo) <a href="https://twitter.com/gppcarleo/status/1372817862939848711?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A new pre-print &quot;Neural tensor contractions and the expressive power of deep neural quantum states&quot; is out. By O. Sharir, <a href="https://twitter.com/AmnonShashua?ref_src=twsrc%5Etfw">@AmnonShashua</a> , <a href="https://twitter.com/gppcarleo?ref_src=twsrc%5Etfw">@gppcarleo</a>, in a collaboration between <a href="https://twitter.com/cqs_lab?ref_src=twsrc%5Etfw">@cqs_lab</a> <a href="https://twitter.com/EPFL_en?ref_src=twsrc%5Etfw">@EPFL_en</a> and <a href="https://twitter.com/HebrewU?ref_src=twsrc%5Etfw">@HebrewU</a> <a href="https://t.co/DXajdvMyXe">https://t.co/DXajdvMyXe</a></p>&mdash; Computational Quantum Science Lab (@cqs_lab) <a href="https://twitter.com/cqs_lab/status/1372810574409695236?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible  Neural Networks

Despoina Paschalidou, Angelos Katharopoulos, Andreas Geiger, Sanja Fidler

- retweets: 110, favorites: 66 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10429) | [pdf](https://arxiv.org/pdf/2103.10429)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Impressive progress in 3D shape extraction led to representations that can capture object geometries with high fidelity. In parallel, primitive-based methods seek to represent objects as semantically consistent part arrangements. However, due to the simplicity of existing primitive representations, these methods fail to accurately reconstruct 3D shapes using a small number of primitives/parts. We address the trade-off between reconstruction quality and number of parts with Neural Parts, a novel 3D primitive representation that defines primitives using an Invertible Neural Network (INN) which implements homeomorphic mappings between a sphere and the target object. The INN allows us to compute the inverse mapping of the homeomorphism, which in turn, enables the efficient computation of both the implicit surface function of a primitive and its mesh, without any additional post-processing. Our model learns to parse 3D objects into semantically consistent part arrangements without any part-level supervision. Evaluations on ShapeNet, D-FAUST and FreiHAND demonstrate that our primitives can capture complex geometries and thus simultaneously achieve geometrically accurate as well as interpretable reconstructions using an order of magnitude fewer primitives than state-of-the-art shape abstraction methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks<br>pdf: <a href="https://t.co/j7TM9ZbCBn">https://t.co/j7TM9ZbCBn</a><br>abs: <a href="https://t.co/vYOijxK3Nf">https://t.co/vYOijxK3Nf</a><br>project page: <a href="https://t.co/qAmh7W3Xxl">https://t.co/qAmh7W3Xxl</a> <a href="https://t.co/av81YMaQZX">pic.twitter.com/av81YMaQZX</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372720556487901192?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. On Semantic Similarity in Video Retrieval

Michael Wray, Hazel Doughty, Dima Damen

- retweets: 86, favorites: 75 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10095) | [pdf](https://arxiv.org/pdf/2103.10095)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Current video retrieval efforts all found their evaluation on an instance-based assumption, that only a single caption is relevant to a query video and vice versa. We demonstrate that this assumption results in performance comparisons often not indicative of models' retrieval capabilities. We propose a move to semantic similarity video retrieval, where (i) multiple videos/captions can be deemed equally relevant, and their relative ranking does not affect a method's reported performance and (ii) retrieved videos/captions are ranked by their similarity to a query. We propose several proxies to estimate semantic similarities in large-scale retrieval datasets, without additional annotations. Our analysis is performed on three commonly used video retrieval datasets (MSR-VTT, YouCook2 and EPIC-KITCHENS).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What is wrong with video retrieval benchmarks?<br>Our <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> work now on ArXiv<a href="https://t.co/1MiQecQyVc">https://t.co/1MiQecQyVc</a><br>w M Wray <a href="https://twitter.com/doughty_hazel?ref_src=twsrc%5Etfw">@doughty_hazel</a> <br>Prior works are based on instance-based assumption.<br>We propose to rank videos by their semantic similarity, with multiple videos being equally relevant. <a href="https://t.co/Y1kLfb3dcG">pic.twitter.com/Y1kLfb3dcG</a></p>&mdash; Dima Damen (@dimadamen) <a href="https://twitter.com/dimadamen/status/1372907428493283336?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Generating Diverse Structure for Image Inpainting With Hierarchical  VQ-VAE

Jialun Peng, Dong Liu, Songcen Xu, Houqiang Li

- retweets: 42, favorites: 47 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10022) | [pdf](https://arxiv.org/pdf/2103.10022)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Given an incomplete image without additional constraint, image inpainting natively allows for multiple solutions as long as they appear plausible. Recently, multiplesolution inpainting methods have been proposed and shown the potential of generating diverse results. However, these methods have difficulty in ensuring the quality of each solution, e.g. they produce distorted structure and/or blurry texture. We propose a two-stage model for diverse inpainting, where the first stage generates multiple coarse results each of which has a different structure, and the second stage refines each coarse result separately by augmenting texture. The proposed model is inspired by the hierarchical vector quantized variational auto-encoder (VQ-VAE), whose hierarchical architecture isentangles structural and textural information. In addition, the vector quantization in VQVAE enables autoregressive modeling of the discrete distribution over the structural information. Sampling from the distribution can easily generate diverse and high-quality structures, making up the first stage of our model. In the second stage, we propose a structural attention module inside the texture generation network, where the module utilizes the structural information to capture distant correlations. We further reuse the VQ-VAE to calculate two feature losses, which help improve structure coherence and texture realism, respectively. Experimental results on CelebA-HQ, Places2, and ImageNet datasets show that our method not only enhances the diversity of the inpainting solutions but also improves the visual quality of the generated multiple images. Code and models are available at: https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating Diverse Structure for Image Inpainting With Hierarchical VQ-VAE<br>pdf: <a href="https://t.co/EEaIYZiGtq">https://t.co/EEaIYZiGtq</a><br>abs: <a href="https://t.co/d5D3ERjMVi">https://t.co/d5D3ERjMVi</a><br>github: <a href="https://t.co/bJXvj2LkPc">https://t.co/bJXvj2LkPc</a> <a href="https://t.co/spAQExoqyv">pic.twitter.com/spAQExoqyv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372712236020551680?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Robust Vision-Based Cheat Detection in Competitive Gaming

Aditya Jonnalagadda, Iuri Frosio, Seth Schneider, Morgan McGuire, Joohwan Kim

- retweets: 49, favorites: 22 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10031) | [pdf](https://arxiv.org/pdf/2103.10031)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Game publishers and anti-cheat companies have been unsuccessful in blocking cheating in online gaming. We propose a novel, vision-based approach that captures the final state of the frame buffer and detects illicit overlays. To this aim, we train and evaluate a DNN detector on a new dataset, collected using two first-person shooter games and three cheating software. We study the advantages and disadvantages of different DNN architectures operating on a local or global scale. We use output confidence analysis to avoid unreliable detections and inform when network retraining is required. In an ablation study, we show how to use Interval Bound Propagation to build a detector that is also resistant to potential adversarial attacks and study its interaction with confidence analysis. Our results show that robust and effective anti-cheating through machine learning is practically feasible and can be used to guarantee fair play in online gaming.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robust Vision-Based Cheat Detection in Competitive<br>Gaming<br>pdf: <a href="https://t.co/DBpaY5H3Lm">https://t.co/DBpaY5H3Lm</a><br>abs: <a href="https://t.co/SDgbo4a0ZY">https://t.co/SDgbo4a0ZY</a> <a href="https://t.co/0iMrGFQQwM">pic.twitter.com/0iMrGFQQwM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372715949984452612?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Porting a sparse linear algebra math library to Intel GPUs

Yuhsiang M. Tsai, Terry Cojean, Hartwig Anzt

- retweets: 42, favorites: 22 (03/20/2021 16:34:39)

- links: [abs](https://arxiv.org/abs/2103.10116) | [pdf](https://arxiv.org/pdf/2103.10116)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.MS](https://arxiv.org/list/cs.MS/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

With the announcement that the Aurora Supercomputer will be composed of general purpose Intel CPUs complemented by discrete high performance Intel GPUs, and the deployment of the oneAPI ecosystem, Intel has committed to enter the arena of discrete high performance GPUs. A central requirement for the scientific computing community is the availability of production-ready software stacks and a glimpse of the performance they can expect to see on Intel high performance GPUs. In this paper, we present the first platform-portable open source math library supporting Intel GPUs via the DPC++ programming environment. We also benchmark some of the developed sparse linear algebra functionality on different Intel GPUs to assess the efficiency of the DPC++ programming ecosystem to translate raw performance into application performance. Aside from quantifying the efficiency within the hardware-specific roofline model, we also compare against routines providing the same functionality that ship with Intel's oneMKL vendor library.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Researchers have presented the first platform-portable open source math library supporting Intel GPUs via the DPC++ programming environment, showing some raw performance results in different Intel GPU gens (Gen9 and Intel Iris Xe Max)<a href="https://t.co/amTIw3YnG1">https://t.co/amTIw3YnG1</a> <a href="https://t.co/xouxGPjR8Q">pic.twitter.com/xouxGPjR8Q</a></p>&mdash; Underfox (@Underfox3) <a href="https://twitter.com/Underfox3/status/1372762977758183432?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Self-Supervised Learning of Audio Representations from Permutations with  Differentiable Ranking

Andrew N Carr, Quentin Berthet, Mathieu Blondel, Olivier Teboul, Neil Zeghidour

- retweets: 8, favorites: 54 (03/20/2021 16:34:40)

- links: [abs](https://arxiv.org/abs/2103.09879) | [pdf](https://arxiv.org/pdf/2103.09879)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Self-supervised pre-training using so-called "pretext" tasks has recently shown impressive performance across a wide range of modalities. In this work, we advance self-supervised learning from permutations, by pre-training a model to reorder shuffled parts of the spectrogram of an audio signal, to improve downstream classification performance. We make two main contributions. First, we overcome the main challenges of integrating permutation inversions into an end-to-end training scheme, using recent advances in differentiable ranking. This was heretofore sidestepped by casting the reordering task as classification, fundamentally reducing the space of permutations that can be exploited. Our experiments validate that learning from all possible permutations improves the quality of the pre-trained representations over using a limited, fixed set. Second, we show that inverting permutations is a meaningful pretext task for learning audio representations in an unsupervised fashion. In particular, we improve instrument classification and pitch estimation of musical notes by reordering spectrogram patches in the time-frequency space.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Self-Supervised Learning of Audio Representations from Permutations with Differentiable Ranking<br>pdf: <a href="https://t.co/lwwitOE6y4">https://t.co/lwwitOE6y4</a><br>abs: <a href="https://t.co/xzuqSPnK7P">https://t.co/xzuqSPnK7P</a> <a href="https://t.co/IwL3TRiKoN">pic.twitter.com/IwL3TRiKoN</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1372713813477310477?ref_src=twsrc%5Etfw">March 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



