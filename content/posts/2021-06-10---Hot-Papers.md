---
title: Hot Papers 2021-06-10
date: 2021-06-11T09:50:30.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-10"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-10"
socialImage: "/media/flying-marine.jpg"

---

# 1. Knowledge distillation: A good teacher is patient and consistent

Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov

- retweets: 5245, favorites: 92 (06/11/2021 09:50:30)

- links: [abs](https://arxiv.org/abs/2106.05237) | [pdf](https://arxiv.org/pdf/2106.05237)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

There is a growing discrepancy in computer vision between large-scale models that achieve state-of-the-art performance and models that are affordable in practical applications. In this paper we address this issue and significantly bridge the gap between these two types of models. Throughout our empirical investigation we do not aim to necessarily propose a new method, but strive to identify a robust and effective recipe for making state-of-the-art large scale models affordable in practice. We demonstrate that, when performed correctly, knowledge distillation can be a powerful tool for reducing the size of large models without compromising their performance. In particular, we uncover that there are certain implicit design choices, which may drastically affect the effectiveness of distillation. Our key contribution is the explicit identification of these design choices, which were not previously articulated in the literature. We back up our findings by a comprehensive empirical study, demonstrate compelling results on a wide range of vision datasets and, in particular, obtain a state-of-the-art ResNet-50 model for ImageNet, which achieves 82.8\% top-1 accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Wondering how to distill big vision models? <br><br>Check our recipe: a good teacher is patient and consistent!<br><br>Thanks to patience and consistency, we obtained the best ever ResNet-50 on ImageNet, of 82.8% accuracy without tricks. <br><br>Paper: <a href="https://t.co/obaBeuKJNl">https://t.co/obaBeuKJNl</a> <a href="https://t.co/Ua9lYZu4df">pic.twitter.com/Ua9lYZu4df</a></p>&mdash; Xiaohua Zhai (@XiaohuaZhai) <a href="https://twitter.com/XiaohuaZhai/status/1402907931406909440?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">So you think you know distillation; it&#39;s easy, right?<br><br>We thought so too with <a href="https://twitter.com/XiaohuaZhai?ref_src=twsrc%5Etfw">@XiaohuaZhai</a> <a href="https://twitter.com/__kolesnikov__?ref_src=twsrc%5Etfw">@__kolesnikov__</a> <a href="https://twitter.com/_arohan_?ref_src=twsrc%5Etfw">@_arohan_</a> and the amazing <a href="https://twitter.com/royaleerieme?ref_src=twsrc%5Etfw">@royaleerieme</a> and Larisa Markeeva.<br><br>Until we didn&#39;t. But now we do again. Hop on for a ride (+the best ever ResNet50?)<br><br>🧵👇<a href="https://t.co/3SlkXVZcG3">https://t.co/3SlkXVZcG3</a> <a href="https://t.co/Qp5qiZzV14">pic.twitter.com/Qp5qiZzV14</a></p>&mdash; Lucas Beyer (@giffmana) <a href="https://twitter.com/giffmana/status/1402836863954599936?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. CoAtNet: Marrying Convolution and Attention for All Data Sizes

Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan

- retweets: 4858, favorites: 27 (06/11/2021 09:50:30)

- links: [abs](https://arxiv.org/abs/2106.04803) | [pdf](https://arxiv.org/pdf/2106.04803)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present CoAtNets(pronounced "coat" nets), a family of hybrid models built from two key insights:(1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency. Experiments show that our CoAtNets achieve state-of-the-art performance under different resource constraints across various datasets. For example, CoAtNet achieves 86.0% ImageNet top-1 accuracy without extra data, and 89.77% with extra JFT data, outperforming prior arts of both convolutional networks and Transformers. Notably, when pre-trained with 13M images fromImageNet-21K, our CoAtNet achieves 88.56% top-1 accuracy, matching ViT-huge pre-trained with 300M images from JFT while using 23x less data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to introduce CoAtNet: combining convolution and self-attention in a principled way to obtain better capacity and better generalization.<br><br>88.56% top-1  with ImageNet21K (13M imgs), matching ViT-huge with JFT (300M imgs). <br><br>Paper: <a href="https://t.co/AQE33LuzSr">https://t.co/AQE33LuzSr</a> <a href="https://t.co/YEly0cSaTp">pic.twitter.com/YEly0cSaTp</a></p>&mdash; Mingxing Tan (@tanmingxing) <a href="https://twitter.com/tanmingxing/status/1402797579365011458?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. SpeechBrain: A General-Purpose Speech Toolkit

Mirco Ravanelli, Titouan Parcollet, Peter Plantinga, Aku Rouhe, Samuele Cornell, Loren Lugosch, Cem Subakan, Nauman Dawalatabad, Abdelwahab Heba, Jianyuan Zhong, Ju-Chieh Chou, Sung-Lin Yeh, Szu-Wei Fu, Chien-Feng Liao, Elena Rastorgueva, François Grondin, William Aris, Hwidong Na, Yan Gao, Renato De Mori, Yoshua Bengio

- retweets: 2432, favorites: 283 (06/11/2021 09:50:31)

- links: [abs](https://arxiv.org/abs/2106.04624) | [pdf](https://arxiv.org/pdf/2106.04624)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

SpeechBrain is an open-source and all-in-one speech toolkit. It is designed to facilitate the research and development of neural speech processing technologies by being simple, flexible, user-friendly, and well-documented. This paper describes the core architecture designed to support several tasks of common interest, allowing users to naturally conceive, compare and share novel speech processing pipelines. SpeechBrain achieves competitive or state-of-the-art performance in a wide range of speech benchmarks. It also provides training recipes, pretrained models, and inference scripts for popular speech datasets, as well as tutorials which allow anyone with basic Python proficiency to familiarize themselves with speech technologies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m happy to announce that a preprint paper on <a href="https://twitter.com/hashtag/SpeechBrain?src=hash&amp;ref_src=twsrc%5Etfw">#SpeechBrain</a> is now available on <a href="https://twitter.com/hashtag/arXiv?src=hash&amp;ref_src=twsrc%5Etfw">#arXiv</a>:<br><br>Preprint: <a href="https://t.co/qibipubtfW">https://t.co/qibipubtfW</a><br>Website: <a href="https://t.co/a1wqxLucgw">https://t.co/a1wqxLucgw</a><br><br>That&#39;s a good read for the weekend after <a href="https://twitter.com/hashtag/ICASSP2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICASSP2021</a>! 😄<a href="https://twitter.com/PyTorch?ref_src=twsrc%5Etfw">@PyTorch</a> <a href="https://twitter.com/huggingface?ref_src=twsrc%5Etfw">@huggingface</a>  <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a> <a href="https://twitter.com/hashtag/Speech?src=hash&amp;ref_src=twsrc%5Etfw">#Speech</a> <a href="https://t.co/YViSKLGYae">pic.twitter.com/YViSKLGYae</a></p>&mdash; Mirco Ravanelli (@mirco_ravanelli) <a href="https://twitter.com/mirco_ravanelli/status/1403003737128460290?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Want to know more about <a href="https://twitter.com/SpeechBrain1?ref_src=twsrc%5Etfw">@SpeechBrain1</a>? Just have a look at the new paper 👀👀<br><br>pdf: <a href="https://t.co/op2i0eXi9r">https://t.co/op2i0eXi9r</a><br>abs: <a href="https://t.co/HLFUQ4vSIi">https://t.co/HLFUQ4vSIi</a><br><br>Tutorials: <a href="https://t.co/30eh9ZOdz6">https://t.co/30eh9ZOdz6</a><br>GitHub: <a href="https://t.co/xN0veKMFqp">https://t.co/xN0veKMFqp</a><br>HuggingFace: <a href="https://t.co/IElc6nYvzN">https://t.co/IElc6nYvzN</a><br>Website: <a href="https://t.co/LXOB2scbSR">https://t.co/LXOB2scbSR</a> <a href="https://t.co/iKbGtHdl6Q">pic.twitter.com/iKbGtHdl6Q</a></p>&mdash; Titouan Parcollet (@ParcolletT) <a href="https://twitter.com/ParcolletT/status/1402887200426102786?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SpeechBrain: A General-Purpose Speech Toolkit<br>pdf: <a href="https://t.co/DmBMdLniMF">https://t.co/DmBMdLniMF</a><br>abs: <a href="https://t.co/P47ckJ22md">https://t.co/P47ckJ22md</a><br><br>open-source and all-in-one speech toolkit <a href="https://t.co/pwANWxBnFT">pic.twitter.com/pwANWxBnFT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402795828713033728?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Pretrained Encoders are All You Need

Mina Khan, P Srivatsa, Advait Rane, Shriram Chenniappa, Rishabh Anand, Sherjil Ozair, Pattie Maes

- retweets: 960, favorites: 126 (06/11/2021 09:50:31)

- links: [abs](https://arxiv.org/abs/2106.05139) | [pdf](https://arxiv.org/pdf/2106.05139)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Data-efficiency and generalization are key challenges in deep learning and deep reinforcement learning as many models are trained on large-scale, domain-specific, and expensive-to-label datasets. Self-supervised models trained on large-scale uncurated datasets have shown successful transfer to diverse settings. We investigate using pretrained image representations and spatio-temporal attention for state representation learning in Atari. We also explore fine-tuning pretrained representations with self-supervised techniques, i.e., contrastive predictive coding, spatio-temporal contrastive learning, and augmentations. Our results show that pretrained representations are at par with state-of-the-art self-supervised methods trained on domain-specific data. Pretrained representations, thus, yield data and compute-efficient state representations. https://github.com/PAL-ML/PEARL_v1

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pretrained Encoders are All You Need<br>pdf: <a href="https://t.co/61H9Es76xA">https://t.co/61H9Es76xA</a><br>abs: <a href="https://t.co/nORLGMoKvr">https://t.co/nORLGMoKvr</a><br>github: <a href="https://t.co/d0nQVCmwk5">https://t.co/d0nQVCmwk5</a> <a href="https://t.co/MG3gU4pdNR">pic.twitter.com/MG3gU4pdNR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402793961886998530?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Pretraining Representations for Data-Efficient Reinforcement Learning

Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, Devon Hjelm, Philip Bachman, Aaron Courville

- retweets: 667, favorites: 198 (06/11/2021 09:50:31)

- links: [abs](https://arxiv.org/abs/2106.04799) | [pdf](https://arxiv.org/pdf/2106.04799)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Data efficiency is a key challenge for deep reinforcement learning. We address this problem by using unlabeled data to pretrain an encoder which is then finetuned on a small amount of task-specific data. To encourage learning representations which capture diverse aspects of the underlying MDP, we employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL. When limited to 100k steps of interaction on Atari games (equivalent to two hours of human experience), our approach significantly surpasses prior work combining offline representation pretraining with task-specific finetuning, and compares favourably with other pretraining methods that require orders of magnitude more data. Our approach shows particular promise when combined with larger models as well as more diverse, task-aligned observational data -- approaching human-level performance and data-efficiency on Atari in our best setting. We provide code associated with this work at https://github.com/mila-iqia/SGI.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep RL agents usually start from tabula rasa, and struggle to match the data efficiency of humans who rely on strong priors. Can we even the playing field by starting agents off with strong representations of their environments?<br><br>We certainly think so: <a href="https://t.co/qttjqn7Yhf">https://t.co/qttjqn7Yhf</a> <a href="https://t.co/EbTjr6vzl0">pic.twitter.com/EbTjr6vzl0</a></p>&mdash; Max Schwarzer (@max_a_schwarzer) <a href="https://twitter.com/max_a_schwarzer/status/1403021936142086145?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pretraining Representations for Data-Efficient Reinforcement Learning<br><br>Proposes SGI, which significantly surpasses prior work on Atari with the steps limited to 100k with an improved unsupervised goal-conditioned RL.<br><br>abs: <a href="https://t.co/27XPu6NajO">https://t.co/27XPu6NajO</a><br>code: <a href="https://t.co/758gWsD2yK">https://t.co/758gWsD2yK</a> <a href="https://t.co/jpGYoxYMim">pic.twitter.com/jpGYoxYMim</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1402797357310242821?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pretraining Representations for Data-Efficient Reinforcement Learning<br>pdf: <a href="https://t.co/inaFoYhQlY">https://t.co/inaFoYhQlY</a><br>abs: <a href="https://t.co/ekXdkFikqF">https://t.co/ekXdkFikqF</a><br>github: <a href="https://t.co/Cj9ml9bbv6">https://t.co/Cj9ml9bbv6</a><br><br>uses a combination of pretraining objectives to encourage the agent to learn multiple aspects of environment dynamics <a href="https://t.co/g7dsofWDG9">pic.twitter.com/g7dsofWDG9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402794735685804034?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain  Adaptation

David Berthelot, Rebecca Roelofs, Kihyuk Sohn, Nicholas Carlini, Alex Kurakin

- retweets: 616, favorites: 111 (06/11/2021 09:50:32)

- links: [abs](https://arxiv.org/abs/2106.04732) | [pdf](https://arxiv.org/pdf/2106.04732)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We extend semi-supervised learning to the problem of domain adaptation to learn significantly higher-accuracy models that train on one data distribution and test on a different one. With the goal of generality, we introduce AdaMatch, a method that unifies the tasks of unsupervised domain adaptation (UDA), semi-supervised learning (SSL), and semi-supervised domain adaptation (SSDA). In an extensive experimental study, we compare its behavior with respective state-of-the-art techniques from SSL, SSDA, and UDA on vision classification tasks. We find AdaMatch either matches or significantly exceeds the state-of-the-art in each case using the same hyper-parameters regardless of the dataset or task. For example, AdaMatch nearly doubles the accuracy compared to that of the prior state-of-the-art on the UDA task for DomainNet and even exceeds the accuracy of the prior state-of-the-art obtained with pre-training by 6.4% when AdaMatch is trained completely from scratch. Furthermore, by providing AdaMatch with just one labeled example per class from the target domain (i.e., the SSDA setting), we increase the target accuracy by an additional 6.1%, and with 5 labeled examples, by 13.6%.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AdaMatch: A Unified Approach to Semi-Supervised<br>Learning and Domain Adaptation<br>pdf: <a href="https://t.co/9BUJQK3SdZ">https://t.co/9BUJQK3SdZ</a><br>abs: <a href="https://t.co/G1AOXAopye">https://t.co/G1AOXAopye</a><br><br>a general method designed to boost accuracy on domain shifts when given access to unlabeled data from the new domain <a href="https://t.co/n8PCfST3ql">pic.twitter.com/n8PCfST3ql</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402825475471327237?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: AdaMatch - Unifying Unsupervised Domain Adaptation (UDA) and Semi-Supervised Learning (SSL) and SSDA. Nearly doubles SotA accuracy for UDA on non-pretrained DomainNet. <a href="https://t.co/F0huwzXvhf">https://t.co/F0huwzXvhf</a><br>1/3</p>&mdash; David Berthelot (@D_Berthelot_ML) <a href="https://twitter.com/D_Berthelot_ML/status/1403140600480747522?ref_src=twsrc%5Etfw">June 11, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via  Relabeling Experience and Unsupervised Pre-training

Kimin Lee, Laura Smith, Pieter Abbeel

- retweets: 358, favorites: 117 (06/11/2021 09:50:32)

- links: [abs](https://arxiv.org/abs/2106.05091) | [pdf](https://arxiv.org/pdf/2106.05091)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Conveying complex objectives to reinforcement learning (RL) agents can often be difficult, involving meticulous design of reward functions that are sufficiently informative yet easy enough to provide. Human-in-the-loop RL methods allow practitioners to instead interactively teach agents through tailored feedback; however, such approaches have been challenging to scale since human feedback is very expensive. In this work, we aim to make this process more sample- and feedback-efficient. We present an off-policy, interactive RL algorithm that capitalizes on the strengths of both feedback and off-policy learning. Specifically, we learn a reward model by actively querying a teacher's preferences between two clips of behavior and use it to train an agent. To enable off-policy learning, we relabel all the agent's past experience when its reward model changes. We additionally show that pre-training our agents with unsupervised exploration substantially increases the mileage of its queries. We demonstrate that our approach is capable of learning tasks of higher complexity than previously considered by human-in-the-loop methods, including a variety of locomotion and robotic manipulation skills. We also show that our method is able to utilize real-time human feedback to effectively prevent reward exploitation and learn new behaviors that are difficult to specify with standard reward functions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we learn policies using human feedback without pre-defined rewards efficiently?<br> <br>We find unsupervised RL and off-policy learning can improve the preference-based RL in PEBBLE!<br> <br>📑Paper: <a href="https://t.co/4NqjsWl1SJ">https://t.co/4NqjsWl1SJ</a><br>💻Code &amp; video: <a href="https://t.co/cdzy6QZayx">https://t.co/cdzy6QZayx</a><br>w/ Laura Smith, <a href="https://twitter.com/pabbeel?ref_src=twsrc%5Etfw">@pabbeel</a> <a href="https://t.co/OmXO9U9Z8i">pic.twitter.com/OmXO9U9Z8i</a></p>&mdash; Kimin (@kimin_le2) <a href="https://twitter.com/kimin_le2/status/1403025578043666432?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training<br>pdf: <a href="https://t.co/ZSP5uNGCgl">https://t.co/ZSP5uNGCgl</a><br>abs: <a href="https://t.co/548k5SKXfY">https://t.co/548k5SKXfY</a><br>project page: <a href="https://t.co/QCDtMv7OeJ">https://t.co/QCDtMv7OeJ</a><br>github: <a href="https://t.co/oncteZAPgC">https://t.co/oncteZAPgC</a> <a href="https://t.co/sheykWyKL6">pic.twitter.com/sheykWyKL6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402819850381189121?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. FastSeq: Make Sequence Generation Faster

Yu Yan, Fei Hu, Jiusheng Chen, Nikhil Bhendawade, Ting Ye, Yeyun Gong, Nan Duan, Desheng Cui, Bingyu Chi, Ruifei Zhang

- retweets: 281, favorites: 107 (06/11/2021 09:50:32)

- links: [abs](https://arxiv.org/abs/2106.04718) | [pdf](https://arxiv.org/pdf/2106.04718)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformer-based models have made tremendous impacts in natural language generation. However the inference speed is a bottleneck due to large model size and intensive computing involved in auto-regressive decoding process. We develop FastSeq framework to accelerate sequence generation without accuracy loss. The proposed optimization techniques include an attention cache optimization, an efficient algorithm for detecting repeated n-grams, and an asynchronous generation pipeline with parallel I/O. These optimizations are general enough to be applicable to Transformer-based models (e.g., T5, GPT2, and UniLM). Our benchmark results on a set of widely used and diverse models demonstrate 4-9x inference speed gain. Additionally, FastSeq is easy to use with a simple one-line code change. The source code is available at https://github.com/microsoft/fastseq.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FastSeq: Make Sequence Generation Faster<br><br>Demonstrates 4-9x inference speed gain on various Transformer-based models with a series of optimization methods.<br><br>abs: <a href="https://t.co/U5opo0VxSf">https://t.co/U5opo0VxSf</a><br>code: <a href="https://t.co/oH0GCkadYD">https://t.co/oH0GCkadYD</a> <a href="https://t.co/USsul0lMAB">pic.twitter.com/USsul0lMAB</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1402793303238615045?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">FastSeq: Make Sequence Generation Faster<br>pdf: <a href="https://t.co/oP1I6IfxGN">https://t.co/oP1I6IfxGN</a><br>abs: <a href="https://t.co/6GOJGfQa3I">https://t.co/6GOJGfQa3I</a><br>github: <a href="https://t.co/fqczhpNqNa">https://t.co/fqczhpNqNa</a><br><br>provides general solutions for speeding up the sequence generation without accuracy loss <a href="https://t.co/Nfiw6ou928">pic.twitter.com/Nfiw6ou928</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402791464673976321?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. NeRF in detail: Learning to sample for view synthesis

Relja Arandjelović, Andrew Zisserman

- retweets: 191, favorites: 147 (06/11/2021 09:50:33)

- links: [abs](https://arxiv.org/abs/2106.05264) | [pdf](https://arxiv.org/pdf/2106.05264)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Neural radiance fields (NeRF) methods have demonstrated impressive novel view synthesis performance. The core approach is to render individual rays by querying a neural network at points sampled along the ray to obtain the density and colour of the sampled points, and integrating this information using the rendering equation. Since dense sampling is computationally prohibitive, a common solution is to perform coarse-to-fine sampling.   In this work we address a clear limitation of the vanilla coarse-to-fine approach -- that it is based on a heuristic and not trained end-to-end for the task at hand. We introduce a differentiable module that learns to propose samples and their importance for the fine network, and consider and compare multiple alternatives for its neural architecture. Training the proposal module from scratch can be unstable due to lack of supervision, so an effective pre-training strategy is also put forward. The approach, named `NeRF in detail' (NeRF-ID), achieves superior view synthesis quality over NeRF and the state-of-the-art on the synthetic Blender benchmark and on par or better performance on the real LLFF-NeRF scenes. Furthermore, by leveraging the predicted sample importance, a 25% saving in computation can be achieved without significantly sacrificing the rendering quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our new paper &quot;NeRF in detail: Learning to sample for view synthesis&quot; (aka yet another NeRF paper on your to-read list) we replace the heuristic coarse-to-fine strategy of NeRF via a learnt one. Improvements in rendering quality and speed. <a href="https://t.co/434YoDKrFZ">https://t.co/434YoDKrFZ</a> <a href="https://t.co/BLclVqYbK5">pic.twitter.com/BLclVqYbK5</a></p>&mdash; Relja Arandjelović (@relja_work) <a href="https://twitter.com/relja_work/status/1402907147181101059?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeRF in detail: Learning to sample for view synthesis<br>pdf: <a href="https://t.co/h4qxLHFthk">https://t.co/h4qxLHFthk</a><br>abs: <a href="https://t.co/Vf7BvqS6sK">https://t.co/Vf7BvqS6sK</a><br><br>a ‘proposer’ module that learns the hierarchical coarse-to-fine sampling, thus enabling NeRF to be trained end-to-end for the view synthesis task <a href="https://t.co/0pp7R4qsZr">pic.twitter.com/0pp7R4qsZr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402792194294095873?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Generative Models as a Data Source for Multiview Representation Learning

Ali Jahanian, Xavier Puig, Yonglong Tian, Phillip Isola

- retweets: 225, favorites: 95 (06/11/2021 09:50:33)

- links: [abs](https://arxiv.org/abs/2106.05258) | [pdf](https://arxiv.org/pdf/2106.05258)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative models are now capable of producing highly realistic images that look nearly indistinguishable from the data on which they are trained. This raises the question: if we have good enough generative models, do we still need datasets? We investigate this question in the setting of learning general-purpose visual representations from a black-box generative model rather than directly from data. Given an off-the-shelf image generator without any access to its training data, we train representations from the samples output by this generator. We compare several representation learning methods that can be applied to this setting, using the latent space of the generator to generate multiple "views" of the same semantic content. We show that for contrastive methods, this multiview data can naturally be used to identify positive pairs (nearby in latent space) and negative pairs (far apart in latent space). We find that the resulting representations rival those learned directly from real data, but that good performance requires care in the sampling strategy applied and the training method. Generative models can be viewed as a compressed and organized copy of a dataset, and we envision a future where more and more "model zoos" proliferate while datasets become increasingly unwieldy, missing, or private. This paper suggests several techniques for dealing with visual representation learning in such a future. Code is released on our project page: https://ali-design.github.io/GenRep/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generative Models as a Data Source for Multiview Representation Learning<br>pdf: <a href="https://t.co/VGQRmL0BS1">https://t.co/VGQRmL0BS1</a><br>abs: <a href="https://t.co/50gFZE92QQ">https://t.co/50gFZE92QQ</a><br>project page: <a href="https://t.co/FfqqQKTWCY">https://t.co/FfqqQKTWCY</a> <a href="https://t.co/rRo5HrZD3i">pic.twitter.com/rRo5HrZD3i</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402802015592402949?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">データセットから生成モデルを学習し、生成したデータのみから表現学習する。潜在変数上での近傍を正例としさらに画像上でオーグメンテーションを適用した方が良い表現が得られる。元のデータ上で学習した場合に近い性能が出るが超えず、サンプル数増で精度改善するがサチる <a href="https://t.co/x7urGJcndz">https://t.co/x7urGJcndz</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1403133169348472836?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Point Cloud Upsampling via Disentangled Refinement

Ruihui Li, Xianzhi Li, Pheng-Ann Heng, Chi-Wing Fu

- retweets: 196, favorites: 59 (06/11/2021 09:50:33)

- links: [abs](https://arxiv.org/abs/2106.04779) | [pdf](https://arxiv.org/pdf/2106.04779)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Point clouds produced by 3D scanning are often sparse, non-uniform, and noisy. Recent upsampling approaches aim to generate a dense point set, while achieving both distribution uniformity and proximity-to-surface, and possibly amending small holes, all in a single network. After revisiting the task, we propose to disentangle the task based on its multi-objective nature and formulate two cascaded sub-networks, a dense generator and a spatial refiner. The dense generator infers a coarse but dense output that roughly describes the underlying surface, while the spatial refiner further fine-tunes the coarse output by adjusting the location of each point. Specifically, we design a pair of local and global refinement units in the spatial refiner to evolve a coarse feature map. Also, in the spatial refiner, we regress a per-point offset vector to further adjust the coarse outputs in fine-scale. Extensive qualitative and quantitative results on both synthetic and real-scanned datasets demonstrate the superiority of our method over the state-of-the-arts.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Point Cloud Upsampling via Disentangled Refinement<br>pdf: <a href="https://t.co/BdfjRbBQfZ">https://t.co/BdfjRbBQfZ</a><br>abs: <a href="https://t.co/iIjoyyd2nL">https://t.co/iIjoyyd2nL</a><br><br>disentangle the task based on its multi-objective nature and formulate two cascaded sub-networks, a dense generator and a spatial refiner <a href="https://t.co/qi5mkwNOm0">pic.twitter.com/qi5mkwNOm0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402830909083164675?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Vector Quantized Models for Planning

Sherjil Ozair, Yazhe Li, Ali Razavi, Ioannis Antonoglou, Aäron van den Oord, Oriol Vinyals

- retweets: 110, favorites: 83 (06/11/2021 09:50:33)

- links: [abs](https://arxiv.org/abs/2106.04615) | [pdf](https://arxiv.org/pdf/2106.04615)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recent developments in the field of model-based RL have proven successful in a range of environments, especially ones where planning is essential. However, such successes have been limited to deterministic fully-observed environments. We present a new approach that handles stochastic and partially-observable environments. Our key insight is to use discrete autoencoders to capture the multiple possible effects of an action in a stochastic environment. We use a stochastic variant of Monte Carlo tree search to plan over both the agent's actions and the discrete latent variables representing the environment's response. Our approach significantly outperforms an offline version of MuZero on a stochastic interpretation of chess where the opponent is considered part of the environment. We also show that our approach scales to DeepMind Lab, a first-person 3D environment with large visual observations and partial observability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vector Quantized Models for Planning<br>pdf: <a href="https://t.co/dWHtizegNQ">https://t.co/dWHtizegNQ</a><br>abs: <a href="https://t.co/MD4oilCux2">https://t.co/MD4oilCux2</a><br>project page: <a href="https://t.co/HCxGZ10BUN">https://t.co/HCxGZ10BUN</a><br><br>outperforms modelfree baselines and performs competitively against offline MuZero and Stockfish Level 15 while being a more general algorithm <a href="https://t.co/HLMjrMDcra">pic.twitter.com/HLMjrMDcra</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402799101637447681?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Geometry-Consistent Neural Shape Representation with Implicit  Displacement Fields

Wang Yifan, Lukas Rahmann, Olga Sorkine-Hornung

- retweets: 132, favorites: 60 (06/11/2021 09:50:33)

- links: [abs](https://arxiv.org/abs/2106.05187) | [pdf](https://arxiv.org/pdf/2106.05187)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present implicit displacement fields, a novel representation for detailed 3D geometry. Inspired by a classic surface deformation technique, displacement mapping, our method represents a complex surface as a smooth base surface plus a displacement along the base's normal directions, resulting in a frequency-based shape decomposition, where the high frequency signal is constrained geometrically by the low frequency signal. Importantly, this disentanglement is unsupervised thanks to a tailored architectural design that has an innate frequency hierarchy by construction. We explore implicit displacement field surface reconstruction and detail transfer and demonstrate superior representational power, training stability and generalizability.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields<br>pdf: <a href="https://t.co/SqfRmlgILy">https://t.co/SqfRmlgILy</a><br>abs: <a href="https://t.co/zRApiIKl3o">https://t.co/zRApiIKl3o</a><a href="https://twitter.com/toomanyyifans?ref_src=twsrc%5Etfw">@toomanyyifans</a><br>novel parameterization of neural implicit shape representation based on displacement mapping for detailed geometry <a href="https://t.co/rFS5YviVCI">pic.twitter.com/rFS5YviVCI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1403011155581276160?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian Approach

Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, Anna Wienhard

- retweets: 56, favorites: 46 (06/11/2021 09:50:34)

- links: [abs](https://arxiv.org/abs/2106.04941) | [pdf](https://arxiv.org/pdf/2106.04941)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CG](https://arxiv.org/list/cs.CG/recent)

Learning faithful graph representations as sets of vertex embeddings has become a fundamental intermediary step in a wide range of machine learning applications. We propose the systematic use of symmetric spaces in representation learning, a class encompassing many of the previously used embedding targets. This enables us to introduce a new method, the use of Finsler metrics integrated in a Riemannian optimization scheme, that better adapts to dissimilar structures in the graph. We develop a tool to analyze the embeddings and infer structural properties of the data sets. For implementation, we choose Siegel spaces, a versatile family of symmetric spaces. Our approach outperforms competitive baselines for graph reconstruction tasks on various synthetic and real-world datasets. We further demonstrate its applicability on two downstream tasks, recommender systems and node classification.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Would you like to embed graphs in a space that simultaneously contains Euclidean and hyperbolic subspaces, products thereof, and SPD submanifolds? 🤯<br><br>Happy to share our work on Symmetric Spaces for Graph Embeddings, to be presented at <a href="https://twitter.com/icmlconf?ref_src=twsrc%5Etfw">@icmlconf</a>: <a href="https://t.co/Nip7GnTaar">https://t.co/Nip7GnTaar</a> (1/5) <a href="https://t.co/yMu3HgN2cI">pic.twitter.com/yMu3HgN2cI</a></p>&mdash; Federico López (@fedelopez77) <a href="https://twitter.com/fedelopez77/status/1402928404920520705?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Recovering AES Keys with a Deep Cold Boot Attack

Itamar Zimerman, Eliya Nachmani, Lior Wolf

- retweets: 30, favorites: 33 (06/11/2021 09:50:34)

- links: [abs](https://arxiv.org/abs/2106.04876) | [pdf](https://arxiv.org/pdf/2106.04876)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Cold boot attacks inspect the corrupted random access memory soon after the power has been shut down. While most of the bits have been corrupted, many bits, at random locations, have not. Since the keys in many encryption schemes are being expanded in memory into longer keys with fixed redundancies, the keys can often be restored. In this work, we combine a novel cryptographic variant of a deep error correcting code technique with a modified SAT solver scheme to apply the attack on AES keys. Even though AES consists of Rijndael S-box elements, that are specifically designed to be resistant to linear and differential cryptanalysis, our method provides a novel formalization of the AES key scheduling as a computational graph, which is implemented by a neural message passing network. Our results show that our methods outperform the state of the art attack methods by a very large margin.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our DeepCrypto paper &quot;Recovering AES Keys with a Deep Cold Boot Attack&quot;. Accepted to ICML 2021. w/ <a href="https://twitter.com/ItamarZimerman?ref_src=twsrc%5Etfw">@ItamarZimerman</a> &amp; Lior Wolf. TD;LR AES attack with neural S-box.<a href="https://t.co/gmdFLGQ1fb">https://t.co/gmdFLGQ1fb</a><br> <a href="https://twitter.com/TelAvivUni?ref_src=twsrc%5Etfw">@TelAvivUni</a>  <a href="https://twitter.com/facebookai?ref_src=twsrc%5Etfw">@facebookai</a> <a href="https://twitter.com/icmlconf?ref_src=twsrc%5Etfw">@icmlconf</a> <a href="https://t.co/bq2k9o6ZHM">pic.twitter.com/bq2k9o6ZHM</a></p>&mdash; Eliya Nachmani (@NachmaniEliya) <a href="https://twitter.com/NachmaniEliya/status/1402871902205616129?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Neural Extractive Search

Shauli Ravfogel, Hillel Taub-Tabib, Yoav Goldberg

- retweets: 30, favorites: 27 (06/11/2021 09:50:34)

- links: [abs](https://arxiv.org/abs/2106.04612) | [pdf](https://arxiv.org/pdf/2106.04612)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Domain experts often need to extract structured information from large corpora. We advocate for a search paradigm called ``extractive search'', in which a search query is enriched with capture-slots, to allow for such rapid extraction. Such an extractive search system can be built around syntactic structures, resulting in high-precision, low-recall results. We show how the recall can be improved using neural retrieval and alignment. The goals of this paper are to concisely introduce the extractive-search paradigm; and to demonstrate a prototype neural retrieval system for extractive search and its benefits and potential. Our prototype is available at \url{https://spike.neural-sim.apps.allenai.org/} and a video demonstration is available at \url{https://vimeo.com/559586687}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Extractive Search<br>pdf: <a href="https://t.co/LH8qulaqfN">https://t.co/LH8qulaqfN</a><br>abs: <a href="https://t.co/KS8MX3wKo5">https://t.co/KS8MX3wKo5</a><br>prototype: <a href="https://t.co/FUYUOujoh5">https://t.co/FUYUOujoh5</a> <a href="https://t.co/tDaHJvxeke">pic.twitter.com/tDaHJvxeke</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1402804754808512513?ref_src=twsrc%5Etfw">June 10, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



