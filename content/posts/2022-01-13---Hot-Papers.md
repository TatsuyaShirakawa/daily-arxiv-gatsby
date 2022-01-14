---
title: Hot Papers 2022-01-13
date: 2022-01-14T11:10:50.Z
template: "post"
draft: false
slug: "hot-papers-2022-01-13"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2022-01-13"
socialImage: "/media/flying-marine.jpg"

---

# 1. HyperTransformer: Model Generation for Supervised and Semi-Supervised  Few-Shot Learning

Andrey Zhmoginov, Mark Sandler, Max Vladymyrov

- retweets: 1343, favorites: 239 (01/14/2022 11:10:50)

- links: [abs](https://arxiv.org/abs/2201.04182) | [pdf](https://arxiv.org/pdf/2201.04182)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work we propose a HyperTransformer, a transformer-based model for few-shot learning that generates weights of a convolutional neural network (CNN) directly from support samples. Since the dependence of a small generated CNN model on a specific task is encoded by a high-capacity transformer model, we effectively decouple the complexity of the large task space from the complexity of individual tasks. Our method is particularly effective for small target CNN architectures where learning a fixed universal task-independent embedding is not optimal and better performance is attained when the information about the task can modulate all model parameters. For larger models we discover that generating the last layer alone allows us to produce competitive or better results than those obtained with state-of-the-art methods while being end-to-end differentiable. Finally, we extend our approach to a semi-supervised regime utilizing unlabeled samples in the support set and further improving few-shot performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning<br>abs: <a href="https://t.co/MFAMb3VnM1">https://t.co/MFAMb3VnM1</a><br><br>a transformer-based model for few-shot learning that generates weights of a convolutional neural network directly from support samples <a href="https://t.co/KTCN45rXnt">pic.twitter.com/KTCN45rXnt</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481443284304154625?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning<br><br>By Andrey Zhmoginov, Mark Sandler, <a href="https://twitter.com/mvladymyrov?ref_src=twsrc%5Etfw">@mvladymyrov</a><br><br>A Transformer trained to produce all of the model parameters of a small ConvNet works really well for meta-learning!<a href="https://t.co/QgefXnWo7b">https://t.co/QgefXnWo7b</a> <a href="https://t.co/OEz2sCtw5n">pic.twitter.com/OEz2sCtw5n</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1481787736264634368?ref_src=twsrc%5Etfw">January 14, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Deep Symbolic Regression for Recurrent Sequences

Stéphane d'Ascoli, Pierre-Alexandre Kamienny, Guillaume Lample, François Charton

- retweets: 535, favorites: 166 (01/14/2022 11:10:50)

- links: [abs](https://arxiv.org/abs/2201.04600) | [pdf](https://arxiv.org/pdf/2201.04600)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Symbolic regression, i.e. predicting a function from the observation of its values, is well-known to be a challenging task. In this paper, we train Transformers to infer the function or recurrence relation underlying sequences of integers or floats, a typical task in human IQ tests which has hardly been tackled in the machine learning literature. We evaluate our integer model on a subset of OEIS sequences, and show that it outperforms built-in Mathematica functions for recurrence prediction. We also demonstrate that our float model is able to yield informative approximations of out-of-vocabulary functions and constants, e.g. $\operatorname{bessel0}(x)\approx \frac{\sin(x)+\cos(x)}{\sqrt{\pi x}}$ and $1.644934\approx \pi^2/6$. An interactive demonstration of our models is provided at https://bit.ly/3niE5FS.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ArXiv <a href="https://t.co/TQ6mPLX4wf">https://t.co/TQ6mPLX4wf</a>: Predicting symbolic functions from values via transformers: outperforms built-in Mathematica functions for recurrence prediction. Gives approximations to different functions which might be useful for efficient implementations.</p>&mdash; Sepp Hochreiter (@HochreiterSepp) <a href="https://twitter.com/HochreiterSepp/status/1481516573458419713?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Multiview Transformers for Video Recognition

Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, Cordelia Schmid

- retweets: 468, favorites: 163 (01/14/2022 11:10:51)

- links: [abs](https://arxiv.org/abs/2201.04288) | [pdf](https://arxiv.org/pdf/2201.04288)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Video understanding requires reasoning at multiple spatiotemporal resolutions -- from short fine-grained motions to events taking place over longer durations. Although transformer architectures have recently advanced the state-of-the-art, they have not explicitly modelled different spatiotemporal resolutions. To this end, we present Multiview Transformers for Video Recognition (MTV). Our model consists of separate encoders to represent different views of the input video with lateral connections to fuse information across views. We present thorough ablation studies of our model and show that MTV consistently performs better than single-view counterparts in terms of accuracy and computational cost across a range of model sizes. Furthermore, we achieve state-of-the-art results on five standard datasets, and improve even further with large-scale pretraining. We will release code and pretrained checkpoints.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multiview Transformers for Video Recognition<br>abs: <a href="https://t.co/Qh5tOu6VHd">https://t.co/Qh5tOu6VHd</a> <a href="https://t.co/Fn6abJNB06">pic.twitter.com/Fn6abJNB06</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481445966532259840?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. PromptBERT: Improving BERT Sentence Embeddings with Prompts

Ting Jiang, Shaohan Huang, Zihan Zhang, Deqing Wang, Fuzhen Zhuang, Furu Wei, Haizhen Huang, Liangjie Zhang, Qi Zhang

- retweets: 219, favorites: 136 (01/14/2022 11:10:51)

- links: [abs](https://arxiv.org/abs/2201.04337) | [pdf](https://arxiv.org/pdf/2201.04337)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The poor performance of the original BERT for sentence semantic similarity has been widely discussed in previous works. We find that unsatisfactory performance is mainly due to the static token embeddings biases and the ineffective BERT layers, rather than the high cosine similarity of the sentence embeddings. To this end, we propose a prompt based sentence embeddings method which can reduce token embeddings biases and make the original BERT layers more effective. By reformulating the sentence embeddings task as the fillin-the-blanks problem, our method significantly improves the performance of original BERT. We discuss two prompt representing methods and three prompt searching methods for prompt based sentence embeddings. Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised setting. For experiments, we evaluate our method on both non fine-tuned and fine-tuned settings. Even a non fine-tuned method can outperform the fine-tuned methods like unsupervised ConSERT on STS tasks. Our fine-tuned method outperforms the state-of-the-art method SimCSE in both unsupervised and supervised settings. Compared to SimCSE, we achieve 2.29 and 2.58 points improvements on BERT and RoBERTa respectively under the unsupervised setting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PromptBERT: Improving BERT Sentence Embeddings with Prompts<br>abs: <a href="https://t.co/N1EQlKRJAi">https://t.co/N1EQlKRJAi</a><br>github: <a href="https://t.co/hrTlbFFks4">https://t.co/hrTlbFFks4</a><br><br>Compared to SimCSE, achieve 2.29 and 2.58 points improvements on BERT and RoBERTa respectively under<br>the unsupervised setting <a href="https://t.co/UkhC9F6jTq">pic.twitter.com/UkhC9F6jTq</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481450080372731906?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PromptBERT proposes a new prompt-based sentence embeddings method to improve the effectiveness of the original BERT layers. <a href="https://t.co/LWgafohEt4">https://t.co/LWgafohEt4</a><br><br>ML engineers, it&#39;s time to start paying attention to prompt-based methods. <a href="https://t.co/SjDjcXC1ep">pic.twitter.com/SjDjcXC1ep</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1481738145863548932?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Robust Contrastive Learning against Noisy Views

Ching-Yao Chuang, R Devon Hjelm, Xin Wang, Vibhav Vineet, Neel Joshi, Antonio Torralba, Stefanie Jegelka, Yale Song

- retweets: 160, favorites: 116 (01/14/2022 11:10:52)

- links: [abs](https://arxiv.org/abs/2201.04309) | [pdf](https://arxiv.org/pdf/2201.04309)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Contrastive learning relies on an assumption that positive pairs contain related views, e.g., patches of an image or co-occurring multimodal signals of a video, that share certain underlying information about an instance. But what if this assumption is violated? The literature suggests that contrastive learning produces suboptimal representations in the presence of noisy views, e.g., false positive pairs with no apparent shared information. In this work, we propose a new contrastive loss function that is robust against noisy views. We provide rigorous theoretical justifications by showing connections to robust symmetric losses for noisy binary classification and by establishing a new contrastive bound for mutual information maximization based on the Wasserstein distance measure. The proposed loss is completely modality-agnostic and a simple drop-in replacement for the InfoNCE loss, which makes it easy to apply to existing contrastive frameworks. We show that our approach provides consistent improvements over the state-of-the-art on image, video, and graph contrastive learning benchmarks that exhibit a variety of real-world noise patterns.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robust Contrastive Learning against Noisy Views<br>abs: <a href="https://t.co/vyY5yTfdrc">https://t.co/vyY5yTfdrc</a> <a href="https://t.co/HVQHYniuCT">pic.twitter.com/HVQHYniuCT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481467943892492289?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ArXiv <a href="https://t.co/m018GTNxAL">https://t.co/m018GTNxAL</a>: New objective for contrastive learning: between InfoNCE and CLUB (upper bound on mutual information) but still a lower bound on the mutual information. Consistent improvements over the state-of-the-art contrastive learning benchmarks.</p>&mdash; Sepp Hochreiter (@HochreiterSepp) <a href="https://twitter.com/HochreiterSepp/status/1481535086625538049?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Incidents1M: a large-scale dataset of images with natural disasters,  damage, and incidents

Ethan Weber, Dim P. Papadopoulos, Agata Lapedriza, Ferda Ofli, Muhammad Imran, Antonio Torralba

- retweets: 180, favorites: 65 (01/14/2022 11:10:53)

- links: [abs](https://arxiv.org/abs/2201.04236) | [pdf](https://arxiv.org/pdf/2201.04236)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Natural disasters, such as floods, tornadoes, or wildfires, are increasingly pervasive as the Earth undergoes global warming. It is difficult to predict when and where an incident will occur, so timely emergency response is critical to saving the lives of those endangered by destructive events. Fortunately, technology can play a role in these situations. Social media posts can be used as a low-latency data source to understand the progression and aftermath of a disaster, yet parsing this data is tedious without automated methods. Prior work has mostly focused on text-based filtering, yet image and video-based filtering remains largely unexplored. In this work, we present the Incidents1M Dataset, a large-scale multi-label dataset which contains 977,088 images, with 43 incident and 49 place categories. We provide details of the dataset construction, statistics and potential biases; introduce and train a model for incident detection; and perform image-filtering experiments on millions of images on Flickr and Twitter. We also present some applications on incident analysis to encourage and enable future work in computer vision for humanitarian aid. Code, data, and models are available at http://incidentsdataset.csail.mit.edu.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Incidents1M: a large-scale dataset of images with natural disasters, damage, and incidents<br>abs: <a href="https://t.co/ehdybJKTr5">https://t.co/ehdybJKTr5</a><br>project page: <a href="https://t.co/HnfvJnYcSM">https://t.co/HnfvJnYcSM</a> <a href="https://t.co/6Q2vHRfCZm">pic.twitter.com/6Q2vHRfCZm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481471033437741059?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Neural Residual Flow Fields for Efficient Video Representations

Daniel Rho, Junwoo Cho, Jong Hwan Ko, Eunbyung Park

- retweets: 48, favorites: 47 (01/14/2022 11:10:53)

- links: [abs](https://arxiv.org/abs/2201.04329) | [pdf](https://arxiv.org/pdf/2201.04329)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Implicit neural representation (INR) has emerged as a powerful paradigm for representing signals, such as images, videos, 3D shapes, etc. Although it has shown the ability to represent fine details, its efficiency as a data representation has not been extensively studied. In INR, the data is stored in the form of parameters of a neural network and general purpose optimization algorithms do not generally exploit the spatial and temporal redundancy in signals. In this paper, we suggest a novel INR approach to representing and compressing videos by explicitly removing data redundancy. Instead of storing raw RGB colors, we propose Neural Residual Flow Fields (NRFF), using motion information across video frames and residuals that are necessary to reconstruct a video. Maintaining the motion information, which is usually smoother and less complex than the raw signals, requires far fewer parameters. Furthermore, reusing redundant pixel values further improves the network parameter efficiency. Experimental results have shown that the proposed method outperforms the baseline methods by a significant margin. The code is available in https://github.com/daniel03c1/eff_video_representation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Neural Residual Flow Fields for Efficient Video Representations<br>abs: <a href="https://t.co/PNhAFRugPE">https://t.co/PNhAFRugPE</a><br>github: <a href="https://t.co/qeQo2r1FWv">https://t.co/qeQo2r1FWv</a> <a href="https://t.co/l7m9mHBuYF">pic.twitter.com/l7m9mHBuYF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481448191405223938?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Real-Time Style Modelling of Human Locomotion via Feature-Wise  Transformations and Local Motion Phases

Ian Mason, Sebastian Starke, Taku Komura

- retweets: 36, favorites: 34 (01/14/2022 11:10:53)

- links: [abs](https://arxiv.org/abs/2201.04439) | [pdf](https://arxiv.org/pdf/2201.04439)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Controlling the manner in which a character moves in a real-time animation system is a challenging task with useful applications. Existing style transfer systems require access to a reference content motion clip, however, in real-time systems the future motion content is unknown and liable to change with user input. In this work we present a style modelling system that uses an animation synthesis network to model motion content based on local motion phases. An additional style modulation network uses feature-wise transformations to modulate style in real-time. To evaluate our method, we create and release a new style modelling dataset, 100STYLE, containing over 4 million frames of stylised locomotion data in 100 different styles that present a number of challenges for existing systems. To model these styles, we extend the local phase calculation with a contact-free formulation. In comparison to other methods for real-time style modelling, we show our system is more robust and efficient in its style representation while improving motion quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Real-Time Style Modelling of Locomotion via Feature-Wise Transformations and Local Motion Phases<br>abs: <a href="https://t.co/VGZ9wiNojv">https://t.co/VGZ9wiNojv</a> <a href="https://t.co/XeCd7QhqQd">pic.twitter.com/XeCd7QhqQd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1481441541784805380?ref_src=twsrc%5Etfw">January 13, 2022</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



