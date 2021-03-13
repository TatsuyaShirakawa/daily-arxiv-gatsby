---
title: Hot Papers 2021-03-12
date: 2021-03-13T14:47:01.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-12"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-12"
socialImage: "/media/flying-marine.jpg"

---

# 1. CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language  Representation

Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting

- retweets: 710, favorites: 293 (03/13/2021 14:47:01)

- links: [abs](https://arxiv.org/abs/2103.06874) | [pdf](https://arxiv.org/pdf/2103.06874)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly-used models still require an explicit tokenization step. While recent tokenization approaches based on data-derived subword lexicons are less brittle than manually engineered tokenizers, these techniques are not equally suited to all languages, and the use of any fixed vocabulary may limit a model's ability to adapt. In this paper, we present CANINE, a neural encoder that operates directly on character sequences--without explicit tokenization or vocabulary--and a pre-training strategy with soft inductive biases in place of hard token boundaries.To use its finer-grained input effectively and efficiently, CANINE combines downsampling, which reduces the input sequence length, with a deep transformer stack, which encodes con-text. CANINE outperforms a comparable mBERT model by >=1 F1 on TyDi QA, a challenging multilingual benchmark, despite having 28% fewer model parameters.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">単語の分かち書きをなしに文字レベルで、BERT超えの手法が発表されました<br><br>CANINE (Character Architecture with No tokenization In Neural Encoders) <br><br>(論文)<a href="https://t.co/greWNpcPsN">https://t.co/greWNpcPsN</a><br><br>日本語が本当にどこまでいけるか疑問ですが、以下で非常に丁寧に解説されています<a href="https://t.co/GcVOQMAiVU">https://t.co/GcVOQMAiVU</a></p>&mdash; 小川雄太郎 (@ISID_AI_team) <a href="https://twitter.com/ISID_AI_team/status/1370273581267963910?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation<br><br>A pre-training strategy w/o hard token boundaries trains a character-level encoder that outperforms mBERT w/ fewer parameters.<a href="https://t.co/1mRWy12E48">https://t.co/1mRWy12E48</a> <a href="https://t.co/XWWlD2X80q">pic.twitter.com/XWWlD2X80q</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370192295165759498?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Holistic 3D Scene Understanding from a Single Image with Implicit  Representation

Cheng Zhang, Zhaopeng Cui, Yinda Zhang, Bing Zeng, Marc Pollefeys, Shuaicheng Liu

- retweets: 626, favorites: 199 (03/13/2021 14:47:01)

- links: [abs](https://arxiv.org/abs/2103.06422) | [pdf](https://arxiv.org/pdf/2103.06422)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present a new pipeline for holistic 3D scene understanding from a single image, which could predict object shape, object pose, and scene layout. As it is a highly ill-posed problem, existing methods usually suffer from inaccurate estimation of both shapes and layout especially for the cluttered scene due to the heavy occlusion between objects. We propose to utilize the latest deep implicit representation to solve this challenge. We not only propose an image-based local structured implicit network to improve the object shape estimation, but also refine 3D object pose and scene layout via a novel implicit scene graph neural network that exploits the implicit local object features. A novel physical violation loss is also proposed to avoid incorrect context between objects. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods in terms of object shape, scene layout estimation, and 3D object detection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Holistic 3D Scene Understanding from a Single Image with Implicit Representation<br>pdf: <a href="https://t.co/XDcIxs9Ggi">https://t.co/XDcIxs9Ggi</a><br>abs: <a href="https://t.co/Hnws5q7WL9">https://t.co/Hnws5q7WL9</a> <a href="https://t.co/Ht3IenDYOB">pic.twitter.com/Ht3IenDYOB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370209523449925633?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Multi-Format Contrastive Learning of Audio Representations

Luyu Wang, Aaron van den Oord

- retweets: 584, favorites: 138 (03/13/2021 14:47:02)

- links: [abs](https://arxiv.org/abs/2103.06508) | [pdf](https://arxiv.org/pdf/2103.06508)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Recent advances suggest the advantage of multi-modal training in comparison with single-modal methods. In contrast to this view, in our work we find that similar gain can be obtained from training with different formats of a single modality. In particular, we investigate the use of the contrastive learning framework to learn audio representations by maximizing the agreement between the raw audio and its spectral representation. We find a significant gain using this multi-format strategy against the single-format counterparts. Moreover, on the downstream AudioSet and ESC-50 classification task, our audio-only approach achieves new state-of-the-art results with a mean average precision of 0.376 and an accuracy of 90.5%, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Format Contrastive Learning of Audio<br>Representations<br><br>Using the contrastive learning to maximize the agreement between the raw audio and its spectral representation leads to  a significant gain and achieves a new SotA. <a href="https://t.co/R6JcpLaCQV">https://t.co/R6JcpLaCQV</a> <a href="https://t.co/ZgQfInnyEK">pic.twitter.com/ZgQfInnyEK</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370193384208355329?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Multi-Format Contrastive Learning of Audio Representations<br>pdf: <a href="https://t.co/l2YDNCPOJM">https://t.co/l2YDNCPOJM</a><br>abs: <a href="https://t.co/w8i5u3nFlr">https://t.co/w8i5u3nFlr</a> <a href="https://t.co/VU3znf2QJm">pic.twitter.com/VU3znf2QJm</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370190871052816384?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement  Learning

Samarth Sinha, Animesh Garg

- retweets: 565, favorites: 144 (03/13/2021 14:47:02)

- links: [abs](https://arxiv.org/abs/2103.06326) | [pdf](https://arxiv.org/pdf/2103.06326)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Offline reinforcement learning proposes to learn policies from large collected datasets without interaction. These algorithms have made it possible to learn useful skills from data that can then be transferred to the environment, making it feasible to deploy the trained policies in real-world settings where interactions may be costly or dangerous, such as self-driving. However, current algorithms overfit to the dataset they are trained on and perform poor out-of-distribution (OOD) generalization to the environment when deployed. We propose a Surprisingly Simple Self-Supervision algorithm (S4RL), which utilizes data augmentations from states to learn value functions that are better at generalizing and extrapolating when deployed in the environment. We investigate different data augmentation techniques that help learning a value function that can extrapolate to OOD data, and how to combine data augmentations and offline RL algorithms to learn a policy. We experimentally show that using S4RL significantly improves the state-of-the-art on most benchmark offline reinforcement learning tasks on popular benchmark datasets from D4RL, despite being simple and easy to implement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning<br><br>Significantly improves the SotA on various offline RL tasks with a better data augmentation strategy.<a href="https://t.co/eEmbOYs41r">https://t.co/eEmbOYs41r</a> <a href="https://t.co/q4ewkW6tF2">pic.twitter.com/q4ewkW6tF2</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370197767428087810?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our recent work:<br><br>Surprisingly Simple Self-Supervision for Offline RL where we propose a Surprisingly Simple method to learn representations using data augmentations from offline data which achieves SOTA performance! <a href="https://t.co/jPX8tt8pc6">https://t.co/jPX8tt8pc6</a><br><br>w/ <a href="https://twitter.com/animesh_garg?ref_src=twsrc%5Etfw">@animesh_garg</a> <a href="https://t.co/CTwA31uLMq">pic.twitter.com/CTwA31uLMq</a></p>&mdash; Samarth Sinha (@_sam_sinha_) <a href="https://twitter.com/_sam_sinha_/status/1370492784319344648?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. SMPLicit: Topology-aware Generative Model for Clothed People

Enric Corona, Albert Pumarola, Guillem Alenyà, Gerard Pons-Moll, Francesc Moreno-Noguer

- retweets: 403, favorites: 236 (03/13/2021 14:47:02)

- links: [abs](https://arxiv.org/abs/2103.06871) | [pdf](https://arxiv.org/pdf/2103.06871)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper we introduce SMPLicit, a novel generative model to jointly represent body pose, shape and clothing geometry. In contrast to existing learning-based approaches that require training specific models for each type of garment, SMPLicit can represent in a unified manner different garment topologies (e.g. from sleeveless tops to hoodies and to open jackets), while controlling other properties like the garment size or tightness/looseness. We show our model to be applicable to a large variety of garments including T-shirts, hoodies, jackets, shorts, pants, skirts, shoes and even hair. The representation flexibility of SMPLicit builds upon an implicit model conditioned with the SMPL human body parameters and a learnable latent space which is semantically interpretable and aligned with the clothing attributes. The proposed model is fully differentiable, allowing for its use into larger end-to-end trainable systems. In the experimental section, we demonstrate SMPLicit can be readily used for fitting 3D scans and for 3D reconstruction in images of dressed people. In both cases we are able to go beyond state of the art, by retrieving complex garment geometries, handling situations with multiple clothing layers and providing a tool for easy outfit editing. To stimulate further research in this direction, we will make our code and model publicly available at http://www.iri.upc.edu/people/ecorona/smplicit/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SMPLicit: Topology-aware Generative Model for Clothed People<br>pdf: <a href="https://t.co/4x1cxKQFnk">https://t.co/4x1cxKQFnk</a><br>abs: <a href="https://t.co/1ksZEdppsb">https://t.co/1ksZEdppsb</a><br>project page: <a href="https://t.co/A6lsQMXEvs">https://t.co/A6lsQMXEvs</a> <a href="https://t.co/QPZM2USAOI">pic.twitter.com/QPZM2USAOI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370198244119183368?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📢📢Check out our new work:<br>SMPLicit: Topology-aware Generative Model for Clothed People!<br>Accepted at <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> with <a href="https://twitter.com/AlbertPumarola?ref_src=twsrc%5Etfw">@AlbertPumarola</a> <a href="https://twitter.com/_guillem_?ref_src=twsrc%5Etfw">@_guillem_</a> <a href="https://twitter.com/GerardPonsMoll1?ref_src=twsrc%5Etfw">@GerardPonsMoll1</a> <a href="https://twitter.com/fmorenoguer?ref_src=twsrc%5Etfw">@fmorenoguer</a> <br><br>Arxiv: <a href="https://t.co/CnGiBrU0LF">https://t.co/CnGiBrU0LF</a><br>project: <a href="https://t.co/ll7NyOIpw0">https://t.co/ll7NyOIpw0</a><br>Code will be available soon, stay tuned! <a href="https://t.co/xXJLzXVAx4">pic.twitter.com/xXJLzXVAx4</a></p>&mdash; Enric Corona (@enric_corona) <a href="https://twitter.com/enric_corona/status/1370283671547166722?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have just released SMPLicit :), a fully differentiable generative model to jointly represent body pose, shape, and clothing geometry. <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a> <br><br>🖥Project: <a href="https://t.co/64eJFJg3Nr">https://t.co/64eJFJg3Nr</a><br>📄PDF: <a href="https://t.co/REhmGNj2ZD">https://t.co/REhmGNj2ZD</a> <a href="https://t.co/N6rzMwvmOt">https://t.co/N6rzMwvmOt</a></p>&mdash; Albert Pumarola (@AlbertPumarola) <a href="https://twitter.com/AlbertPumarola/status/1370344895781732353?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SMPLicit:  a differentiable body model with clothing of varied topology (neural implicit functions), we fit it to images too. <a href="https://twitter.com/hashtag/CVPR2021?src=hash&amp;ref_src=twsrc%5Etfw">#CVPR2021</a>. Check it out:<br><br>paper: <a href="https://t.co/vhXsEuluPq">https://t.co/vhXsEuluPq</a><br>project: <a href="https://t.co/03FQgO0PN9">https://t.co/03FQgO0PN9</a><br><br>with <a href="https://twitter.com/enric_corona?ref_src=twsrc%5Etfw">@enric_corona</a> <a href="https://twitter.com/AlbertPumarola?ref_src=twsrc%5Etfw">@AlbertPumarola</a> <a href="https://twitter.com/_guillem_?ref_src=twsrc%5Etfw">@_guillem_</a> <a href="https://twitter.com/fmorenoguer?ref_src=twsrc%5Etfw">@fmorenoguer</a> <a href="https://t.co/hfVhNWQs3K">https://t.co/hfVhNWQs3K</a></p>&mdash; Gerard Pons-Moll (@GerardPonsMoll1) <a href="https://twitter.com/GerardPonsMoll1/status/1370420589593255936?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. BYOL for Audio: Self-Supervised Learning for General-Purpose Audio  Representation

Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, Kunio Kashino

- retweets: 260, favorites: 96 (03/13/2021 14:47:03)

- links: [abs](https://arxiv.org/abs/2103.06695) | [pdf](https://arxiv.org/pdf/2103.06695)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Inspired by the recent progress in self-supervised learning for computer vision that generates supervision using data augmentations, we explore a new general-purpose audio representation learning approach. We propose learning general-purpose audio representation from a single audio segment without expecting relationships between different time segments of audio samples. To implement this principle, we introduce Bootstrap Your Own Latent (BYOL) for Audio (BYOL-A, pronounced "viola"), an audio self-supervised learning method based on BYOL for learning general-purpose audio representation. Unlike most previous audio self-supervised learning methods that rely on agreement of vicinity audio segments or disagreement of remote ones, BYOL-A creates contrasts in an augmented audio segment pair derived from a single audio segment. With a combination of normalization and augmentation techniques, BYOL-A achieves state-of-the-art results in various downstream tasks. Extensive ablation studies also clarified the contribution of each component and their combinations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation<br>pdf: <a href="https://t.co/zrT9CanBQc">https://t.co/zrT9CanBQc</a><br>abs: <a href="https://t.co/fjRbMEd36g">https://t.co/fjRbMEd36g</a> <a href="https://t.co/nl2ISCTZao">pic.twitter.com/nl2ISCTZao</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370198900208963586?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our new paper is out!🚀 First professional research paper in my 20+yrs career. 😭<br>This is for training better audio encoder, especially for a practical small audio DNN.<br>&quot;BYOL for Audio: Self-Supervised Learning for General-Purpose Audio Representation&quot;<a href="https://t.co/DwnNOpWsEX">https://t.co/DwnNOpWsEX</a></p>&mdash; daisukelab (@nizumical) <a href="https://twitter.com/nizumical/status/1370363959157563398?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Hurdles to Progress in Long-form Question Answering

Kalpesh Krishna, Aurko Roy, Mohit Iyyer

- retweets: 132, favorites: 66 (03/13/2021 14:47:03)

- links: [abs](https://arxiv.org/abs/2103.06332) | [pdf](https://arxiv.org/pdf/2103.06332)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The task of long-form question answering (LFQA) involves retrieving documents relevant to a given question and using them to generate a paragraph-length answer. While many models have recently been proposed for LFQA, we show in this paper that the task formulation raises fundamental challenges regarding evaluation and dataset creation that currently preclude meaningful modeling progress. To demonstrate these challenges, we first design a new system that relies on sparse attention and contrastive retriever learning to achieve state-of-the-art performance on the ELI5 LFQA dataset. While our system tops the public leaderboard, a detailed analysis reveals several troubling trends: (1) our system's generated answers are not actually grounded in the documents that it retrieves; (2) ELI5 contains significant train / test overlap, as at least 81% of ELI5 validation questions occur in paraphrased form in the training set; (3) ROUGE-L is not an informative metric of generated answer quality and can be easily gamed; and (4) human evaluations used for other text generation tasks are unreliable for LFQA. We provide suggestions to mitigate each of these issues, which we hope will lead to more rigorous LFQA research and meaningful progress in the future.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hurdles to Progress in Long-form Question Answering<br><br>- ELI5 contains significant train / test overlap<br><br>- ROUGE-L is uninformative for this task<br><br>- human evaluations used for other text gen tasks<br>are unreliable for long-form QA<a href="https://t.co/hNDUTd4YdJ">https://t.co/hNDUTd4YdJ</a> <a href="https://t.co/gSvdr24qow">pic.twitter.com/gSvdr24qow</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370195017352970241?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Fast and Accurate Model Scaling

Piotr Dollár, Mannat Singh, Ross Girshick

- retweets: 100, favorites: 79 (03/13/2021 14:47:03)

- links: [abs](https://arxiv.org/abs/2103.06877) | [pdf](https://arxiv.org/pdf/2103.06877)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this work we analyze strategies for convolutional neural network scaling; that is, the process of scaling a base convolutional network to endow it with greater computational complexity and consequently representational power. Example scaling strategies may include increasing model width, depth, resolution, etc. While various scaling strategies exist, their tradeoffs are not fully understood. Existing analysis typically focuses on the interplay of accuracy and flops (floating point operations). Yet, as we demonstrate, various scaling strategies affect model parameters, activations, and consequently actual runtime quite differently. In our experiments we show the surprising result that numerous scaling strategies yield networks with similar accuracy but with widely varying properties. This leads us to propose a simple fast compound scaling strategy that encourages primarily scaling model width, while scaling depth and resolution to a lesser extent. Unlike currently popular scaling strategies, which result in about $O(s)$ increase in model activation w.r.t. scaling flops by a factor of $s$, the proposed fast compound scaling results in close to $O(\sqrt{s})$ increase in activations, while achieving excellent accuracy. This leads to comparable speedups on modern memory-limited hardware (e.g., GPU, TPU). More generally, we hope this work provides a framework for analyzing and selecting scaling strategies under various computational constraints.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Fast and Accurate Model Scaling<br><br>Investigates various scaling strategies and finds that scaling up the width by O(sqrt(s)) works the best in terms of performance-computes trade-off with GPUs.<a href="https://t.co/7AjlABlspY">https://t.co/7AjlABlspY</a> <a href="https://t.co/XJJcq0BXNJ">pic.twitter.com/XJJcq0BXNJ</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370189948922462208?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. WenLan: Bridging Vision and Language by Large-Scale Multi-Modal  Pre-Training

Yuqi Huo, Manli Zhang, Guangzhen Liu, Haoyu Lu, Yizhao Gao, Guoxing Yang, Jingyuan Wen, Heng Zhang, Baogui Xu, Weihao Zheng, Zongzheng Xi, Yueqian Yang, Anwen Hu, Jinming Zhao, Ruichen Li, Yida Zhao, Liang Zhang, Yuqing Song, Xin Hong, Wanqing Cui, Danyang Hou, Yingyan Li, Junyi Li, Peiyu Liu, Zheng Gong, Chuhao Jin, Yuchong Sun, Shizhe Chen, Zhiwu Lu, Zhicheng Dou, Qin Jin, Yanyan Lan, Wayne Xin Zhao, Ruihua Song, Ji-Rong Wen

- retweets: 72, favorites: 72 (03/13/2021 14:47:03)

- links: [abs](https://arxiv.org/abs/2103.06561) | [pdf](https://arxiv.org/pdf/2103.06561)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent)

Multi-modal pre-training models have been intensively explored to bridge vision and language in recent years. However, most of them explicitly model the cross-modal interaction between image-text pairs, by assuming that there exists strong semantic correlation between the text and image modalities. Since this strong assumption is often invalid in real-world scenarios, we choose to implicitly model the cross-modal correlation for large-scale multi-modal pre-training, which is the focus of the Chinese project `WenLan' led by our team. Specifically, with the weak correlation assumption over image-text pairs, we propose a two-tower pre-training model within the cross-modal contrastive learning (CMCL) framework. Unlike OpenAI CLIP that adopts a simple contrastive learning method, we devise a more advanced algorithm by adapting the latest method MoCo into the cross-modal scenario. By building a large queue-based dictionary, our CMCL can incorporate more negative samples in limited GPU resources. We further construct a large Chinese multi-source image-text dataset called RUC-CAS-WenLan for pre-training our CMCL model. Extensive experiments demonstrate that the pre-trained CMCL model outperforms both UNITER and OpenAI CLIP on various downstream tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training<br>pdf: <a href="https://t.co/LPcVgF6lFO">https://t.co/LPcVgF6lFO</a><br>abs: <a href="https://t.co/UfzAjDZyYh">https://t.co/UfzAjDZyYh</a> <a href="https://t.co/MQz6qZXMaT">pic.twitter.com/MQz6qZXMaT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370189803816431619?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training<br><br>Constructs a large Chinese multi-source image-text dataset for pre-training, which leads to their model outperforming both UNITER and CLIP on various downstream tasks.<a href="https://t.co/NWVA7aqt17">https://t.co/NWVA7aqt17</a> <a href="https://t.co/foX43PaUn1">pic.twitter.com/foX43PaUn1</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1370190585693229058?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. MERMAID: Metaphor Generation with Symbolism and Discriminative Decoding

Tuhin Chakrabarty, Xurui Zhang, Smaranda Muresan, Nanyun Peng

- retweets: 56, favorites: 61 (03/13/2021 14:47:03)

- links: [abs](https://arxiv.org/abs/2103.06779) | [pdf](https://arxiv.org/pdf/2103.06779)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Generating metaphors is a challenging task as it requires a proper understanding of abstract concepts, making connections between unrelated concepts, and deviating from the literal meaning. Based on a theoretically-grounded connection between metaphors and symbols, we propose a method to automatically construct a parallel corpus by transforming a large number of metaphorical sentences from the Gutenberg Poetry corpus (Jacobs, 2018) to their literal counterpart using recent advances in masked language modeling coupled with commonsense inference. For the generation task, we incorporate a metaphor discriminator to guide the decoding of a sequence to sequence model fine-tuned on our parallel data to generate high-quality metaphors. Human evaluation on an independent test set of literal statements shows that our best model generates metaphors better than three well-crafted baselines 66% of the time on average. A task-based evaluation shows that human-written poems enhanced with metaphors proposed by our model are preferred 68% of the time compared to poems without metaphors.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🔥Metaphors are not to be trifled with🔥 Excited to share <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> preprint titled “MERMAID: Metaphor Generation with Symbolism and Discriminative Decoding”<a href="https://t.co/gcnvn995vR">https://t.co/gcnvn995vR</a> . Joint work with my figurative NLG constants <a href="https://twitter.com/VioletNPeng?ref_src=twsrc%5Etfw">@VioletNPeng</a> and Smaranda Muresan. <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://t.co/3mJqlLV6j2">pic.twitter.com/3mJqlLV6j2</a></p>&mdash; Tuhin Chakrabarty (@TuhinChakr) <a href="https://twitter.com/TuhinChakr/status/1370408736343343106?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Fair Mixup: Fairness via Interpolation

Ching-Yao Chuang, Youssef Mroueh

- retweets: 78, favorites: 27 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06503) | [pdf](https://arxiv.org/pdf/2103.06503)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Training classifiers under fairness constraints such as group fairness, regularizes the disparities of predictions between the groups. Nevertheless, even though the constraints are satisfied during training, they might not generalize at evaluation time. To improve the generalizability of fair classifiers, we propose fair mixup, a new data augmentation strategy for imposing the fairness constraint. In particular, we show that fairness can be achieved by regularizing the models on paths of interpolated samples between the groups. We use mixup, a powerful data augmentation strategy to generate these interpolates. We analyze fair mixup and empirically show that it ensures a better generalization for both accuracy and fairness measurement in tabular, vision, and language benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">&quot;Fair Mixup: Fairness via Interpolation&quot;<br><br>Happy to share that the work done during my internship at IBM Research AI was accepted at ICLR 2021! <br><br>Paper: <a href="https://t.co/uN28ntnwfV">https://t.co/uN28ntnwfV</a><br>Code: <a href="https://t.co/vaQigPhooR">https://t.co/vaQigPhooR</a><br><br>with Youssef Mroueh <a href="https://t.co/Ucc6Zsy7T2">pic.twitter.com/Ucc6Zsy7T2</a></p>&mdash; Ching-Yao Chuang (@ChingYaoChuang) <a href="https://twitter.com/ChingYaoChuang/status/1370202717478473728?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. 3D Head-Position Prediction in First-Person View by Considering Head  Pose for Human-Robot Eye Contact

Yuki Tamaru, Yasunori Ozaki, Yuki Okafuji, Jun Baba, Junya Nakanishi, Yuichiro Yoshikawa

- retweets: 53, favorites: 33 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06417) | [pdf](https://arxiv.org/pdf/2103.06417)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

For a humanoid robot to make eye contact to initiate communication with a human, it is necessary to estimate the human's head position.However, eye contact becomes difficult due to the mechanical delay of the robot while the subject with whom the robot is interacting with is moving. Owing to these issues, it is important to perform head-position prediction to mitigate the effect of the delay in the robot's motion. Based on the fact that humans turn their heads before changing direction while walking, we hypothesized that the accuracy of three-dimensional(3D) head-position prediction from the first-person view can be improved by considering the head pose into account.We compared our method with the conventional Kalman filter-based method, and found our method to be more accurate. The experimental results show that considering the head pose helps improve the accuracy of 3D head-position prediction.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">インターンシップで一緒に研究した東大の田丸さんとの研究をarXivで公開しました。研究内容は、人間とロボットとでアイコンタクトするための一人称映像からの三次元顔位置予測です。ロボットの国際会議IROSに投稿しているので、運が良ければ採択されるといいですね。<a href="https://t.co/Q2JGwQmSqH">https://t.co/Q2JGwQmSqH</a> <a href="https://t.co/6jkLVQfOym">pic.twitter.com/6jkLVQfOym</a></p>&mdash; あるふ (@alfredplpl) <a href="https://twitter.com/alfredplpl/status/1370263525172596738?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. A semi-agnostic ansatz with variable structure for quantum machine  learning

M. Bilkis, M. Cerezo, Guillaume Verdon, Patrick J. Coles, Lukasz Cincio

- retweets: 32, favorites: 52 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06712) | [pdf](https://arxiv.org/pdf/2103.06712)
- [quant-ph](https://arxiv.org/list/quant-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Quantum machine learning (QML) offers a powerful, flexible paradigm for programming near-term quantum computers, with applications in chemistry, metrology, materials science, data science, and mathematics. Here, one trains an ansatz, in the form of a parameterized quantum circuit, to accomplish a task of interest. However, challenges have recently emerged suggesting that deep ansatzes are difficult to train, due to flat training landscapes caused by randomness or by hardware noise. This motivates our work, where we present a variable structure approach to build ansatzes for QML. Our approach, called VAns (Variable Ansatz), applies a set of rules to both grow and (crucially) remove quantum gates in an informed manner during the optimization. Consequently, VAns is ideally suited to mitigate trainability and noise-related issues by keeping the ansatz shallow. We employ VAns in the variational quantum eigensolver for condensed matter and quantum chemistry applications and also in the quantum autoencoder for data compression, showing successful results in all cases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to debut on Quantum-Twitter-World by announcing<br><br>A semi-agnostic ansatz with variable structure for quantum machine learning 🔥<br><br>Exciting collaboration with <a href="https://twitter.com/MvsCerezo?ref_src=twsrc%5Etfw">@MvsCerezo</a> <a href="https://twitter.com/quantumVerd?ref_src=twsrc%5Etfw">@quantumVerd</a>  <a href="https://twitter.com/ColesQuantum?ref_src=twsrc%5Etfw">@ColesQuantum</a> <a href="https://twitter.com/LCincio?ref_src=twsrc%5Etfw">@LCincio</a> and unnoficially sponsored by <a href="https://twitter.com/VANS_66?ref_src=twsrc%5Etfw">@vans_66</a>  👟👟<a href="https://t.co/vxXueHLEjB">https://t.co/vxXueHLEjB</a></p>&mdash; Mati Bilkis (@MatiasBilkis) <a href="https://twitter.com/MatiasBilkis/status/1370393849961533440?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Robust High-speed Running for Quadruped Robots via Deep Reinforcement  Learning

Guillaume Bellegarda, Quan Nguyen

- retweets: 36, favorites: 31 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06484) | [pdf](https://arxiv.org/pdf/2103.06484)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

Deep reinforcement learning has emerged as a popular and powerful way to develop locomotion controllers for quadruped robots. Common approaches have largely focused on learning actions directly in joint space, or learning to modify and offset foot positions produced by trajectory generators. Both approaches typically require careful reward shaping and training for millions of time steps, and with trajectory generators introduce human bias into the resulting control policies. In this paper, we instead explore learning foot positions in Cartesian space, which we track with impedance control, for a task of running as fast as possible subject to environmental disturbances. Compared with other action spaces, we observe less needed reward shaping, much improved sample efficiency, the emergence of natural gaits such as galloping and bounding, and ease of sim-to-sim transfer. Policies can be learned in only a few million time steps, even for challenging tasks of running over rough terrain with loads of over 100% of the nominal quadruped mass. Training occurs in PyBullet, and we perform a sim-to-sim transfer to Gazebo, where our quadruped is able to run at over 4 m/s without a load, and 3.5 m/s with a 10 kg load, which is over 83% of the nominal quadruped mass. Video results can be found at https://youtu.be/roE1vxpEWfw.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Robust High-speed Running for Quadruped Robots via Deep Reinforcement Learning<br>pdf: <a href="https://t.co/0r5Gcerjmw">https://t.co/0r5Gcerjmw</a><br>abs: <a href="https://t.co/9Y0GBAHNBp">https://t.co/9Y0GBAHNBp</a> <a href="https://t.co/6f7X1XG07k">pic.twitter.com/6f7X1XG07k</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370202496887574536?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Hard Attention Control By Mutual Information Maximization

Himanshu Sahni, Charles Isbell

- retweets: 42, favorites: 20 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06371) | [pdf](https://arxiv.org/pdf/2103.06371)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

Biological agents have adopted the principle of attention to limit the rate of incoming information from the environment. One question that arises is if an artificial agent has access to only a limited view of its surroundings, how can it control its attention to effectively solve tasks? We propose an approach for learning how to control a hard attention window by maximizing the mutual information between the environment state and the attention location at each step. The agent employs an internal world model to make predictions about its state and focuses attention towards where the predictions may be wrong. Attention is trained jointly with a dynamic memory architecture that stores partial observations and keeps track of the unobserved state. We demonstrate that our approach is effective in predicting the full state from a sequence of partial observations. We also show that the agent's internal representation of the surroundings, a live mental map, can be used for control in two partially observable reinforcement learning tasks. Videos of the trained agent can be found at https://sites.google.com/view/hard-attention-control.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hard Attention Control By Mutual Information Maximization<br>pdf: <a href="https://t.co/Uiq9yoYz7y">https://t.co/Uiq9yoYz7y</a><br>abs: <a href="https://t.co/vVaEMj59Lp">https://t.co/vVaEMj59Lp</a><br>project page: <a href="https://t.co/4vybnkLlK6">https://t.co/4vybnkLlK6</a> <a href="https://t.co/4Vx20T8cT5">pic.twitter.com/4Vx20T8cT5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1370201307244597249?ref_src=twsrc%5Etfw">March 12, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Research Software Sustainability and Citation

Stephan Druskat, Daniel S. Katz, Ilian T. Todorov

- retweets: 42, favorites: 15 (03/13/2021 14:47:04)

- links: [abs](https://arxiv.org/abs/2103.06681) | [pdf](https://arxiv.org/pdf/2103.06681)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Software citation contributes to achieving software sustainability in two ways: It provides an impact metric to incentivize stakeholders to make software sustainable. It also provides references to software used in research, which can be reused and adapted to become sustainable. While software citation faces a host of technical and social challenges, community initiatives have defined the principles of software citation and are working on implementing solutions.



