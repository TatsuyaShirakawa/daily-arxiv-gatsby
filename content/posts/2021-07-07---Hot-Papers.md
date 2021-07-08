---
title: Hot Papers 2021-07-07
date: 2021-07-08T10:03:54.Z
template: "post"
draft: false
slug: "hot-papers-2021-07-07"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-07-07"
socialImage: "/media/flying-marine.jpg"

---

# 1. Growing Urban Bicycle Networks

Michael Szell, Sayat Mimar, Tyler Perlman, Gourab Ghoshal, Roberta Sinatra

- retweets: 10612, favorites: 11 (07/08/2021 10:03:54)

- links: [abs](https://arxiv.org/abs/2107.02185) | [pdf](https://arxiv.org/pdf/2107.02185)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Cycling is a promising solution to unsustainable car-centric urban transport systems. However, prevailing bicycle network development follows a slow and piecewise process, without taking into account the structural complexity of transportation networks. Here we explore systematically the topological limitations of urban bicycle network development. For 62 cities we study different variations of growing a synthetic bicycle network between an arbitrary set of points routed on the urban street network. We find initially decreasing returns on investment until a critical threshold, posing fundamental consequences to sustainable urban planning: Cities must invest into bicycle networks with the right growth strategy, and persistently, to surpass a critical mass. We also find pronounced overlaps of synthetically grown networks in cities with well-developed existing bicycle networks, showing that our model reflects reality. Growing networks from scratch makes our approach a generally applicable starting point for sustainable urban bicycle network planning with minimal data requirements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🚨New Preprint! Growing Urban Bicycle Networks<a href="https://t.co/GYXk4Camap">https://t.co/GYXk4Camap</a><br>Explore at <a href="https://t.co/xOIplgTNMm">https://t.co/xOIplgTNMm</a><br><br>We study the limitations of growing 🚲🕸️ <br>Main finding: Cities must invest 1) with right growth strategy and 2) *persistently*, to overcome a critical mass. 🧵 <a href="https://t.co/JJkrPhsZlU">pic.twitter.com/JJkrPhsZlU</a></p>&mdash; Michael Szell (@mszll) <a href="https://twitter.com/mszll/status/1412662575435636736?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Long-Short Transformer: Efficient Transformers for Language and Vision

Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, Bryan Catanzaro

- retweets: 2402, favorites: 275 (07/08/2021 10:03:54)

- links: [abs](https://arxiv.org/abs/2107.02192) | [pdf](https://arxiv.org/pdf/2107.02192)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Transformers have achieved success in both language and vision domains. However, it is prohibitively expensive to scale them to long sequences such as long documents or high-resolution images, because self-attention mechanism has quadratic time and memory complexities with respect to the input sequence length. In this paper, we propose Long-Short Transformer (Transformer-LS), an efficient self-attention mechanism for modeling long sequences with linear complexity for both language and vision tasks. It aggregates a novel long-range attention with dynamic projection to model distant correlations and a short-term attention to capture fine-grained local correlations. We propose a dual normalization strategy to account for the scale mismatch between the two attention mechanisms. Transformer-LS can be applied to both autoregressive and bidirectional models without additional complexity. Our method outperforms the state-of-the-art models on multiple tasks in language and vision domains, including the Long Range Arena benchmark, autoregressive language modeling, and ImageNet classification. For instance, Transformer-LS achieves 0.97 test BPC on enwik8 using half the number of parameters than previous method, while being faster and is able to handle 3$\times$ as long sequences compared to its full-attention version on the same hardware. On ImageNet, it can obtain the state-of-the-art results~(e.g., Top-1 accuracy 84.1% trained on 224$\times$224 ImageNet-1K only), while being more scalable on high-resolution images. The models and source code will be released soon.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Long-Short Transformer: Efficient Transformers for Language and Vision<br>pdf: <a href="https://t.co/NBRjNcTdGa">https://t.co/NBRjNcTdGa</a><br>abs: <a href="https://t.co/V8qKUkVH1c">https://t.co/V8qKUkVH1c</a><br><br>On ImageNet, sota results (e.g., Top-1 accuracy 84.1% trained on 224 × 224 ImageNet-1K only), while being more scalable on high-resolution images <a href="https://t.co/F3ijZL5WjM">pic.twitter.com/F3ijZL5WjM</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412574448973135875?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. TransformerFusion: Monocular RGB Scene Reconstruction using Transformers

Aljaž Božič, Pablo Palafox, Justus Thies, Angela Dai, Matthias Nießner

- retweets: 1798, favorites: 285 (07/08/2021 10:03:55)

- links: [abs](https://arxiv.org/abs/2107.02191) | [pdf](https://arxiv.org/pdf/2107.02191)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce TransformerFusion, a transformer-based 3D scene reconstruction approach. From an input monocular RGB video, the video frames are processed by a transformer network that fuses the observations into a volumetric feature grid representing the scene; this feature grid is then decoded into an implicit 3D scene representation. Key to our approach is the transformer architecture that enables the network to learn to attend to the most relevant image frames for each 3D location in the scene, supervised only by the scene reconstruction task. Features are fused in a coarse-to-fine fashion, storing fine-level features only where needed, requiring lower memory storage and enabling fusion at interactive rates. The feature grid is then decoded to a higher-resolution scene reconstruction, using an MLP-based surface occupancy prediction from interpolated coarse-to-fine 3D features. Our approach results in an accurate surface reconstruction, outperforming state-of-the-art multi-view stereo depth estimation methods, fully-convolutional 3D reconstruction approaches, and approaches using LSTM- or GRU-based recurrent networks for video sequence fusion.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out <a href="https://twitter.com/BozicAljaz?ref_src=twsrc%5Etfw">@BozicAljaz</a> work TransformerFusion, an online monocular RGB scene reconstruction using a transformer to learn to attend to the most relevant pixel observations.<br><br>Project page: <a href="https://t.co/4VJjmJ5sP7">https://t.co/4VJjmJ5sP7</a><br>Paper: <a href="https://t.co/bZVnSnYQgo">https://t.co/bZVnSnYQgo</a><br>Video: <a href="https://t.co/Qma0T7fQYI">https://t.co/Qma0T7fQYI</a> <a href="https://t.co/h2ZiOw5ySI">pic.twitter.com/h2ZiOw5ySI</a></p>&mdash; Matthias Niessner (@MattNiessner) <a href="https://twitter.com/MattNiessner/status/1412796072028774400?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TransformerFusion: Monocular RGB Scene Reconstruction using Transformers<br>pdf: <a href="https://t.co/N5Uj3gl6sK">https://t.co/N5Uj3gl6sK</a><br>abs: <a href="https://t.co/QV5rYbuGYM">https://t.co/QV5rYbuGYM</a><br>project page: <a href="https://t.co/IoJKwt0pUg">https://t.co/IoJKwt0pUg</a> <a href="https://t.co/sjSSVyRKcr">pic.twitter.com/sjSSVyRKcr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412577927133380612?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Depth-supervised NeRF: Fewer Views and Faster Training for Free

Kangle Deng, Andrew Liu, Jun-Yan Zhu, Deva Ramanan

- retweets: 1764, favorites: 253 (07/08/2021 10:03:55)

- links: [abs](https://arxiv.org/abs/2107.02791) | [pdf](https://arxiv.org/pdf/2107.02791)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

One common failure mode of Neural Radiance Field (NeRF) models is fitting incorrect geometries when given an insufficient number of input views. We propose DS-NeRF (Depth-supervised Neural Radiance Fields), a loss for learning neural radiance fields that takes advantage of readily-available depth supervision. Our key insight is that sparse depth supervision can be used to regularize the learned geometry, a crucial component for effectively rendering novel views using NeRF. We exploit the fact that current NeRF pipelines require images with known camera poses that are typically estimated by running structure-from-motion (SFM). Crucially, SFM also produces sparse 3D points that can be used as ``free" depth supervision during training: we simply add a loss to ensure that depth rendered along rays that intersect these 3D points is close to the observed depth. We find that DS-NeRF can render more accurate images given fewer training views while training 2-6x faster. With only two training views on real-world images, DS-NeRF significantly outperforms NeRF as well as other sparse-view variants. We show that our loss is compatible with these NeRF models, demonstrating that depth is a cheap and easily digestible supervisory signal. Finally, we show that DS-NeRF supports other types of depth supervision such as scanned depth sensors and RGBD reconstruction outputs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Depth-supervised NeRF: Fewer Views and Faster Training for Free <a href="https://t.co/aGOBDkNDPG">https://t.co/aGOBDkNDPG</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://twitter.com/hashtag/3d?src=hash&amp;ref_src=twsrc%5Etfw">#3d</a><br><br>Project Page: <a href="https://t.co/uimE8H6Zcf">https://t.co/uimE8H6Zcf</a> <a href="https://t.co/Vq20qAXx4V">pic.twitter.com/Vq20qAXx4V</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1412584024585932802?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Depth-supervised NeRF: Fewer Views and Faster Training for Free<br>pdf: <a href="https://t.co/0CCZvyEWHM">https://t.co/0CCZvyEWHM</a><br>abs: <a href="https://t.co/SkhRVt35ye">https://t.co/SkhRVt35ye</a><br>project page: <a href="https://t.co/akD4hzLfpD">https://t.co/akD4hzLfpD</a> <a href="https://t.co/gnuFSTCpZe">pic.twitter.com/gnuFSTCpZe</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412574038984118275?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Rethinking Positional Encoding

Jianqiao Zheng, Sameera Ramasinghe, Simon Lucey

- retweets: 1476, favorites: 213 (07/08/2021 10:03:55)

- links: [abs](https://arxiv.org/abs/2107.02561) | [pdf](https://arxiv.org/pdf/2107.02561)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

It is well noted that coordinate based MLPs benefit greatly -- in terms of preserving high-frequency information -- through the encoding of coordinate positions as an array of Fourier features. Hitherto, the rationale for the effectiveness of these positional encodings has been solely studied through a Fourier lens. In this paper, we strive to broaden this understanding by showing that alternative non-Fourier embedding functions can indeed be used for positional encoding. Moreover, we show that their performance is entirely determined by a trade-off between the stable rank of the embedded matrix and the distance preservation between embedded coordinates. We further establish that the now ubiquitous Fourier feature mapping of position is a special case that fulfills these conditions. Consequently, we present a more general theory to analyze positional encoding in terms of shifted basis functions. To this end, we develop the necessary theoretical formulae and empirically verify that our theoretical claims hold in practice. Codes available at https://github.com/osiriszjq/Rethinking-positional-encoding.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking Positional Encoding<br>pdf: <a href="https://t.co/J6wrROIYLk">https://t.co/J6wrROIYLk</a><br>abs: <a href="https://t.co/eIRZEaAXg7">https://t.co/eIRZEaAXg7</a><br>github: <a href="https://t.co/FHxaSYLt0m">https://t.co/FHxaSYLt0m</a> <a href="https://t.co/I2nBemH2b3">pic.twitter.com/I2nBemH2b3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412602268944326656?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Mind Your Outliers! Investigating the Negative Impact of Outliers on  Active Learning for Visual Question Answering

Siddharth Karamcheti, Ranjay Krishna, Li Fei-Fei, Christopher D. Manning

- retweets: 315, favorites: 120 (07/08/2021 10:03:56)

- links: [abs](https://arxiv.org/abs/2107.02331) | [pdf](https://arxiv.org/pdf/2107.02331)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Active learning promises to alleviate the massive data needs of supervised machine learning: it has successfully improved sample efficiency by an order of magnitude on traditional tasks like topic classification and object recognition. However, we uncover a striking contrast to this promise: across 5 models and 4 datasets on the task of visual question answering, a wide variety of active learning approaches fail to outperform random selection. To understand this discrepancy, we profile 8 active learning methods on a per-example basis, and identify the problem as collective outliers -- groups of examples that active learning methods prefer to acquire but models fail to learn (e.g., questions that ask about text in images or require external knowledge). Through systematic ablation experiments and qualitative visualizations, we verify that collective outliers are a general phenomenon responsible for degrading pool-based active learning. Notably, we show that active learning sample efficiency increases significantly as the number of collective outliers in the active learning pool decreases. We conclude with a discussion and prescriptive recommendations for mitigating the effects of these outliers in future work.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Congrats to <a href="https://twitter.com/siddkaramcheti?ref_src=twsrc%5Etfw">@siddkaramcheti</a>, <a href="https://twitter.com/RanjayKrishna?ref_src=twsrc%5Etfw">@RanjayKrishna</a>, <a href="https://twitter.com/drfeifei?ref_src=twsrc%5Etfw">@drfeifei</a> &amp; <a href="https://twitter.com/chrmanning?ref_src=twsrc%5Etfw">@chrmanning</a> for <a href="https://twitter.com/hashtag/ACL2021NLP?src=hash&amp;ref_src=twsrc%5Etfw">#ACL2021NLP</a> Outstanding Paper Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering. <a href="https://t.co/Y4CI1EXmoH">https://t.co/Y4CI1EXmoH</a> Code <a href="https://t.co/mJXXD5NkNq">https://t.co/mJXXD5NkNq</a> <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://t.co/PJZyeVlWCW">pic.twitter.com/PJZyeVlWCW</a></p>&mdash; Stanford NLP Group (@stanfordnlp) <a href="https://twitter.com/stanfordnlp/status/1412799581692170244?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering<br>pdf: <a href="https://t.co/41cilbLbcf">https://t.co/41cilbLbcf</a><br>abs: <a href="https://t.co/1d1kVSSj9O">https://t.co/1d1kVSSj9O</a><br>github: <a href="https://t.co/xA7yO4KD62">https://t.co/xA7yO4KD62</a> <a href="https://t.co/jSsNhhPQrk">pic.twitter.com/jSsNhhPQrk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412619697942675457?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. VidLanKD: Improving Language Understanding via Video-Distilled Knowledge  Transfer

Zineng Tang, Jaemin Cho, Hao Tan, Mohit Bansal

- retweets: 211, favorites: 50 (07/08/2021 10:03:56)

- links: [abs](https://arxiv.org/abs/2107.02681) | [pdf](https://arxiv.org/pdf/2107.02681)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Since visual perception can give rich information beyond text descriptions for world understanding, there has been increasing interest in leveraging visual grounding for language learning. Recently, vokenization has attracted attention by using the predictions of a text-to-image retrieval model as labels for language model supervision. Despite its success, the method suffers from approximation error of using finite image labels and the lack of vocabulary diversity of a small image-text dataset. To overcome these limitations, we present VidLanKD, a video-language knowledge distillation method for improving language understanding. We train a multi-modal teacher model on a video-text dataset, and then transfer its knowledge to a student language model with a text dataset. To avoid approximation error, we propose to use different knowledge distillation objectives. In addition, the use of a large-scale video-text dataset helps learn diverse and richer vocabularies. In our experiments, VidLanKD achieves consistent improvements over text-only language models and vokenization models, on several downstream language understanding tasks including GLUE, SQuAD, and SWAG. We also demonstrate the improved world knowledge, physical reasoning, and temporal reasoning capabilities of our model by evaluating on the GLUE-diagnostics, PIQA, and TRACIE datasets. Lastly, we present comprehensive ablation studies as well as visualizations of the learned text-to-video grounding results of our teacher and student language models. Our code and models are available at: https://github.com/zinengtang/VidLanKD

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can video-based grounding improve NLU tasks (incl. world/temporal/physical knowledge)?<br><br>Check out our new work *VidLanKD*: Improving Language Understanding via Video-Distilled Knowledge Transfer!<a href="https://t.co/97vq73KgLs">https://t.co/97vq73KgLs</a><br><br>Led by <a href="https://twitter.com/ZinengTang?ref_src=twsrc%5Etfw">@ZinengTang</a>, w/ <a href="https://twitter.com/HaoTan5?ref_src=twsrc%5Etfw">@HaoTan5</a> <a href="https://twitter.com/mohitban47?ref_src=twsrc%5Etfw">@MohitBan47</a> (<a href="https://twitter.com/uncnlp?ref_src=twsrc%5Etfw">@uncnlp</a>)<br><br>🧵 <a href="https://t.co/NfIQn1NrX3">pic.twitter.com/NfIQn1NrX3</a></p>&mdash; Jaemin Cho (@jmin__cho) <a href="https://twitter.com/jmin__cho/status/1412813488330661893?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Agents that Listen: High-Throughput Reinforcement Learning with Multiple  Sensory Systems

Shashank Hegde, Anssi Kanervisto, Aleksei Petrenko

- retweets: 196, favorites: 47 (07/08/2021 10:03:56)

- links: [abs](https://arxiv.org/abs/2107.02195) | [pdf](https://arxiv.org/pdf/2107.02195)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Humans and other intelligent animals evolved highly sophisticated perception systems that combine multiple sensory modalities. On the other hand, state-of-the-art artificial agents rely mostly on visual inputs or structured low-dimensional observations provided by instrumented environments. Learning to act based on combined visual and auditory inputs is still a new topic of research that has not been explored beyond simple scenarios. To facilitate progress in this area we introduce a new version of VizDoom simulator to create a highly efficient learning environment that provides raw audio observations. We study the performance of different model architectures in a series of tasks that require the agent to recognize sounds and execute instructions given in natural language. Finally, we train our agent to play the full game of Doom and find that it can consistently defeat a traditional vision-based adversary. We are currently in the process of merging the augmented simulator with the main ViZDoom code repository. Video demonstrations and experiment code can be found at https://sites.google.com/view/sound-rl.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Agents that Listen: High-Throughput Reinforcement Learning with Multiple Sensory Systems<br>pdf: <a href="https://t.co/8sZDs7qq04">https://t.co/8sZDs7qq04</a><br>project page: <a href="https://t.co/pBwtxAefp5">https://t.co/pBwtxAefp5</a><br>train agent to play the full game of Doom and find that it can consistently defeat a traditional vision-based adversary <a href="https://t.co/E0gKE8NoPY">pic.twitter.com/E0gKE8NoPY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412582314270498818?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Prioritized training on points that are learnable, worth learning, and  not yet learned

Sören Mindermann, Muhammed Razzak, Winnie Xu, Andreas Kirsch, Mrinank Sharma, Adrien Morisot, Aidan N. Gomez, Sebastian Farquhar, Jan Brauner, Yarin Gal

- retweets: 125, favorites: 79 (07/08/2021 10:03:56)

- links: [abs](https://arxiv.org/abs/2107.02565) | [pdf](https://arxiv.org/pdf/2107.02565)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent)

We introduce Goldilocks Selection, a technique for faster model training which selects a sequence of training points that are "just right". We propose an information-theoretic acquisition function -- the reducible validation loss -- and compute it with a small proxy model -- GoldiProx -- to efficiently choose training points that maximize information about a validation set. We show that the "hard" (e.g. high loss) points usually selected in the optimization literature are typically noisy, while the "easy" (e.g. low noise) samples often prioritized for curriculum learning confer less information. Further, points with uncertain labels, typically targeted by active learning, tend to be less relevant to the task. In contrast, Goldilocks Selection chooses points that are "just right" and empirically outperforms the above approaches. Moreover, the selected sequence can transfer to other architectures; practitioners can share and reuse it without the need to recreate it.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Prioritized training on points that are learnable, worth learning, and not yet learned<br>pdf: <a href="https://t.co/zHvIpkjCB9">https://t.co/zHvIpkjCB9</a><br>abs: <a href="https://t.co/mnapyrOgUn">https://t.co/mnapyrOgUn</a><br><br>Goldilocks Selection, a technique for faster model training which selects a sequence of training points that are “just right” <a href="https://t.co/fIJ4HdzZ1K">pic.twitter.com/fIJ4HdzZ1K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412627347854004226?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. iPOKE: Poking a Still Image for Controlled Stochastic Video Synthesis

Andreas Blattmann, Timo Milbich, Michael Dorkenwald, Björn Ommer

- retweets: 130, favorites: 58 (07/08/2021 10:03:56)

- links: [abs](https://arxiv.org/abs/2107.02790) | [pdf](https://arxiv.org/pdf/2107.02790)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

How would a static scene react to a local poke? What are the effects on other parts of an object if you could locally push it? There will be distinctive movement, despite evident variations caused by the stochastic nature of our world. These outcomes are governed by the characteristic kinematics of objects that dictate their overall motion caused by a local interaction. Conversely, the movement of an object provides crucial information about its underlying distinctive kinematics and the interdependencies between its parts. This two-way relation motivates learning a bijective mapping between object kinematics and plausible future image sequences. Therefore, we propose iPOKE - invertible Prediction of Object Kinematics - that, conditioned on an initial frame and a local poke, allows to sample object kinematics and establishes a one-to-one correspondence to the corresponding plausible videos, thereby providing a controlled stochastic video synthesis. In contrast to previous works, we do not generate arbitrary realistic videos, but provide efficient control of movements, while still capturing the stochastic nature of our environment and the diversity of plausible outcomes it entails. Moreover, our approach can transfer kinematics onto novel object instances and is not confined to particular object classes. Project page is available at https://bit.ly/3dJN4Lf

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">iPOKE: Poking a Still Image for Controlled Stochastic Video Synthesis<br>pdf: <a href="https://t.co/HG15q17E35">https://t.co/HG15q17E35</a><br>abs: <a href="https://t.co/OO1kcWpEXK">https://t.co/OO1kcWpEXK</a><br>project page: <a href="https://t.co/PKcR7OnUH2">https://t.co/PKcR7OnUH2</a><br>github: <a href="https://t.co/CyUPv26aSN">https://t.co/CyUPv26aSN</a> <a href="https://t.co/VULe8L4IxS">pic.twitter.com/VULe8L4IxS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412584633607331842?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Foreground-Aware Stylization and Consensus Pseudo-Labeling for Domain  Adaptation of First-Person Hand Segmentation

Takehiko Ohkawa, Takuma Yagi, Atsushi Hashimoto, Yoshitaka Ushiku, Yoichi Sato

- retweets: 132, favorites: 39 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02718) | [pdf](https://arxiv.org/pdf/2107.02718)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Hand segmentation is a crucial task in first-person vision. Since first-person images exhibit strong bias in appearance among different environments, adapting a pre-trained segmentation model to a new domain is required in hand segmentation. Here, we focus on appearance gaps for hand regions and backgrounds separately. We propose (i) foreground-aware image stylization and (ii) consensus pseudo-labeling for domain adaptation of hand segmentation. We stylize source images independently for the foreground and background using target images as style. To resolve the domain shift that the stylization has not addressed, we apply careful pseudo-labeling by taking a consensus between the models trained on the source and stylized source images. We validated our method on domain adaptation of hand segmentation from real and simulation images. Our method achieved state-of-the-art performance in both settings. We also demonstrated promising results in challenging multi-target domain adaptation and domain generalization settings. Code is available at https://github.com/ut-vision/FgSty-CPL.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">IEEE Accessに採択された論文を公開しました！<br>MIRU&#39;21ロングオーラルでも発表します！<br>Foreground-Aware Stylization and Consensus Pseudo-Labeling for Domain Adaptation of First-Person Hand Segmentation, w/ <a href="https://twitter.com/omron_sinicx?ref_src=twsrc%5Etfw">@omron_sinicx</a> <br>project: <a href="https://t.co/k4m2Xzq8ev">https://t.co/k4m2Xzq8ev</a><br>paper: <a href="https://t.co/RJTRC3udsR">https://t.co/RJTRC3udsR</a> <a href="https://t.co/KK7wKzcQWI">pic.twitter.com/KK7wKzcQWI</a></p>&mdash; Take Ohkawa / 大川 武彦 (@tkhkaeio) <a href="https://twitter.com/tkhkaeio/status/1412656160574894080?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Garbage, Glitter, or Gold: Assigning Multi-dimensional Quality Scores to  Social Media Seeds for Web Archive Collections

Alexander C. Nwala, Michele C. Weigle, Michael L. Nelson

- retweets: 132, favorites: 4 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02680) | [pdf](https://arxiv.org/pdf/2107.02680)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

From popular uprisings to pandemics, the Web is an essential source consulted by scientists and historians for reconstructing and studying past events. Unfortunately, the Web is plagued by reference rot which causes important Web resources to disappear. Web archive collections help reduce the costly effects of reference rot by saving Web resources that chronicle important stories/events before they disappear. These collections often begin with URLs called seeds, hand-selected by experts or scraped from social media. The quality of social media content varies widely, therefore, we propose a framework for assigning multi-dimensional quality scores to social media seeds for Web archive collections about stories and events. We leveraged contributions from social media research for attributing quality to social media content and users based on credibility, reputation, and influence. We combined these with additional contributions from the Web archive research that emphasizes the importance of considering geographical and temporal constraints when selecting seeds. Next, we developed the Quality Proxies (QP) framework which assigns seeds extracted from social media a quality score across 10 major dimensions: popularity, geographical, temporal, subject expert, retrievability, relevance, reputation, and scarcity. We instantiated the framework and showed that seeds can be scored across multiple QP classes that map to different policies for ranking seeds such as prioritizing seeds from local news, reputable and/or popular sources, etc. The QP framework is extensible and robust. Our results showed that Quality Proxies resulted in the selection of quality seeds with increased precision (by ~0.13) when novelty is and is not prioritized. These contributions provide an explainable score applicable to rank and select quality seeds for Web archive collections and other domains.




# 13. MAJORITY-3SAT (and Related Problems) in Polynomial Time

Shyan Akmal, Ryan Williams

- retweets: 72, favorites: 28 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02748) | [pdf](https://arxiv.org/pdf/2107.02748)
- [cs.CC](https://arxiv.org/list/cs.CC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DS](https://arxiv.org/list/cs.DS/recent)

Majority-SAT is the problem of determining whether an input $n$-variable formula in conjunctive normal form (CNF) has at least $2^{n-1}$ satisfying assignments. Majority-SAT and related problems have been studied extensively in various AI communities interested in the complexity of probabilistic planning and inference. Although Majority-SAT has been known to be PP-complete for over 40 years, the complexity of a natural variant has remained open: Majority-$k$SAT, where the input CNF formula is restricted to have clause width at most $k$.   We prove that for every $k$, Majority-$k$SAT is in P. In fact, for any positive integer $k$ and rational $\rho \in (0,1)$ with bounded denominator, we give an algorithm that can determine whether a given $k$-CNF has at least $\rho \cdot 2^n$ satisfying assignments, in deterministic linear time (whereas the previous best-known algorithm ran in exponential time). Our algorithms have interesting positive implications for counting complexity and the complexity of inference, significantly reducing the known complexities of related problems such as E-MAJ-$k$SAT and MAJ-MAJ-$k$SAT. At the heart of our approach is an efficient method for solving threshold counting problems by extracting sunflowers found in the corresponding set system of a $k$-CNF.   We also show that the tractability of Majority-$k$SAT is somewhat fragile. For the closely related GtMajority-SAT problem (where we ask whether a given formula has greater than $2^{n-1}$ satisfying assignments) which is known to be PP-complete, we show that GtMajority-$k$SAT is in P for $k\le 3$, but becomes NP-complete for $k\geq 4$. These results are counterintuitive, because the ``natural'' classifications of these problems would have been PP-completeness, and because there is a stark difference in the complexity of GtMajority-$k$SAT and Majority-$k$SAT for all $k\ge 4$.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">So, Shyan Akmal and <a href="https://twitter.com/rrwilliams?ref_src=twsrc%5Etfw">@rrwilliams</a> solve this open problem I needed to know the answer to 11 years ago, and in the unexpected direction? Majority-kSAT in P (linear time even!) for all k? This is awesome. <a href="https://t.co/sYZOnfQKLX">https://t.co/sYZOnfQKLX</a> <a href="https://t.co/d9y09Gdau0">https://t.co/d9y09Gdau0</a></p>&mdash; Antonio E. Porreca 🐳💉💉🛡️ (@aeporreca) <a href="https://twitter.com/aeporreca/status/1412842896231911431?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Comparing PCG metrics with Human Evaluation in Minecraft Settlement  Generation

Jean-Baptiste Hervé, Christoph Salge

- retweets: 42, favorites: 21 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02457) | [pdf](https://arxiv.org/pdf/2107.02457)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

There are a range of metrics that can be applied to the artifacts produced by procedural content generation, and several of them come with qualitative claims. In this paper, we adapt a range of existing PCG metrics to generated Minecraft settlements, develop a few new metrics inspired by PCG literature, and compare the resulting measurements to existing human evaluations. The aim is to analyze how those metrics capture human evaluation scores in different categories, how the metrics generalize to another game domain, and how metrics deal with more complex artifacts. We provide an exploratory look at a variety of metrics and provide an information gain and several correlation analyses. We found some relationships between human scores and metrics counting specific elements, measuring the diversity of blocks and measuring the presence of crafting materials for the present complex blocks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My very first paper, &#39;Comparing PCG metrics with Human Evaluation in Minecraft Settlement Generation&#39; has just been published. I co-authored it with my main PhD supervisor <a href="https://twitter.com/ChristophSalge?ref_src=twsrc%5Etfw">@ChristophSalge</a> <a href="https://t.co/q9UgFoEL2v">https://t.co/q9UgFoEL2v</a> <a href="https://t.co/Twe4OTAzlU">pic.twitter.com/Twe4OTAzlU</a></p>&mdash; JB Hervé (@jibeherve) <a href="https://twitter.com/jibeherve/status/1412760501499613188?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. AdaSpeech 3: Adaptive Text to Speech for Spontaneous Style

Yuzi Yan, Xu Tan, Bohan Li, Guangyan Zhang, Tao Qin, Sheng Zhao, Yuan Shen, Wei-Qiang Zhang, Tie-Yan Liu

- retweets: 37, favorites: 24 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02530) | [pdf](https://arxiv.org/pdf/2107.02530)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

While recent text to speech (TTS) models perform very well in synthesizing reading-style (e.g., audiobook) speech, it is still challenging to synthesize spontaneous-style speech (e.g., podcast or conversation), mainly because of two reasons: 1) the lack of training data for spontaneous speech; 2) the difficulty in modeling the filled pauses (um and uh) and diverse rhythms in spontaneous speech. In this paper, we develop AdaSpeech 3, an adaptive TTS system that fine-tunes a well-trained reading-style TTS model for spontaneous-style speech. Specifically, 1) to insert filled pauses (FP) in the text sequence appropriately, we introduce an FP predictor to the TTS model; 2) to model the varying rhythms, we introduce a duration predictor based on mixture of experts (MoE), which contains three experts responsible for the generation of fast, medium and slow speech respectively, and fine-tune it as well as the pitch predictor for rhythm adaptation; 3) to adapt to other speaker timbre, we fine-tune some parameters in the decoder with few speech data. To address the challenge of lack of training data, we mine a spontaneous speech dataset to support our research this work and facilitate future research on spontaneous TTS. Experiments show that AdaSpeech 3 synthesizes speech with natural FP and rhythms in spontaneous styles, and achieves much better MOS and SMOS scores than previous adaptive TTS systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">AdaSpeech 3: Adaptive Text to Speech for Spontaneous Style<br>pdf: <a href="https://t.co/2tDfMazzKV">https://t.co/2tDfMazzKV</a><br>abs: <a href="https://t.co/z4GwoYPFPV">https://t.co/z4GwoYPFPV</a><br>project page: <a href="https://t.co/gNwH4PDl7G">https://t.co/gNwH4PDl7G</a><br>synthesizes speech with natural FP and rhythms in spontaneous styles <a href="https://t.co/oLYu40q96d">pic.twitter.com/oLYu40q96d</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412579199429263366?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Feature Fusion Vision Transformer for Fine-Grained Visual Categorization

Jun Wang, Xiaohan Yu, Yongsheng Gao

- retweets: 30, favorites: 28 (07/08/2021 10:03:57)

- links: [abs](https://arxiv.org/abs/2107.02341) | [pdf](https://arxiv.org/pdf/2107.02341)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The core for tackling the fine-grained visual categorization (FGVC) is to learn subtle yet discriminative features. Most previous works achieve this by explicitly selecting the discriminative parts or integrating the attention mechanism via CNN-based approaches.However, these methods enhance the computational complexity and make the modeldominated by the regions containing the most of the objects. Recently, vision trans-former (ViT) has achieved SOTA performance on general image recognition tasks. Theself-attention mechanism aggregates and weights the information from all patches to the classification token, making it perfectly suitable for FGVC. Nonetheless, the classifi-cation token in the deep layer pays more attention to the global information, lacking the local and low-level features that are essential for FGVC. In this work, we proposea novel pure transformer-based framework Feature Fusion Vision Transformer (FFVT)where we aggregate the important tokens from each transformer layer to compensate thelocal, low-level and middle-level information. We design a novel token selection mod-ule called mutual attention weight selection (MAWS) to guide the network effectively and efficiently towards selecting discriminative tokens without introducing extra param-eters. We verify the effectiveness of FFVT on three benchmarks where FFVT achieves the state-of-the-art performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Feature Fusion Vision Transformer Fine-Grained Visual Categorization<br>pdf: <a href="https://t.co/Y048ovs758">https://t.co/Y048ovs758</a><br>abs: <a href="https://t.co/Xmr1NN9AhZ">https://t.co/Xmr1NN9AhZ</a> <a href="https://t.co/5kyO6zKfSG">pic.twitter.com/5kyO6zKfSG</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1412592315584192513?ref_src=twsrc%5Etfw">July 7, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



