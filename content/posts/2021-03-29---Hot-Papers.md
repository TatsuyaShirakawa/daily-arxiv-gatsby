---
title: Hot Papers 2021-03-29
date: 2021-03-30T08:47:17.Z
template: "post"
draft: false
slug: "hot-papers-2021-03-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-03-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. Understanding Robustness of Transformers for Image Classification

Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, Andreas Veit

- retweets: 1541, favorites: 202 (03/30/2021 08:47:17)

- links: [abs](https://arxiv.org/abs/2103.14586) | [pdf](https://arxiv.org/pdf/2103.14586)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep Convolutional Neural Networks (CNNs) have long been the architecture of choice for computer vision tasks. Recently, Transformer-based architectures like Vision Transformer (ViT) have matched or even surpassed ResNets for image classification. However, details of the Transformer architecture -- such as the use of non-overlapping patches -- lead one to wonder whether these networks are as robust. In this paper, we perform an extensive study of a variety of different measures of robustness of ViT models and compare the findings to ResNet baselines. We investigate robustness to input perturbations as well as robustness to model perturbations. We find that when pre-trained with a sufficient amount of data, ViT models are at least as robust as the ResNet counterparts on a broad range of perturbations. We also find that Transformers are robust to the removal of almost any single layer, and that while activations from later layers are highly correlated with each other, they nevertheless play an important role in classification.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Understanding Robustness of Transformers for Image Classification<br>pdf: <a href="https://t.co/uyMhFw3KB9">https://t.co/uyMhFw3KB9</a><br>abs: <a href="https://t.co/Z2xhJasCZm">https://t.co/Z2xhJasCZm</a> <a href="https://t.co/sd7ISqHhrl">pic.twitter.com/sd7ISqHhrl</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376339015981486081?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. A Practical Survey on Faster and Lighter Transformers

Quentin Fournier, Gaétan Marceau Caron, Daniel Aloise

- retweets: 1224, favorites: 131 (03/30/2021 08:47:17)

- links: [abs](https://arxiv.org/abs/2103.14636) | [pdf](https://arxiv.org/pdf/2103.14636)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recurrent neural networks are effective models to process sequences. However, they are unable to learn long-term dependencies because of their inherent sequential nature. As a solution, Vaswani et al. introduced the Transformer, a model solely based on the attention mechanism that is able to relate any two positions of the input sequence, hence modelling arbitrary long dependencies. The Transformer has improved the state-of-the-art across numerous sequence modelling tasks. However, its effectiveness comes at the expense of a quadratic computational and memory complexity with respect to the sequence length, hindering its adoption. Fortunately, the deep learning community has always been interested in improving the models' efficiency, leading to a plethora of solutions such as parameter sharing, pruning, mixed-precision, and knowledge distillation. Recently, researchers have directly addressed the Transformer's limitation by designing lower-complexity alternatives such as the Longformer, Reformer, Linformer, and Performer. However, due to the wide range of solutions, it has become challenging for the deep learning community to determine which methods to apply in practice to meet the desired trade-off between capacity, computation, and memory. This survey addresses this issue by investigating popular approaches to make the Transformer faster and lighter and by providing a comprehensive explanation of the methods' strengths, limitations, and underlying assumptions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A survey on popular approaches used to make the Transformer lighter and faster.<br><br>Also, look at the fast adoption of Transformers over the last couple of years. Close to 2000 machine learning papers have mentioned the word &quot;Transformer&quot;! <a href="https://t.co/vYb38vpMyn">https://t.co/vYb38vpMyn</a> <a href="https://t.co/2tKFVMfaAH">pic.twitter.com/2tKFVMfaAH</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1376600060264337409?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Dodrio: Exploring Transformer Models with Interactive Visualization

Zijie J. Wang, Robert Turko, Duen Horng Chau

- retweets: 1190, favorites: 143 (03/30/2021 08:47:17)

- links: [abs](https://arxiv.org/abs/2103.14625) | [pdf](https://arxiv.org/pdf/2103.14625)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Why do large pre-trained transformer-based models perform so well across a wide variety of NLP tasks? Recent research suggests the key may lie in multi-headed attention mechanism's ability to learn and represent linguistic information. Understanding how these models represent both syntactic and semantic knowledge is vital to investigate why they succeed and fail, what they have learned, and how they can improve. We present Dodrio, an open-source interactive visualization tool to help NLP researchers and practitioners analyze attention mechanisms in transformer-based models with linguistic knowledge. Dodrio tightly integrates an overview that summarizes the roles of different attention heads, and detailed views that help users compare attention weights with the syntactic structure and semantic information in the input text. To facilitate the visual comparison of attention weights and linguistic knowledge, Dodrio applies different graph visualization techniques to represent attention weights with longer input text. Case studies highlight how Dodrio provides insights into understanding the attention mechanism in transformer-based models. Dodrio is available at https://poloclub.github.io/dodrio/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dodrio: Exploring Transformer Models with Interactive Visualization<br>pdf: <a href="https://t.co/9fOpC9rLQW">https://t.co/9fOpC9rLQW</a><br>abs: <a href="https://t.co/9G5K3FQqW9">https://t.co/9G5K3FQqW9</a><br>project page: <a href="https://t.co/JecfZrdQcX">https://t.co/JecfZrdQcX</a> <a href="https://t.co/FKaqISyH2N">pic.twitter.com/FKaqISyH2N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376339818121195521?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. MedSelect: Selective Labeling for Medical Image Classification Combining  Meta-Learning with Deep Reinforcement Learning

Akshay Smit, Damir Vrabac, Yujie He, Andrew Y. Ng, Andrew L. Beam, Pranav Rajpurkar

- retweets: 1090, favorites: 130 (03/30/2021 08:47:18)

- links: [abs](https://arxiv.org/abs/2103.14339) | [pdf](https://arxiv.org/pdf/2103.14339)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a selective learning method using meta-learning and deep reinforcement learning for medical image interpretation in the setting of limited labeling resources. Our method, MedSelect, consists of a trainable deep learning selector that uses image embeddings obtained from contrastive pretraining for determining which images to label, and a non-parametric selector that uses cosine similarity to classify unseen images. We demonstrate that MedSelect learns an effective selection strategy outperforming baseline selection strategies across seen and unseen medical conditions for chest X-ray interpretation. We also perform an analysis of the selections performed by MedSelect comparing the distribution of latent embeddings and clinical features, and find significant differences compared to the strongest performing baseline. We believe that our method may be broadly applicable across medical imaging settings where labels are expensive to acquire.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It is $$$ to label a large number of medical images for deep learning.<br><br>Can we select the “best” few images to label in a single step? <a href="https://twitter.com/hashtag/MedSelect?src=hash&amp;ref_src=twsrc%5Etfw">#MedSelect</a><br><br>Yes, we can! Using meta learning w/ deep RL <a href="https://t.co/8CcSDsqQHk">https://t.co/8CcSDsqQHk</a><a href="https://twitter.com/AkshaySmit?ref_src=twsrc%5Etfw">@AkshaySmit</a> <a href="https://twitter.com/dvrabac?ref_src=twsrc%5Etfw">@dvrabac</a> <a href="https://twitter.com/yujiehe9?ref_src=twsrc%5Etfw">@yujiehe9</a> &amp; <a href="https://twitter.com/AndrewYNg?ref_src=twsrc%5Etfw">@AndrewYNg</a> <a href="https://twitter.com/AndrewLBeam?ref_src=twsrc%5Etfw">@AndrewLBeam</a> <br><br>1/n <a href="https://t.co/glUwxEbQ1f">pic.twitter.com/glUwxEbQ1f</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1376588135904112641?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Leveraging neural representations for facilitating access to  untranscribed speech from endangered languages

Nay San, Martijn Bartelds, Mitchell Browne, Lily Clifford, Fiona Gibson, John Mansfield, David Nash, Jane Simpson, Myfany Turpin, Maria Vollmer, Sasha Wilmoth, Dan Jurafsky

- retweets: 306, favorites: 74 (03/30/2021 08:47:18)

- links: [abs](https://arxiv.org/abs/2103.14583) | [pdf](https://arxiv.org/pdf/2103.14583)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

For languages with insufficient resources to train speech recognition systems, query-by-example spoken term detection (QbE-STD) offers a way of accessing an untranscribed speech corpus by helping identify regions where spoken query terms occur. Yet retrieval performance can be poor when the query and corpus are spoken by different speakers and produced in different recording conditions. Using data selected from a variety of speakers and recording conditions from 7 Australian Aboriginal languages and a regional variety of Dutch, all of which are endangered or vulnerable, we evaluated whether QbE-STD performance on these languages could be improved by leveraging representations extracted from the pre-trained English wav2vec 2.0 model. Compared to the use of Mel-frequency cepstral coefficients and bottleneck features, we find that representations from the middle layers of the wav2vec 2.0 Transformer offer large gains in task performance (between 56% and 86%). While features extracted using the pre-trained English model yielded improved detection on all the evaluation languages, better detection performance was associated with the evaluation language's phonological similarity to English.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited about our latest paper! 🎉 We show that neural representations from the English wav2vec 2.0 model can be used for searching untranscribed speech from endangered languages which have insufficient data for training or fine-tuning.<br><br>📝 <a href="https://t.co/5aoAnPkcMj">https://t.co/5aoAnPkcMj</a> <a href="https://t.co/C3wDZ93MYA">pic.twitter.com/C3wDZ93MYA</a></p>&mdash; Martijn Bartelds (@BarteldsMartijn) <a href="https://twitter.com/BarteldsMartijn/status/1376543710138269704?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with  Differentiable Duration Modeling

Isaac Elias, Heiga Zen, Jonathan Shen, Yu Zhang, Jia Ye, RJ Ryan, Yonghui Wu

- retweets: 240, favorites: 85 (03/30/2021 08:47:18)

- links: [abs](https://arxiv.org/abs/2103.14574) | [pdf](https://arxiv.org/pdf/2103.14574)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper introduces Parallel Tacotron 2, a non-autoregressive neural text-to-speech model with a fully differentiable duration model which does not require supervised duration signals. The duration model is based on a novel attention mechanism and an iterative reconstruction loss based on Soft Dynamic Time Warping, this model can learn token-frame alignments as well as token durations automatically. Experimental results show that Parallel Tacotron 2 outperforms baselines in subjective naturalness in several diverse multi speaker evaluations. Its duration control capability is also demonstrated.

<blockquote class="twitter-tweet"><p lang="it" dir="ltr">Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with Differentiable Duration Modeling<br>pdf: <a href="https://t.co/K1LblNx4FN">https://t.co/K1LblNx4FN</a><br>abs: <a href="https://t.co/X8qDoKIzoG">https://t.co/X8qDoKIzoG</a> <a href="https://t.co/be3NDc353N">pic.twitter.com/be3NDc353N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376352367646498817?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Deadlock-Free Session Types in Linear Haskell

Wen Kokke, Ornela Dardha

- retweets: 256, favorites: 53 (03/30/2021 08:47:18)

- links: [abs](https://arxiv.org/abs/2103.14481) | [pdf](https://arxiv.org/pdf/2103.14481)
- [cs.PL](https://arxiv.org/list/cs.PL/recent)

Priority Sesh is a library for session-typed communication in Linear Haskell which offers strong compile-time correctness guarantees. Priority Sesh offers two deadlock-free APIs for session-typed communication. The first guarantees deadlock freedom by restricting the process structure to trees and forests. It is simple and composeable, but rules out cyclic structures. The second guarantees deadlock freedom via priorities, which allows the programmer to safely use cyclic structures as well.   Our library relies on Linear Haskell to guarantee linearity, which leads to easy-to-write session types and highly idiomatic code, and lets us avoid the complex encodings of linearity in the Haskell type system that made previous libraries difficult to use.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Folks are wasting no time to make good use of Linear Haskell. 🚀 This paper promises deadlock-free concurrent programs without needing exotic encodings: <a href="https://t.co/IjLuwisZln">https://t.co/IjLuwisZln</a>.</p>&mdash; Tweag (@tweagio) <a href="https://twitter.com/tweagio/status/1376470492220116994?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. COTR: Correspondence Transformer for Matching Across Images

Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, Kwang Moo Yi

- retweets: 125, favorites: 113 (03/30/2021 08:47:18)

- links: [abs](https://arxiv.org/abs/2103.14167) | [pdf](https://arxiv.org/pdf/2103.14167)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We propose a novel framework for finding correspondences in images based on a deep neural network that, given two images and a query point in one of them, finds its correspondence in the other. By doing so, one has the option to query only the points of interest and retrieve sparse correspondences, or to query all points in an image and obtain dense mappings. Importantly, in order to capture both local and global priors, and to let our model relate between image regions using the most relevant among said priors, we realize our network using a transformer. At inference time, we apply our correspondence network by recursively zooming in around the estimates, yielding a multiscale pipeline able to provide highly-accurate correspondences. Our method significantly outperforms the state of the art on both sparse and dense correspondence problems on multiple datasets and tasks, ranging from wide-baseline stereo to optical flow, without any retraining for a specific dataset. We commit to releasing data, code, and all the tools necessary to train from scratch and ensure reproducibility.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">COTR: Correspondence Transformer for Matching Across Images<br>pdf: <a href="https://t.co/KAHp0BmUDZ">https://t.co/KAHp0BmUDZ</a><br>abs: <a href="https://t.co/LRHqQKO23L">https://t.co/LRHqQKO23L</a> <a href="https://t.co/08MG6aEPlk">pic.twitter.com/08MG6aEPlk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376336996185034753?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">COTR: Correspondence Transformer for Matching Across Images<br><br>Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, <a href="https://twitter.com/kwangmoo_yi?ref_src=twsrc%5Etfw">@kwangmoo_yi</a> <a href="https://t.co/Hd82QOdKN8">https://t.co/Hd82QOdKN8</a><br><br>ResNet50 dense -&gt; concat -&gt; transformer -&gt; MLP to regress the (x2,y2) given query (random) (x1,y1) <a href="https://t.co/43UGw57iqF">pic.twitter.com/43UGw57iqF</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1376473557396508673?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dense or sparse, finding correspondences requires answering the same question: &quot;where did my point go?&quot; Our new work implements this via a Transformer. Trained once on Phototourism, works for a whole host of things! <a href="https://twitter.com/WeiJian48036441?ref_src=twsrc%5Etfw">@WeiJian48036441</a> <a href="https://twitter.com/taiyasaki?ref_src=twsrc%5Etfw">@taiyasaki</a><a href="https://t.co/inQVTevxd7">https://t.co/inQVTevxd7</a> <a href="https://t.co/K7w4giPvl8">pic.twitter.com/K7w4giPvl8</a></p>&mdash; Kwang Moo Yi (@kwangmoo_yi) <a href="https://twitter.com/kwangmoo_yi/status/1376625866256515073?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Describing and Localizing Multiple Changes with Transformers

Yue Qiu, Shintaro Yamamoto, Kodai Nakashima, Ryota Suzuki, Kenji Iwata, Hirokatsu Kataoka, Yutaka Satoh

- retweets: 103, favorites: 49 (03/30/2021 08:47:19)

- links: [abs](https://arxiv.org/abs/2103.14146) | [pdf](https://arxiv.org/pdf/2103.14146)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Change captioning tasks aim to detect changes in image pairs observed before and after a scene change and generate a natural language description of the changes. Existing change captioning studies have mainly focused on scenes with a single change. However, detecting and describing multiple changed parts in image pairs is essential for enhancing adaptability to complex scenarios. We solve the above issues from three aspects: (i) We propose a CG-based multi-change captioning dataset; (ii) We benchmark existing state-of-the-art methods of single change captioning on multi-change captioning; (iii) We further propose Multi-Change Captioning transformers (MCCFormers) that identify change regions by densely correlating different regions in image pairs and dynamically determines the related change regions with words in sentences. The proposed method obtained the highest scores on four conventional change captioning evaluation metrics for multi-change captioning. In addition, existing methods generate a single attention map for multiple changes and lack the ability to distinguish change regions. In contrast, our proposed method can separate attention maps for each change and performs well with respect to change localization. Moreover, the proposed framework outperformed the previous state-of-the-art methods on an existing change captioning benchmark, CLEVR-Change, by a large margin (+6.1 on BLEU-4 and +9.7 on CIDEr scores), indicating its general ability in change captioning tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Describing and Localizing Multiple Changes with Transformers<br>pdf: <a href="https://t.co/x0CZt9068P">https://t.co/x0CZt9068P</a><br>abs: <a href="https://t.co/goz7LLCVne">https://t.co/goz7LLCVne</a><br>project page: <a href="https://t.co/lbeSNBhOrO">https://t.co/lbeSNBhOrO</a> <a href="https://t.co/AgE9pS1h6K">pic.twitter.com/AgE9pS1h6K</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376336657524359170?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. What happens when a journal converts to Open Access? A bibliometric  analysis

Fakhri Momeni, Philipp Mayr, Nicholas Fraser, Isabella Peters

- retweets: 110, favorites: 20 (03/30/2021 08:47:19)

- links: [abs](https://arxiv.org/abs/2103.14522) | [pdf](https://arxiv.org/pdf/2103.14522)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

In recent years, increased stakeholder pressure to transition research to Open Access has led to many journals converting, or 'flipping', from a closed access (CA) to an open access (OA) publishing model. Changing the publishing model can influence the decision of authors to submit their papers to a journal, and increased article accessibility may influence citation behaviour. In this paper we aimed to understand how flipping a journal to an OA model influences the journal's future publication volumes and citation impact. We analysed two independent sets of journals that had flipped to an OA model, one from the Directory of Open Access Journals (DOAJ) and one from the Open Access Directory (OAD), and compared their development with two respective control groups of similar journals. For bibliometric analyses, journals were matched to the Scopus database. We assessed changes in the number of articles published over time, as well as two citation metrics at the journal and article level: the normalised impact factor (IF) and the average relative citations (ARC), respectively. Our results show that overall, journals that flipped to an OA model increased their publication output compared to journals that remained closed. Mean normalised IF and ARC also generally increased following the flip to an OA model, at a greater rate than was observed in the control groups. However, the changes appear to vary largely by scientific discipline. Overall, these results indicate that flipping to an OA publishing model can bring positive changes to a journal.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What happens when a journal converts to Open Access? <br>Preprint: <a href="https://t.co/ly4TBJ3Wyw">https://t.co/ly4TBJ3Wyw</a><br>Research in <a href="https://twitter.com/BMBF_Bund?ref_src=twsrc%5Etfw">@BMBF_Bund</a> OASE project with Fakhri Momeni <a href="https://twitter.com/nicholasmfraser?ref_src=twsrc%5Etfw">@nicholasmfraser</a> <a href="https://twitter.com/Isabella83?ref_src=twsrc%5Etfw">@Isabella83</a> <a href="https://twitter.com/hashtag/flipping?src=hash&amp;ref_src=twsrc%5Etfw">#flipping</a> <a href="https://twitter.com/hashtag/journals?src=hash&amp;ref_src=twsrc%5Etfw">#journals</a> <a href="https://twitter.com/hashtag/openaccess?src=hash&amp;ref_src=twsrc%5Etfw">#openaccess</a> <a href="https://twitter.com/DOAJplus?ref_src=twsrc%5Etfw">@DOAJplus</a> <a href="https://t.co/25edKB9lOs">pic.twitter.com/25edKB9lOs</a></p>&mdash; Philipp Mayr (@Philipp_Mayr) <a href="https://twitter.com/Philipp_Mayr/status/1376484992528814084?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Few-Shot Human Motion Transfer by Personalized Geometry and Texture  Modeling

Zhichao Huang, Xintong Han, Jia Xu, Tong Zhang

- retweets: 81, favorites: 46 (03/30/2021 08:47:19)

- links: [abs](https://arxiv.org/abs/2103.14338) | [pdf](https://arxiv.org/pdf/2103.14338)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a new method for few-shot human motion transfer that achieves realistic human image generation with only a small number of appearance inputs. Despite recent advances in single person motion transfer, prior methods often require a large number of training images and take long training time. One promising direction is to perform few-shot human motion transfer, which only needs a few of source images for appearance transfer. However, it is particularly challenging to obtain satisfactory transfer results. In this paper, we address this issue by rendering a human texture map to a surface geometry (represented as a UV map), which is personalized to the source person. Our geometry generator combines the shape information from source images, and the pose information from 2D keypoints to synthesize the personalized UV map. A texture generator then generates the texture map conditioned on the texture of source images to fill out invisible parts. Furthermore, we may fine-tune the texture map on the manifold of the texture generator from a few source images at the test time, which improves the quality of the texture map without over-fitting or artifacts. Extensive experiments show the proposed method outperforms state-of-the-art methods both qualitatively and quantitatively. Our code is available at https://github.com/HuangZhiChao95/FewShotMotionTransfer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling<br>pdf: <a href="https://t.co/tIi5lodDEQ">https://t.co/tIi5lodDEQ</a><br>abs: <a href="https://t.co/xwataSPkqK">https://t.co/xwataSPkqK</a> <a href="https://t.co/DFPVyRbUmW">pic.twitter.com/DFPVyRbUmW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376343318892793861?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Planar Surface Reconstruction from Sparse Views

Linyi Jin, Shengyi Qian, Andrew Owens, David F. Fouhey

- retweets: 72, favorites: 29 (03/30/2021 08:47:19)

- links: [abs](https://arxiv.org/abs/2103.14644) | [pdf](https://arxiv.org/pdf/2103.14644)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The paper studies planar surface reconstruction of indoor scenes from two views with unknown camera poses. While prior approaches have successfully created object-centric reconstructions of many scenes, they fail to exploit other structures, such as planes, which are typically the dominant components of indoor scenes. In this paper, we reconstruct planar surfaces from multiple views, while jointly estimating camera pose. Our experiments demonstrate that our method is able to advance the state of the art of reconstruction from sparse views, on challenging scenes from Matterport3D. Project site: https://jinlinyi.github.io/SparsePlanes/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Planar Surface Reconstruction from Sparse Views<a href="https://twitter.com/jin_linyi?ref_src=twsrc%5Etfw">@jin_linyi</a>, <a href="https://twitter.com/JasonQSY?ref_src=twsrc%5Etfw">@JasonQSY</a>, <a href="https://twitter.com/andrewhowens?ref_src=twsrc%5Etfw">@andrewhowens</a>, David F. Fouhey<br><br>Idea: WxBS, but for planes not features. Plane detector, planes descriptor, camera pose. Neat idea!<a href="https://t.co/qBtK8ZTkSr">https://t.co/qBtK8ZTkSr</a> <br>P.S. Will be another multi-plane paper tweet today <a href="https://t.co/Rm1uxL2Nz3">pic.twitter.com/Rm1uxL2Nz3</a></p>&mdash; Dmytro Mishkin (@ducha_aiki) <a href="https://twitter.com/ducha_aiki/status/1376503222756651012?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes

Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, Dieter Fox

- retweets: 30, favorites: 41 (03/30/2021 08:47:19)

- links: [abs](https://arxiv.org/abs/2103.14127) | [pdf](https://arxiv.org/pdf/2103.14127)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Grasping unseen objects in unconstrained, cluttered environments is an essential skill for autonomous robotic manipulation. Despite recent progress in full 6-DoF grasp learning, existing approaches often consist of complex sequential pipelines that possess several potential failure points and run-times unsuitable for closed-loop grasping. Therefore, we propose an end-to-end network that efficiently generates a distribution of 6-DoF parallel-jaw grasps directly from a depth recording of a scene. Our novel grasp representation treats 3D points of the recorded point cloud as potential grasp contacts. By rooting the full 6-DoF grasp pose and width in the observed point cloud, we can reduce the dimensionality of our grasp representation to 4-DoF which greatly facilitates the learning process. Our class-agnostic approach is trained on 17 million simulated grasps and generalizes well to real world sensor data. In a robotic grasping study of unseen objects in structured clutter we achieve over 90% success rate, cutting the failure rate in half compared to a recent state-of-the-art method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes<br>pdf: <a href="https://t.co/KzXfaM9aU5">https://t.co/KzXfaM9aU5</a><br>abs: <a href="https://t.co/CHkFxYNCx2">https://t.co/CHkFxYNCx2</a><br>project page: <a href="https://t.co/j7DrOicF7G">https://t.co/j7DrOicF7G</a> <a href="https://t.co/NwcLHufwlH">pic.twitter.com/NwcLHufwlH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1376366820895117317?ref_src=twsrc%5Etfw">March 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



