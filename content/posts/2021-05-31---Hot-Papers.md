---
title: Hot Papers 2021-05-31
date: 2021-06-01T07:56:23.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-31"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-31"
socialImage: "/media/flying-marine.jpg"

---

# 1. ByT5: Towards a token-free future with pre-trained byte-to-byte models

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel

- retweets: 4496, favorites: 446 (06/01/2021 07:56:23)

- links: [abs](https://arxiv.org/abs/2105.13626) | [pdf](https://arxiv.org/pdf/2105.13626)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. Encoding text as a sequence of tokens requires a tokenizer, which is typically created as an independent artifact from the model. Token-free models that instead operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We carefully characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our experiments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ByT5: Towards a token-free future with pre-trained byte-to-byte models<br><br>Shows that byte-level models are competitive with their token-level counterparts and more robust to noise.<br><br>abs: <a href="https://t.co/Nt6mgTIi29">https://t.co/Nt6mgTIi29</a><br>code: <a href="https://t.co/cRWQfFDBFv">https://t.co/cRWQfFDBFv</a> <a href="https://t.co/wZtxmqXjsf">pic.twitter.com/wZtxmqXjsf</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1399164694938591232?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Boosting Monocular Depth Estimation Models to High-Resolution via  Content-Adaptive Multi-Resolution Merging

S. Mahdi H. Miangoleh, Sebastian Dille, Long Mai, Sylvain Paris, Yağız Aksoy

- retweets: 3720, favorites: 329 (06/01/2021 07:56:24)

- links: [abs](https://arxiv.org/abs/2105.14021) | [pdf](https://arxiv.org/pdf/2105.14021)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Neural networks have shown great abilities in estimating depth from a single image. However, the inferred depth maps are well below one-megapixel resolution and often lack fine-grained details, which limits their practicality. Our method builds on our analysis on how the input resolution and the scene structure affects depth estimation performance. We demonstrate that there is a trade-off between a consistent scene structure and the high-frequency details, and merge low- and high-resolution estimations to take advantage of this duality using a simple depth merging network. We present a double estimation method that improves the whole-image depth estimation and a patch selection method that adds local details to the final result. We demonstrate that by merging estimations at different resolutions with changing context, we can generate multi-megapixel depth maps with a high level of detail using a pre-trained model.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging<br>pdf: <a href="https://t.co/uHtoSUIsLk">https://t.co/uHtoSUIsLk</a><br>abs: <a href="https://t.co/oktiTnwuOl">https://t.co/oktiTnwuOl</a><br>project page: <a href="https://t.co/2L9ZFr7zRI">https://t.co/2L9ZFr7zRI</a> <a href="https://t.co/aZs0VgWhLi">pic.twitter.com/aZs0VgWhLi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399219089860612096?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Changing the World by Changing the Data

Anna Rogers

- retweets: 2156, favorites: 223 (06/01/2021 07:56:24)

- links: [abs](https://arxiv.org/abs/2105.13947) | [pdf](https://arxiv.org/pdf/2105.13947)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

NLP community is currently investing a lot more research and resources into development of deep learning models than training data. While we have made a lot of progress, it is now clear that our models learn all kinds of spurious patterns, social biases, and annotation artifacts. Algorithmic solutions have so far had limited success. An alternative that is being actively discussed is more careful design of datasets so as to deliver specific signals. This position paper maps out the arguments for and against data curation, and argues that fundamentally the point is moot: curation already is and will be happening, and it is changing the world. The question is only how much thought we want to invest into that process.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">🎈 <a href="https://twitter.com/hashtag/NLPaperAlert?src=hash&amp;ref_src=twsrc%5Etfw">#NLPaperAlert</a>: Changing the World 🌍  by Changing the Data 🗃<a href="https://t.co/GHDTfcbW5Y">https://t.co/GHDTfcbW5Y</a><br>A soul-searching piece that made it to ACL 2021:<br>- how NLP resources affect the world<br>- what does it even mean to &#39;work in NLP&#39;<br>- how we can make better use of our subcommunities.<br>/1</p>&mdash; Anna Rogers (@annargrs) <a href="https://twitter.com/annargrs/status/1399290146495860739?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. ResT: An Efficient Transformer for Visual Recognition

Qinglong Zhang, Yubin Yang

- retweets: 575, favorites: 106 (06/01/2021 07:56:24)

- links: [abs](https://arxiv.org/abs/2105.13677) | [pdf](https://arxiv.org/pdf/2105.13677)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents an efficient multi-scale vision Transformer, called ResT, that capably served as a general-purpose backbone for image recognition. Unlike existing Transformer methods, which employ standard Transformer blocks to tackle raw images with a fixed resolution, our ResT have several advantages: (1) A memory-efficient multi-head self-attention is built, which compresses the memory by a simple depth-wise convolution, and projects the interaction across the attention-heads dimension while keeping the diversity ability of multi-heads; (2) Position encoding is constructed as spatial attention, which is more flexible and can tackle with input images of arbitrary size without interpolation or fine-tune; (3) Instead of the straightforward tokenization at the beginning of each stage, we design the patch embedding as a stack of overlapping convolution operation with stride on the 2D-reshaped token map. We comprehensively validate ResT on image classification and downstream tasks. Experimental results show that the proposed ResT can outperform the recently state-of-the-art backbones by a large margin, demonstrating the potential of ResT as strong backbones. The code and models will be made publicly available at https://github.com/wofmanaf/ResT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ResT: An Efficient Transformer for Visual Recognition<br>pdf: <a href="https://t.co/buYVDxY7wb">https://t.co/buYVDxY7wb</a><br>abs: <a href="https://t.co/MOPWQd5BA2">https://t.co/MOPWQd5BA2</a><br>github: <a href="https://t.co/pwLeXbcpSE">https://t.co/pwLeXbcpSE</a><br><br>multi-scale Transformer which produces hierarchical feature representations for dense prediction <a href="https://t.co/U10h0Cnhje">pic.twitter.com/U10h0Cnhje</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399214507969363975?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. What Is Considered Complete for Visual Recognition?

Lingxi Xie, Xiaopeng Zhang, Longhui Wei, Jianlong Chang, Qi Tian

- retweets: 402, favorites: 73 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13978) | [pdf](https://arxiv.org/pdf/2105.13978)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This is an opinion paper. We hope to deliver a key message that current visual recognition systems are far from complete, i.e., recognizing everything that human can recognize, yet it is very unlikely that the gap can be bridged by continuously increasing human annotations. Based on the observation, we advocate for a new type of pre-training task named learning-by-compression. The computational models (e.g., a deep network) are optimized to represent the visual data using compact features, and the features preserve the ability to recover the original data. Semantic annotations, when available, play the role of weak supervision. An important yet challenging issue is the evaluation of image recovery, where we suggest some design principles and future research directions. We hope our proposal can inspire the community to pursue the compression-recovery tradeoff rather than the accuracy-complexity tradeoff.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What Is Considered Complete for Visual Recognition?<br><br>Describes the limitations of the current visual recognition systems and suggest some future research directions.<a href="https://t.co/mcgSXSJSnf">https://t.co/mcgSXSJSnf</a> <a href="https://t.co/qO3Ajn7AzE">pic.twitter.com/qO3Ajn7AzE</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1399165854676230146?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion

Songxiang Liu, Yuewen Cao, Dan Su, Helen Meng

- retweets: 130, favorites: 85 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13871) | [pdf](https://arxiv.org/pdf/2105.13871)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Singing voice conversion (SVC) is one promising technique which can enrich the way of human-computer interaction by endowing a computer the ability to produce high-fidelity and expressive singing voice. In this paper, we propose DiffSVC, an SVC system based on denoising diffusion probabilistic model. DiffSVC uses phonetic posteriorgrams (PPGs) as content features. A denoising module is trained in DiffSVC, which takes destroyed mel spectrogram produced by the diffusion/forward process and its corresponding step information as input to predict the added Gaussian noise. We use PPGs, fundamental frequency features and loudness features as auxiliary input to assist the denoising process. Experiments show that DiffSVC can achieve superior conversion performance in terms of naturalness and voice similarity to current state-of-the-art SVC approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion<br>pdf: <a href="https://t.co/945wanF1up">https://t.co/945wanF1up</a><br>abs: <a href="https://t.co/299tKHAzzV">https://t.co/299tKHAzzV</a> <a href="https://t.co/RVQb656dK0">pic.twitter.com/RVQb656dK0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399229121998249985?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Mapping urban socioeconomic inequalities in developing countries through  Facebook advertising data

Serena Giurgola, Simone Piaggesi, Márton Karsai, Yelena Mejova, André Panisson, Michele Tizzoni

- retweets: 156, favorites: 54 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13774) | [pdf](https://arxiv.org/pdf/2105.13774)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Ending poverty in all its forms everywhere is the number one Sustainable Development Goal of the UN 2030 Agenda. To monitor the progress towards such an ambitious target, reliable, up-to-date and fine-grained measurements of socioeconomic indicators are necessary. When it comes to socioeconomic development, novel digital traces can provide a complementary data source to overcome the limits of traditional data collection methods, which are often not regularly updated and lack adequate spatial resolution. In this study, we collect publicly available and anonymous advertising audience estimates from Facebook to predict socioeconomic conditions of urban residents, at a fine spatial granularity, in four large urban areas: Atlanta (USA), Bogot\'a (Colombia), Santiago (Chile), and Casablanca (Morocco). We find that behavioral attributes inferred from the Facebook marketing platform can accurately map the socioeconomic status of residential areas within cities, and that predictive performance is comparable in both high and low-resource settings. We also show that training a model on attributes of adult Facebook users, aged more than 25, leads to a more accurate mapping of socioeconomic conditions in all cities. Our work provides additional evidence of the value of social advertising media data to measure human development.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New pre-print is out on &quot;Mapping urban socioeconomic inequalities in developing countries through Facebook advertising data&quot;. An <a href="https://twitter.com/ISI_Fondazione?ref_src=twsrc%5Etfw">@ISI_Fondazione</a>  teamwork with <a href="https://twitter.com/GiurgolaSerena?ref_src=twsrc%5Etfw">@GiurgolaSerena</a> <a href="https://twitter.com/simonepiaggesi?ref_src=twsrc%5Etfw">@simonepiaggesi</a> <a href="https://twitter.com/yelenamejova?ref_src=twsrc%5Etfw">@yelenamejova</a> <a href="https://twitter.com/apanisson?ref_src=twsrc%5Etfw">@apanisson</a> and <a href="https://twitter.com/mtizzoni?ref_src=twsrc%5Etfw">@mtizzoni</a> <a href="https://t.co/v4P71oMxbY">https://t.co/v4P71oMxbY</a> <a href="https://t.co/fEsHV6rYVI">pic.twitter.com/fEsHV6rYVI</a></p>&mdash; Marton Karsai (@MartonKarsai) <a href="https://twitter.com/MartonKarsai/status/1399343194781986820?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Learning to Stylize Novel Views

Hsin-Ping Huang, Hung-Yu Tseng, Saurabh Saini, Maneesh Singh, Ming-Hsuan Yang

- retweets: 102, favorites: 79 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13509) | [pdf](https://arxiv.org/pdf/2105.13509)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We tackle a 3D scene stylization problem - generating stylized images of a scene from arbitrary novel views given a set of images of the same scene and a reference image of the desired style as inputs. Direct solution of combining novel view synthesis and stylization approaches lead to results that are blurry or not consistent across different views. We propose a point cloud-based method for consistent 3D scene stylization. First, we construct the point cloud by back-projecting the image features to the 3D space. Second, we develop point cloud aggregation modules to gather the style information of the 3D scene, and then modulate the features in the point cloud with a linear transformation matrix. Finally, we project the transformed features to 2D space to obtain the novel views. Experimental results on two diverse datasets of real-world scenes validate that our method generates consistent stylized novel view synthesis results against other alternative approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning to Stylize Novel Views<br>pdf: <a href="https://t.co/sEtbrpMlV7">https://t.co/sEtbrpMlV7</a><br>abs: <a href="https://t.co/99z2DjbN1r">https://t.co/99z2DjbN1r</a><br>project page: <a href="https://t.co/9wbPWgaUsU">https://t.co/9wbPWgaUsU</a><br><br>design a point cloud transformation module to<br>transfer the style of the reference image to the 3D representation <a href="https://t.co/2nn9ZCN3qu">pic.twitter.com/2nn9ZCN3qu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399210896103256064?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. "Why Would I Trust Your Numbers?" On the Explainability of Expected  Values in Soccer

Jan Van Haaren

- retweets: 64, favorites: 104 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13778) | [pdf](https://arxiv.org/pdf/2105.13778)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.AP](https://arxiv.org/list/stat.AP/recent)

In recent years, many different approaches have been proposed to quantify the performances of soccer players. Since player performances are challenging to quantify directly due to the low-scoring nature of soccer, most approaches estimate the expected impact of the players' on-the-ball actions on the scoreline. While effective, these approaches are yet to be widely embraced by soccer practitioners. The soccer analytics community has primarily focused on improving the accuracy of the models, while the explainability of the produced metrics is often much more important to practitioners.   To help bridge the gap between scientists and practitioners, we introduce an explainable Generalized Additive Model that estimates the expected value for shots. Unlike existing models, our model leverages features corresponding to widespread soccer concepts. To this end, we represent the locations of shots by fuzzily assigning the shots to designated zones on the pitch that practitioners are familiar with. Our experimental evaluation shows that our model is as accurate as existing models, while being easier to explain to soccer practitioners.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The limited explainability of expected value metrics for football is holding back their adoption by practitioners. Therefore, I am exploring ways to improve their explainability in a paper that I will be presenting at the AI for Sports Analytics workshop.<a href="https://t.co/idIo3ubr0D">https://t.co/idIo3ubr0D</a></p>&mdash; Jan Van Haaren (@JanVanHaaren) <a href="https://twitter.com/JanVanHaaren/status/1399350176666558468?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. OTTers: One-turn Topic Transitions for Open-Domain Dialogue

Karin Sevegnani, David M. Howcroft, Ioannis Konstas, Verena Rieser

- retweets: 102, favorites: 47 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13710) | [pdf](https://arxiv.org/pdf/2105.13710)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Mixed initiative in open-domain dialogue requires a system to pro-actively introduce new topics. The one-turn topic transition task explores how a system connects two topics in a cooperative and coherent manner. The goal of the task is to generate a "bridging" utterance connecting the new topic to the topic of the previous conversation turn. We are especially interested in commonsense explanations of how a new topic relates to what has been mentioned before. We first collect a new dataset of human one-turn topic transitions, which we call OTTers. We then explore different strategies used by humans when asked to complete such a task, and notice that the use of a bridging utterance to connect the two topics is the approach used the most. We finally show how existing state-of-the-art text generation models can be adapted to this task and examine the performance of these baselines on different splits of the OTTers data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Most open-domain <a href="https://twitter.com/hashtag/ConvAI?src=hash&amp;ref_src=twsrc%5Etfw">#ConvAI</a> systems are purely reactive. How can a system introduce new topics without sounding abrupt or incoherent? Check-out our new paper to appear <a href="https://twitter.com/aclmeeting?ref_src=twsrc%5Etfw">@aclmeeting</a> with <a href="https://twitter.com/KarinSevegnani?ref_src=twsrc%5Etfw">@KarinSevegnani</a> <a href="https://twitter.com/sinantie?ref_src=twsrc%5Etfw">@sinantie</a> <a href="https://twitter.com/_dmh?ref_src=twsrc%5Etfw">@_dmh</a>  <a href="https://t.co/dmqLEV2Rcn">https://t.co/dmqLEV2Rcn</a></p>&mdash; Verena Rieser (@verena_rieser) <a href="https://twitter.com/verena_rieser/status/1399318979244707842?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. NViSII: A Scriptable Tool for Photorealistic Image Generation

Nathan Morrical, Jonathan Tremblay, Yunzhi Lin, Stephen Tyree, Stan Birchfield, Valerio Pascucci, Ingo Wald

- retweets: 58, favorites: 84 (06/01/2021 07:56:25)

- links: [abs](https://arxiv.org/abs/2105.13962) | [pdf](https://arxiv.org/pdf/2105.13962)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We present a Python-based renderer built on NVIDIA's OptiX ray tracing engine and the OptiX AI denoiser, designed to generate high-quality synthetic images for research in computer vision and deep learning. Our tool enables the description and manipulation of complex dynamic 3D scenes containing object meshes, materials, textures, lighting, volumetric data (e.g., smoke), and backgrounds. Metadata, such as 2D/3D bounding boxes, segmentation masks, depth maps, normal maps, material properties, and optical flow vectors, can also be generated. In this work, we discuss design goals, architecture, and performance. We demonstrate the use of data generated by path tracing for training an object detector and pose estimator, showing improved performance in sim-to-real transfer in situations that are difficult for traditional raster-based renderers. We offer this tool as an easy-to-use, performant, high-quality renderer for advancing research in synthetic data generation and deep learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NViSII: A Scriptable Tool for Photorealistic Image Generation<a href="https://t.co/j6FQLkPzML">https://t.co/j6FQLkPzML</a> <a href="https://t.co/zfiwzYxYZG">pic.twitter.com/zfiwzYxYZG</a></p>&mdash; sim2real (@sim2realAIorg) <a href="https://twitter.com/sim2realAIorg/status/1399169186555781123?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NViSII: A Scriptable Tool for Photorealistic Image Generation<br>pdf: <a href="https://t.co/yPJavd1Muw">https://t.co/yPJavd1Muw</a><br>abs: <a href="https://t.co/wnE4mIpUlf">https://t.co/wnE4mIpUlf</a><br>github: <a href="https://t.co/HkiJQM4dFh">https://t.co/HkiJQM4dFh</a> <a href="https://t.co/81kaGkgrHA">pic.twitter.com/81kaGkgrHA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399217168638676993?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. ResearchGate and Google Scholar: How much do they differ in  publications, citations and different metrics and why?

Vivek Kumar Singh, Satya Swarup Srichandan, Hiran H. Lathabai

- retweets: 102, favorites: 28 (06/01/2021 07:56:26)

- links: [abs](https://arxiv.org/abs/2105.13602) | [pdf](https://arxiv.org/pdf/2105.13602)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

ResearchGate has emerged as a popular professional network for scientists and researchers in a very short span of time. Similar to Google Scholar, the ResearchGate indexing uses an automatic crawling algorithm that extracts bibliographic data, citations and other information about scholarly articles from various sources. However, it has been observed that the two platforms often show different publication and citation data for the same institutions, journals and authors. This paper, therefore, attempts to analyse and measure the differences in publication counts, citations and different metrics of the two platforms for a large data set of highly cited authors. The results indicate that there are significantly high differences in publication counts and citations for the same authors in the two platforms, with Google Scholar having higher counts for a vast majority of the cases. The different metrics computed by the two platforms also differ in their values, showing different degrees of correlations. The coverage policy, indexing errors, author attribution mechanism and strategy to deal with predatory publishing are found to be the main probable reasons for the differences in the two platforms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ResearchGate and Google Scholar: How much do they differ in publications, citations and different metrics and why? Preprint now in arxiv: <a href="https://t.co/sWPZ7FajyI">https://t.co/sWPZ7FajyI</a><a href="https://twitter.com/googlescholar_?ref_src=twsrc%5Etfw">@googlescholar_</a> <a href="https://twitter.com/ResearchGate?ref_src=twsrc%5Etfw">@ResearchGate</a> <a href="https://twitter.com/mikethelwall?ref_src=twsrc%5Etfw">@mikethelwall</a> <a href="https://twitter.com/JLOrtegaPriego?ref_src=twsrc%5Etfw">@JLOrtegaPriego</a> <a href="https://twitter.com/eomalea?ref_src=twsrc%5Etfw">@eomalea</a> <a href="https://twitter.com/albertomartin?ref_src=twsrc%5Etfw">@albertomartin</a> <a href="https://twitter.com/HIRAN31021775?ref_src=twsrc%5Etfw">@HIRAN31021775</a> <a href="https://twitter.com/satyaswarup98?ref_src=twsrc%5Etfw">@satyaswarup98</a></p>&mdash; Vivek Singh (@vivekks12) <a href="https://twitter.com/vivekks12/status/1399215643405131778?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Knowledge Inheritance for Pre-trained Language Models

Yujia Qin, Yankai Lin, Jing Yi, Jiajie Zhang, Xu Han, Zhengyan Zhang, Yusheng Su, Zhiyuan Liu, Peng Li, Maosong Sun, Jie Zhou

- retweets: 30, favorites: 35 (06/01/2021 07:56:26)

- links: [abs](https://arxiv.org/abs/2105.13880) | [pdf](https://arxiv.org/pdf/2105.13880)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent explorations of large-scale pre-trained language models (PLMs) such as GPT-3 have revealed the power of PLMs with huge amounts of parameters, setting off a wave of training ever-larger PLMs. However, training a large-scale PLM requires tremendous amounts of computational resources, which is time-consuming and expensive. In addition, existing large-scale PLMs are mainly trained from scratch individually, ignoring the availability of many existing well-trained PLMs. To this end, we explore the question that how can previously trained PLMs benefit training larger PLMs in future. Specifically, we introduce a novel pre-training framework named "knowledge inheritance" (KI), which combines both self-learning and teacher-guided learning to efficiently train larger PLMs. Sufficient experimental results demonstrate the feasibility of our KI framework. We also conduct empirical analyses to explore the effects of teacher PLMs' pre-training settings, including model architecture, pre-training data, etc. Finally, we show that KI can well support lifelong learning and knowledge transfer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Knowledge Inheritance for Pre-trained Language Models<br>pdf: <a href="https://t.co/a1b1PcrjdD">https://t.co/a1b1PcrjdD</a><br>abs: <a href="https://t.co/Mog8wrnqJF">https://t.co/Mog8wrnqJF</a><br>github: <a href="https://t.co/eqHVMMrw0K">https://t.co/eqHVMMrw0K</a><br><br>pre-training framework, knowledge inheritance, combines both self-learning and teacher-guided learning to efficiently train larger PLMs <a href="https://t.co/RVlodUWKjv">pic.twitter.com/RVlodUWKjv</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1399173015758426120?ref_src=twsrc%5Etfw">May 31, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



