---
title: Hot Papers 2021-02-18
date: 2021-02-19T11:14:00.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Contrastive Learning Inverts the Data Generating Process

Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel

- retweets: 3135, favorites: 351 (02/19/2021 11:14:00)

- links: [abs](https://arxiv.org/abs/2102.08850) | [pdf](https://arxiv.org/pdf/2102.08850)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Contrastive learning has recently seen tremendous success in self-supervised learning. So far, however, it is largely unclear why the learned representations generalize so effectively to a large variety of downstream tasks. We here prove that feedforward models trained with objectives belonging to the commonly used InfoNCE family learn to implicitly invert the underlying generative model of the observed data. While the proofs make certain statistical assumptions about the generative model, we observe empirically that our findings hold even if these assumptions are severely violated. Our theory highlights a fundamental connection between contrastive learning, generative modeling, and nonlinear independent component analysis, thereby furthering our understanding of the learned representations as well as providing a theoretical foundation to derive more effective contrastive losses.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Why is contrastive representation learning so useful? Our latest work shows that contrastive learning can invert the data generating process, and it paves the way towards more effective contrastive losses.<br>Paper: <a href="https://t.co/uhVkSTDuP7">https://t.co/uhVkSTDuP7</a><br>Website/Code: <a href="https://t.co/lzaLdT7n6t">https://t.co/lzaLdT7n6t</a><br>[1/5] <a href="https://t.co/MZuxpwen27">pic.twitter.com/MZuxpwen27</a></p>&mdash; Wieland Brendel (@wielandbr) <a href="https://twitter.com/wielandbr/status/1362436472142446596?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. ShaRF: Shape-conditioned Radiance Fields from a Single View

Konstantinos Rematas, Ricardo Martin-Brualla, Vittorio Ferrari

- retweets: 788, favorites: 200 (02/19/2021 11:14:00)

- links: [abs](https://arxiv.org/abs/2102.08860) | [pdf](https://arxiv.org/pdf/2102.08860)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We present a method for estimating neural scenes representations of objects given only a single image. The core of our method is the estimation of a geometric scaffold for the object and its use as a guide for the reconstruction of the underlying radiance field. Our formulation is based on a generative process that first maps a latent code to a voxelized shape, and then renders it to an image, with the object appearance being controlled by a second latent code. During inference, we optimize both the latent codes and the networks to fit a test image of a new object. The explicit disentanglement of shape and appearance allows our model to be fine-tuned given a single image. We can then render new views in a geometrically consistent manner and they represent faithfully the input object. Additionally, our method is able to generalize to images outside of the training domain (more realistic renderings and even real photographs). Finally, the inferred geometric scaffold is itself an accurate estimate of the object's 3D shape. We demonstrate in several experiments the effectiveness of our approach in both synthetic and real images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ShaRF: Shape-conditioned Radiance Fields from a Single View<br>pdf: <a href="https://t.co/IHYBOSXNn1">https://t.co/IHYBOSXNn1</a><br>abs: <a href="https://t.co/TzHHoHRhUs">https://t.co/TzHHoHRhUs</a><br>project page: <a href="https://t.co/VXqCcUg5J3">https://t.co/VXqCcUg5J3</a> <a href="https://t.co/4X4zyUgaVr">pic.twitter.com/4X4zyUgaVr</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1362227354811129861?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Centroid Transformers: Learning to Abstract with Attention

Lemeng Wu, Xingchao Liu, Qiang Liu

- retweets: 504, favorites: 105 (02/19/2021 11:14:00)

- links: [abs](https://arxiv.org/abs/2102.08606) | [pdf](https://arxiv.org/pdf/2102.08606)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Self-attention, as the key block of transformers, is a powerful mechanism for extracting features from the inputs. In essence, what self-attention does to infer the pairwise relations between the elements of the inputs, and modify the inputs by propagating information between input pairs. As a result, it maps inputs to N outputs and casts a quadratic $O(N^2)$ memory and time complexity. We propose centroid attention, a generalization of self-attention that maps N inputs to M outputs $(M\leq N)$, such that the key information in the inputs are summarized in the smaller number of outputs (called centroids). We design centroid attention by amortizing the gradient descent update rule of a clustering objective function on the inputs, which reveals an underlying connection between attention and clustering. By compressing the inputs to the centroids, we extract the key information useful for prediction and also reduce the computation of the attention module and the subsequent layers. We apply our method to various applications, including abstractive text summarization, 3D vision, and image processing. Empirical results demonstrate the effectiveness of our method over the standard transformers.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Centroid Transformer: Learning to Abstract with Attention<br>pdf: <a href="https://t.co/jafxf0DJUq">https://t.co/jafxf0DJUq</a><br>abs: <a href="https://t.co/rO4bGFYw6N">https://t.co/rO4bGFYw6N</a> <a href="https://t.co/rnPLQnx9ZH">pic.twitter.com/rnPLQnx9ZH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1362221635571433481?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Surveying the Landscape of Ethics-Focused Design Methods

Shruthi Sai Chivukula, Ziqing Li, Anne C. Pivonka, Jingning Chen, Colin M. Gray

- retweets: 182, favorites: 57 (02/19/2021 11:14:00)

- links: [abs](https://arxiv.org/abs/2102.08909) | [pdf](https://arxiv.org/pdf/2102.08909)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Over the past decade, HCI researchers, design researchers, and practitioners have increasingly addressed ethics-focused issues through a range of theoretical, methodological and pragmatic contributions to the field. While many forms of design knowledge have been proposed and described, we focus explicitly on knowledge that has been codified as "methods," which we define as any supports for everyday work practices of designers. In this paper, we identify, analyze, and map a collection of 63 existing ethics-focused methods intentionally designed for ethical impact. We present a content analysis, providing a descriptive record of how they operationalize ethics, their intended audience or context of use, their "core" or "script," and the means by which these methods are formulated, articulated, and languaged. Building on these results, we provide an initial definition of ethics-focused methods, identifying potential opportunities for the development of future methods to support design practice and research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are excited to release a preprint: “Surveying the Landscape of Ethics-Focused Design Methods.” In this paper, we identify, analyze, and map 63 existing design methods that support ethical awareness and impact. <a href="https://t.co/h8LDqxE1I7">https://t.co/h8LDqxE1I7</a> 1/4</p>&mdash; colin gray (@graydesign) <a href="https://twitter.com/graydesign/status/1362461448316858371?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Weakly Supervised Learning of Rigid 3D Scene Flow

Zan Gojcic, Or Litany, Andreas Wieser, Leonidas J. Guibas, Tolga Birdal

- retweets: 81, favorites: 67 (02/19/2021 11:14:00)

- links: [abs](https://arxiv.org/abs/2102.08945) | [pdf](https://arxiv.org/pdf/2102.08945)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

We propose a data-driven scene flow estimation algorithm exploiting the observation that many 3D scenes can be explained by a collection of agents moving as rigid bodies. At the core of our method lies a deep architecture able to reason at the \textbf{object-level} by considering 3D scene flow in conjunction with other 3D tasks. This object level abstraction, enables us to relax the requirement for dense scene flow supervision with simpler binary background segmentation mask and ego-motion annotations. Our mild supervision requirements make our method well suited for recently released massive data collections for autonomous driving, which do not contain dense scene flow annotations. As output, our model provides low-level cues like pointwise flow and higher-level cues such as holistic scene understanding at the level of rigid objects. We further propose a test-time optimization refining the predicted rigid scene flow. We showcase the effectiveness and generalization capacity of our method on four different autonomous driving datasets. We release our source code and pre-trained models under \url{github.com/zgojcic/Rigid3DSceneFlow}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Weakly Supervised Learning of Rigid 3D Scene Flow <a href="https://t.co/QgW5GPZGpV">https://t.co/QgW5GPZGpV</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://twitter.com/hashtag/3D?src=hash&amp;ref_src=twsrc%5Etfw">#3D</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <br><br>GitHub: <a href="https://t.co/3BrJ89S2lQ">https://t.co/3BrJ89S2lQ</a> <a href="https://t.co/tl8vMLVkL6">pic.twitter.com/tl8vMLVkL6</a></p>&mdash; Tomasz Malisiewicz (@quantombone) <a href="https://twitter.com/quantombone/status/1362271249469239296?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Weakly Supervised Learning of Rigid 3D Scene Flow<br>pdf: <a href="https://t.co/TixQZ8uW9Q">https://t.co/TixQZ8uW9Q</a><br>abs: <a href="https://t.co/w9XDTphfYu">https://t.co/w9XDTphfYu</a><br>project page: <a href="https://t.co/rgScLS0Pgx">https://t.co/rgScLS0Pgx</a><br>github: <a href="https://t.co/iekVWvyrGj">https://t.co/iekVWvyrGj</a> <a href="https://t.co/8efV5zsGGB">pic.twitter.com/8efV5zsGGB</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1362264425932206080?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. CheXternal: Generalization of Deep Learning Models for Chest X-ray  Interpretation to Photos of Chest X-rays and External Clinical Settings

Pranav Rajpurkar, Anirudh Joshi, Anuj Pareek, Andrew Y. Ng, Matthew P. Lungren

- retweets: 57, favorites: 63 (02/19/2021 11:14:01)

- links: [abs](https://arxiv.org/abs/2102.08660) | [pdf](https://arxiv.org/pdf/2102.08660)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent advances in training deep learning models have demonstrated the potential to provide accurate chest X-ray interpretation and increase access to radiology expertise. However, poor generalization due to data distribution shifts in clinical settings is a key barrier to implementation. In this study, we measured the diagnostic performance for 8 different chest X-ray models when applied to (1) smartphone photos of chest X-rays and (2) external datasets without any finetuning. All models were developed by different groups and submitted to the CheXpert challenge, and re-applied to test datasets without further tuning. We found that (1) on photos of chest X-rays, all 8 models experienced a statistically significant drop in task performance, but only 3 performed significantly worse than radiologists on average, and (2) on the external set, none of the models performed statistically significantly worse than radiologists, and five models performed statistically significantly better than radiologists. Our results demonstrate that some chest X-ray models, under clinically relevant distribution shifts, were comparable to radiologists while other models were not. Future work should investigate aspects of model training procedures and dataset collection that influence generalization in the presence of data distribution shifts.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can medical imaging models maintain radiologist performance when tested on external clinical settings without fine-tuning?<br><br>Surprisingly, the answer is YES!<br><br>Paper: <a href="https://t.co/aIg9ZlkIGJ">https://t.co/aIg9ZlkIGJ</a><a href="https://twitter.com/anirrjoshi?ref_src=twsrc%5Etfw">@anirrjoshi</a> <a href="https://twitter.com/anujpareek?ref_src=twsrc%5Etfw">@anujpareek</a> <a href="https://twitter.com/AndrewYNg?ref_src=twsrc%5Etfw">@AndrewYNg</a> <a href="https://twitter.com/mattlungrenMD?ref_src=twsrc%5Etfw">@mattlungrenMD</a> <a href="https://twitter.com/StanfordAILab?ref_src=twsrc%5Etfw">@StanfordAILab</a> <br><br>1/6 <a href="https://t.co/SXMIOTf5fz">pic.twitter.com/SXMIOTf5fz</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1362365937412759553?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Galaxy Zoo DECaLS: Detailed Visual Morphology Measurements from  Volunteers and Deep Learning for 314,000 Galaxies

Mike Walmsley, Chris Lintott, Tobias Geron, Sandor Kruk, Coleman Krawczyk, Kyle W. Willett, Steven Bamford, William Keel, Lee S. Kelvin, Lucy Fortson, Karen L. Masters, Vihang Mehta, Brooke D. Simmons, Rebecca Smethurst, Elisabeth M. Baeten, Christine Macmillan

- retweets: 62, favorites: 53 (02/19/2021 11:14:01)

- links: [abs](https://arxiv.org/abs/2102.08414) | [pdf](https://arxiv.org/pdf/2102.08414)
- [astro-ph.GA](https://arxiv.org/list/astro-ph.GA/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present Galaxy Zoo DECaLS: detailed visual morphological classifications for Dark Energy Camera Legacy Survey images of galaxies within the SDSS DR8 footprint. Deeper DECaLS images (r=23.6 vs. r=22.2 from SDSS) reveal spiral arms, weak bars, and tidal features not previously visible in SDSS imaging. To best exploit the greater depth of DECaLS images, volunteers select from a new set of answers designed to improve our sensitivity to mergers and bars. Galaxy Zoo volunteers provide 7.5 million individual classifications over 314,000 galaxies. 140,000 galaxies receive at least 30 classifications, sufficient to accurately measure detailed morphology like bars, and the remainder receive approximately 5. All classifications are used to train an ensemble of Bayesian convolutional neural networks (a state-of-the-art deep learning method) to predict posteriors for the detailed morphology of all 314,000 galaxies. When measured against confident volunteer classifications, the networks are approximately 99% accurate on every question. Morphology is a fundamental feature of every galaxy; our human and machine classifications are an accurate and detailed resource for understanding how galaxies evolve.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Take a look at some of the galaxy classifications done by volunteers on <a href="https://twitter.com/galaxyzoo?ref_src=twsrc%5Etfw">@galaxyzoo</a> on the DECaLS imaging! Mike has put together a fun interface for you to explore them yourself <br><br>All from our new research paper published today: <a href="https://t.co/UPJ8Ewrijy">https://t.co/UPJ8Ewrijy</a> <a href="https://t.co/LiN5QBmdxO">https://t.co/LiN5QBmdxO</a></p>&mdash; Dr Becky Smethurst (@drbecky_) <a href="https://twitter.com/drbecky_/status/1362358042482851844?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Rethinking Co-design of Neural Architectures and Hardware Accelerators

Yanqi Zhou, Xuanyi Dong, Berkin Akin, Mingxing Tan, Daiyi Peng, Tianjian Meng, Amir Yazdanbakhsh, Da Huang, Ravi Narayanaswami, James Laudon

- retweets: 58, favorites: 48 (02/19/2021 11:14:01)

- links: [abs](https://arxiv.org/abs/2102.08619) | [pdf](https://arxiv.org/pdf/2102.08619)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AR](https://arxiv.org/list/cs.AR/recent)

Neural architectures and hardware accelerators have been two driving forces for the progress in deep learning. Previous works typically attempt to optimize hardware given a fixed model architecture or model architecture given fixed hardware. And the dominant hardware architecture explored in this prior work is FPGAs. In our work, we target the optimization of hardware and software configurations on an industry-standard edge accelerator. We systematically study the importance and strategies of co-designing neural architectures and hardware accelerators. We make three observations: 1) the software search space has to be customized to fully leverage the targeted hardware architecture, 2) the search for the model architecture and hardware architecture should be done jointly to achieve the best of both worlds, and 3) different use cases lead to very different search outcomes. Our experiments show that the joint search method consistently outperforms previous platform-aware neural architecture search, manually crafted models, and the state-of-the-art EfficientNet on all latency targets by around 1% on ImageNet top-1 accuracy. Our method can reduce energy consumption of an edge accelerator by up to 2x under the same accuracy constraint, when co-adapting the model architecture and hardware accelerator configurations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking Co-design of Neural Architectures and Hardware Accelerators<br><br>Reduces energy consumption of Edge TPU by up to 2x over EfficientNet w/ the same accuracy on Imagenet by jointly searching over hardware &amp; neural architecture design spaces.<a href="https://t.co/jh7hIgA9Fh">https://t.co/jh7hIgA9Fh</a> <a href="https://t.co/lycQk5ihvV">pic.twitter.com/lycQk5ihvV</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1362270964646514689?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Globally-Robust Neural Networks

Klas Leino, Zifan Wang, Matt Fredrikson

- retweets: 64, favorites: 8 (02/19/2021 11:14:01)

- links: [abs](https://arxiv.org/abs/2102.08452) | [pdf](https://arxiv.org/pdf/2102.08452)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CR](https://arxiv.org/list/cs.CR/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The threat of adversarial examples has motivated work on training certifiably robust neural networks, to facilitate efficient verification of local robustness at inference time. We formalize a notion of global robustness, which captures the operational properties of on-line local robustness certification while yielding a natural learning objective for robust training. We show that widely-used architectures can be easily adapted to this objective by incorporating efficient global Lipschitz bounds into the network, yielding certifiably-robust models by construction that achieve state-of-the-art verifiable and clean accuracy. Notably, this approach requires significantly less time and memory than recent certifiable training methods, and leads to negligible costs when certifying points on-line; for example, our evaluation shows that it is possible to train a large tiny-imagenet model in a matter of hours. We posit that this is possible using inexpensive global bounds -- despite prior suggestions that tighter local bounds are needed for good performance -- because these models are trained to achieve tighter global bounds. Namely, we prove that the maximum achievable verifiable accuracy for a given dataset is not improved by using a local bound.




# 10. Crop mapping from image time series: deep learning with multi-scale  label hierarchies

Mehmet Ozgur Turkoglu, Stefano D'Aronco, Gregor Perich, Frank Liebisch, Constantin Streit, Konrad Schindler, Jan Dirk Wegner

- retweets: 34, favorites: 37 (02/19/2021 11:14:01)

- links: [abs](https://arxiv.org/abs/2102.08820) | [pdf](https://arxiv.org/pdf/2102.08820)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The aim of this paper is to map agricultural crops by classifying satellite image time series. Domain experts in agriculture work with crop type labels that are organised in a hierarchical tree structure, where coarse classes (like orchards) are subdivided into finer ones (like apples, pears, vines, etc.). We develop a crop classification method that exploits this expert knowledge and significantly improves the mapping of rare crop types. The three-level label hierarchy is encoded in a convolutional, recurrent neural network (convRNN), such that for each pixel the model predicts three labels at different level of granularity. This end-to-end trainable, hierarchical network architecture allows the model to learn joint feature representations of rare classes (e.g., apples, pears) at a coarser level (e.g., orchard), thereby boosting classification performance at the fine-grained level. Additionally, labelling at different granularity also makes it possible to adjust the output according to the classification scores; as coarser labels with high confidence are sometimes more useful for agricultural practice than fine-grained but very uncertain labels. We validate the proposed method on a new, large dataset that we make public. ZueriCrop covers an area of 50 km x 48 km in the Swiss cantons of Zurich and Thurgau with a total of 116'000 individual fields spanning 48 crop classes, and 28,000 (multi-temporal) image patches from Sentinel-2. We compare our proposed hierarchical convRNN model with several baselines, including methods designed for imbalanced class distributions. The hierarchical approach performs superior by at least 9.9 percentage points in F1-score.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">interested in crop mapping with <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> from <a href="https://twitter.com/hashtag/SatelliteImages?src=hash&amp;ref_src=twsrc%5Etfw">#SatelliteImages</a>? Check our new paper where we exploit expert’s knowledge on label taxonomy to improve the classification of rare crop types.<a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/RNN?src=hash&amp;ref_src=twsrc%5Etfw">#RNN</a> <a href="https://twitter.com/hashtag/TimeSeries?src=hash&amp;ref_src=twsrc%5Etfw">#TimeSeries</a> <a href="https://t.co/ez4U05wfLN">https://t.co/ez4U05wfLN</a> <a href="https://t.co/ZF6AwxhZMK">pic.twitter.com/ZF6AwxhZMK</a></p>&mdash; EcoVision (@EcoVisionETH) <a href="https://twitter.com/EcoVisionETH/status/1362345194063073282?ref_src=twsrc%5Etfw">February 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



