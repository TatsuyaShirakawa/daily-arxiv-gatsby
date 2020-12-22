---
title: Hot Papers 2020-12-21
date: 2020-12-22T13:45:59.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-21"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-21"
socialImage: "/media/flying-marine.jpg"

---

# 1. Toward Transformer-Based Object Detection

Josh Beal, Eric Kim, Eric Tzeng, Dong Huk Park, Andrew Zhai, Dmitry Kislyuk

- retweets: 1140, favorites: 221 (12/22/2020 13:45:59)

- links: [abs](https://arxiv.org/abs/2012.09958) | [pdf](https://arxiv.org/pdf/2012.09958)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers have become the dominant model in natural language processing, owing to their ability to pretrain on massive amounts of data, then transfer to smaller, more specific tasks via fine-tuning. The Vision Transformer was the first major attempt to apply a pure transformer model directly to images as input, demonstrating that as compared to convolutional networks, transformer-based architectures can achieve competitive results on benchmark classification tasks. However, the computational complexity of the attention operator means that we are limited to low-resolution inputs. For more complex tasks such as detection or segmentation, maintaining a high input resolution is crucial to ensure that models can properly identify and reflect fine details in their output. This naturally raises the question of whether or not transformer-based architectures such as the Vision Transformer are capable of performing tasks other than classification. In this paper, we determine that Vision Transformers can be used as a backbone by a common detection task head to produce competitive COCO results. The model that we propose, ViT-FRCNN, demonstrates several known properties associated with transformers, including large pretraining capacity and fast fine-tuning performance. We also investigate improvements over a standard detection backbone, including superior performance on out-of-domain images, better performance on large objects, and a lessened reliance on non-maximum suppression. We view ViT-FRCNN as an important stepping stone toward a pure-transformer solution of complex vision tasks such as object detection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Toward Transformer-Based Object Detection<br>pdf: <a href="https://t.co/DqSS4M1alP">https://t.co/DqSS4M1alP</a><br>abs: <a href="https://t.co/HJzqQXrxMT">https://t.co/HJzqQXrxMT</a> <a href="https://t.co/B8FUqyczl5">pic.twitter.com/B8FUqyczl5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340842611775565825?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Content Masked Loss: Human-Like Brush Stroke Planning in a Reinforcement  Learning Painting Agent

Peter Schaldenbrand, Jean Oh

- retweets: 1064, favorites: 178 (12/22/2020 13:45:59)

- links: [abs](https://arxiv.org/abs/2012.10043) | [pdf](https://arxiv.org/pdf/2012.10043)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The objective of most Reinforcement Learning painting agents is to minimize the loss between a target image and the paint canvas. Human painter artistry emphasizes important features of the target image rather than simply reproducing it (DiPaola 2007). Using adversarial or L2 losses in the RL painting models, although its final output is generally a work of finesse, produces a stroke sequence that is vastly different from that which a human would produce since the model does not have knowledge about the abstract features in the target image. In order to increase the human-like planning of the model without the use of expensive human data, we introduce a new loss function for use with the model's reward function: Content Masked Loss. In the context of robot painting, Content Masked Loss employs an object detection model to extract features which are used to assign higher weight to regions of the canvas that a human would find important for recognizing content. The results, based on 332 human evaluators, show that the digital paintings produced by our Content Masked model show detectable subject matter earlier in the stroke sequence than existing methods without compromising on the quality of the final painting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Content Masked Loss: Human-Like Brush Stroke Planning in a Reinforcement Learning Painting Agent<br>pdf: <a href="https://t.co/2ERb3CSNem">https://t.co/2ERb3CSNem</a><br>abs: <a href="https://t.co/XRGdAjCf97">https://t.co/XRGdAjCf97</a> <a href="https://t.co/GRbenP21v0">pic.twitter.com/GRbenP21v0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340875549623902214?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. YouNiverse: Large-Scale Channel and Video Metadata from English-Speaking  YouTube

Manoel Horta Ribeiro, Robert West

- retweets: 900, favorites: 81 (12/22/2020 13:45:59)

- links: [abs](https://arxiv.org/abs/2012.10378) | [pdf](https://arxiv.org/pdf/2012.10378)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

YouTube plays a key role in entertaining and informing people around the globe. However, studying the platform is difficult due to the lack of randomly sampled data and of systematic ways to query the platform's colossal catalog. In this paper, we present YouNiverse, a large collection of channel and video metadata from English-language YouTube. YouNiverse comprises metadata from over 136k channels and 72.9M videos published between May 2005 and October 2019, as well as channel-level time-series data with weekly subscriber and view counts. Leveraging channel ranks from socialblade.com, an online service that provides information about YouTube, we are able to assess and enhance the representativeness of the sample of channels. YouNiverse, publicly available at https://doi.org/10.5281/zenodo.4327607, will empower the community to do research with and about YouTube.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We (me + <a href="https://twitter.com/cervisiarius?ref_src=twsrc%5Etfw">@cervisiarius</a>) just released the YouNiverse dataset (<a href="https://t.co/9mWAAHTzvO">https://t.co/9mWAAHTzvO</a>)! 🎉🚀<br><br>It contains metadata from over 136k channels and 73M videos published between May 2005 and Oct 2019, as well as channel-level time-series data with weekly subscriber and view counts.<br><br>🧵 <a href="https://t.co/ggBfyJF7dC">pic.twitter.com/ggBfyJF7dC</a></p>&mdash; Manoel (@manoelribeiro) <a href="https://twitter.com/manoelribeiro/status/1341082283223240706?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Deep Learning and the Global Workspace Theory

Rufin VanRullen, Ryota Kanai

- retweets: 600, favorites: 91 (12/22/2020 13:46:00)

- links: [abs](https://arxiv.org/abs/2012.10390) | [pdf](https://arxiv.org/pdf/2012.10390)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent)

Recent advances in deep learning have allowed Artificial Intelligence (AI) to reach near human-level performance in many sensory, perceptual, linguistic or cognitive tasks. There is a growing need, however, for novel, brain-inspired cognitive architectures. The Global Workspace theory refers to a large-scale system integrating and distributing information among networks of specialized modules to create higher-level forms of cognition and awareness. We argue that the time is ripe to consider explicit implementations of this theory using deep learning techniques. We propose a roadmap based on unsupervised neural translation between multiple latent spaces (neural networks trained for distinct tasks, on distinct sensory inputs and/or modalities) to create a unique, amodal global latent workspace (GLW). Potential functional advantages of GLW are reviewed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A new opinion paper on deep learning and global workspace theory with Rufin VanRullen. In this paper, we proposed a  possible way to implement the global workspace by interpreting it as a shared latent space.  <a href="https://t.co/TRbTFe84Su">https://t.co/TRbTFe84Su</a></p>&mdash; Ryota Kanai (@kanair) <a href="https://twitter.com/kanair/status/1340885092160323586?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild  with Pose Annotations

Adel Ahmadyan, Liangkai Zhang, Jianing Wei, Artsiom Ablavatski, Matthias Grundmann

- retweets: 306, favorites: 99 (12/22/2020 13:46:00)

- links: [abs](https://arxiv.org/abs/2012.09988) | [pdf](https://arxiv.org/pdf/2012.09988)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

3D object detection has recently become popular due to many applications in robotics, augmented reality, autonomy, and image retrieval. We introduce the Objectron dataset to advance the state of the art in 3D object detection and foster new research and applications, such as 3D object tracking, view synthesis, and improved 3D shape representation. The dataset contains object-centric short videos with pose annotations for nine categories and includes 4 million annotated images in 14,819 annotated videos. We also propose a new evaluation metric, 3D Intersection over Union, for 3D object detection. We demonstrate the usefulness of our dataset in 3D object detection tasks by providing baseline models trained on this dataset. Our dataset and evaluation source code are available online at http://www.objectron.dev

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild with Pose Annotations<br>pdf: <a href="https://t.co/7LAnuYN7S7">https://t.co/7LAnuYN7S7</a><br>abs: <a href="https://t.co/5aMrj9Pfd3">https://t.co/5aMrj9Pfd3</a><br>github: <a href="https://t.co/YrMyxQEfAG">https://t.co/YrMyxQEfAG</a> <a href="https://t.co/IBTE8LmMRz">pic.twitter.com/IBTE8LmMRz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340850081650667520?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Learning Compositional Radiance Fields of Dynamic Human Heads

Ziyan Wang, Timur Bagautdinov, Stephen Lombardi, Tomas Simon, Jason Saragih, Jessica Hodgins, Michael Zollhöfer

- retweets: 156, favorites: 58 (12/22/2020 13:46:00)

- links: [abs](https://arxiv.org/abs/2012.09955) | [pdf](https://arxiv.org/pdf/2012.09955)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Photorealistic rendering of dynamic humans is an important ability for telepresence systems, virtual shopping, synthetic data generation, and more. Recently, neural rendering methods, which combine techniques from computer graphics and machine learning, have created high-fidelity models of humans and objects. Some of these methods do not produce results with high-enough fidelity for driveable human models (Neural Volumes) whereas others have extremely long rendering times (NeRF). We propose a novel compositional 3D representation that combines the best of previous methods to produce both higher-resolution and faster results. Our representation bridges the gap between discrete and continuous volumetric representations by combining a coarse 3D-structure-aware grid of animation codes with a continuous learned scene function that maps every position and its corresponding local animation code to its view-dependent emitted radiance and local volume density. Differentiable volume rendering is employed to compute photo-realistic novel views of the human head and upper body as well as to train our novel representation end-to-end using only 2D supervision. In addition, we show that the learned dynamic radiance field can be used to synthesize novel unseen expressions based on a global animation code. Our approach achieves state-of-the-art results for synthesizing novel views of dynamic human heads and the upper body.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Compositional Radiance Fields of Dynamic Human Heads<br>pdf: <a href="https://t.co/gKTVy2ZNdj">https://t.co/gKTVy2ZNdj</a><br>abs: <a href="https://t.co/G1HhZTRt15">https://t.co/G1HhZTRt15</a><br>project page: <a href="https://t.co/3rTowtqSsF">https://t.co/3rTowtqSsF</a> <a href="https://t.co/tY464D6mt8">pic.twitter.com/tY464D6mt8</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340845649911279617?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Relightable 3D Head Portraits from a Smartphone Video

Artem Sevastopolsky, Savva Ignatiev, Gonzalo Ferrer, Evgeny Burnaev, Victor Lempitsky

- retweets: 121, favorites: 65 (12/22/2020 13:46:00)

- links: [abs](https://arxiv.org/abs/2012.09963) | [pdf](https://arxiv.org/pdf/2012.09963)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this work, a system for creating a relightable 3D portrait of a human head is presented. Our neural pipeline operates on a sequence of frames captured by a smartphone camera with the flash blinking (flash-no flash sequence). A coarse point cloud reconstructed via structure-from-motion software and multi-view denoising is then used as a geometric proxy. Afterwards, a deep rendering network is trained to regress dense albedo, normals, and environmental lighting maps for arbitrary new viewpoints. Effectively, the proxy geometry and the rendering network constitute a relightable 3D portrait model, that can be synthesized from an arbitrary viewpoint and under arbitrary lighting, e.g. directional light, point light, or an environment map. The model is fitted to the sequence of frames with human face-specific priors that enforce the plausibility of albedo-lighting decomposition and operates at the interactive frame rate. We evaluate the performance of the method under varying lighting conditions and at the extrapolated viewpoints and compare with existing relighting methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Relightable 3D Head Portraits from a Smartphone Video<br>pdf: <a href="https://t.co/XvEiZgiFAH">https://t.co/XvEiZgiFAH</a><br>abs: <a href="https://t.co/7T68gzGqj1">https://t.co/7T68gzGqj1</a> <a href="https://t.co/xgxdVB9K8e">pic.twitter.com/xgxdVB9K8e</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340872663900839937?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Trader-Company Method: A Metaheuristic for Interpretable Stock Price  Prediction

Katsuya Ito, Kentaro Minami, Kentaro Imajo, Kei Nakagawa

- retweets: 112, favorites: 51 (12/22/2020 13:46:00)

- links: [abs](https://arxiv.org/abs/2012.10215) | [pdf](https://arxiv.org/pdf/2012.10215)
- [q-fin.TR](https://arxiv.org/list/q-fin.TR/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Investors try to predict returns of financial assets to make successful investment. Many quantitative analysts have used machine learning-based methods to find unknown profitable market rules from large amounts of market data. However, there are several challenges in financial markets hindering practical applications of machine learning-based models. First, in financial markets, there is no single model that can consistently make accurate prediction because traders in markets quickly adapt to newly available information. Instead, there are a number of ephemeral and partially correct models called "alpha factors". Second, since financial markets are highly uncertain, ensuring interpretability of prediction models is quite important to make reliable trading strategies. To overcome these challenges, we propose the Trader-Company method, a novel evolutionary model that mimics the roles of a financial institute and traders belonging to it. Our method predicts future stock returns by aggregating suggestions from multiple weak learners called Traders. A Trader holds a collection of simple mathematical formulae, each of which represents a candidate of an alpha factor and would be interpretable for real-world investors. The aggregation algorithm, called a Company, maintains multiple Traders. By randomly generating new Traders and retraining them, Companies can efficiently find financially meaningful formulae whilst avoiding overfitting to a transient state of the market. We show the effectiveness of our method by conducting experiments on real market data.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">AAAI、AAMASそれぞれの研究に内容についてすでに論文とPFN社のTechブログでの解説が公開されています。<br>よろしければ読んでみてください！<br>AAAI ArXiv: <a href="https://t.co/oUkZzKwHpr">https://t.co/oUkZzKwHpr</a><br>TechBlog:<a href="https://t.co/FN4OaaQ4HZ">https://t.co/FN4OaaQ4HZ</a><br>AAMAS ArXiv: <a href="https://t.co/HCXMuzso0I">https://t.co/HCXMuzso0I</a><br>TechBlog:<a href="https://t.co/rdPvkbvQlc">https://t.co/rdPvkbvQlc</a></p>&mdash; Kei NAKAGAWA (@keeeeei0315) <a href="https://twitter.com/keeeeei0315/status/1340959013077336064?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Frequency Consistent Adaptation for Real World Super Resolution

Xiaozhong Ji, Guangpin Tao, Yun Cao, Ying Tai, Tong Lu, Chengjie Wang, Jilin Li, Feiyue Huang

- retweets: 81, favorites: 44 (12/22/2020 13:46:01)

- links: [abs](https://arxiv.org/abs/2012.10102) | [pdf](https://arxiv.org/pdf/2012.10102)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent deep-learning based Super-Resolution (SR) methods have achieved remarkable performance on images with known degradation. However, these methods always fail in real-world scene, since the Low-Resolution (LR) images after the ideal degradation (e.g., bicubic down-sampling) deviate from real source domain. The domain gap between the LR images and the real-world images can be observed clearly on frequency density, which inspires us to explictly narrow the undesired gap caused by incorrect degradation. From this point of view, we design a novel Frequency Consistent Adaptation (FCA) that ensures the frequency domain consistency when applying existing SR methods to the real scene. We estimate degradation kernels from unsupervised images and generate the corresponding LR images. To provide useful gradient information for kernel estimation, we propose Frequency Density Comparator (FDC) by distinguishing the frequency density of images on different scales. Based on the domain-consistent LR-HR pairs, we train easy-implemented Convolutional Neural Network (CNN) SR models. Extensive experiments show that the proposed FCA improves the performance of the SR model under real-world setting achieving state-of-the-art results with high fidelity and plausible perception, thus providing a novel effective framework for real-world SR application.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Frequency Consistent Adaptation for Real World Super Resolution<br>pdf: <a href="https://t.co/RQlpRPqSY4">https://t.co/RQlpRPqSY4</a><br>abs: <a href="https://t.co/IizKf3HtRY">https://t.co/IizKf3HtRY</a> <a href="https://t.co/4xhychaoZI">pic.twitter.com/4xhychaoZI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1340876491085758466?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Upper and Lower Bounds on the Performance of Kernel PCA

Maxime Haddouche, Benjamin Guedj, Omar Rivasplata, John Shawe-Taylor

- retweets: 36, favorites: 48 (12/22/2020 13:46:01)

- links: [abs](https://arxiv.org/abs/2012.10369) | [pdf](https://arxiv.org/pdf/2012.10369)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [math.ST](https://arxiv.org/list/math.ST/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Principal Component Analysis (PCA) is a popular method for dimension reduction and has attracted an unfailing interest for decades. Recently, kernel PCA has emerged as an extension of PCA but, despite its use in practice, a sound theoretical understanding of kernel PCA is missing. In this paper, we contribute lower and upper bounds on the efficiency of kernel PCA, involving the empirical eigenvalues of the kernel Gram matrix. Two bounds are for fixed estimators, and two are for randomized estimators through the PAC-Bayes theory. We control how much information is captured by kernel PCA on average, and we dissect the bounds to highlight strengths and limitations of the kernel PCA algorithm. Therefore, we contribute to the better understanding of kernel PCA. Our bounds are briefly illustrated on a toy numerical example.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint, with Maxime Haddouche, <a href="https://twitter.com/OmarRivasplata?ref_src=twsrc%5Etfw">@OmarRivasplata</a>  and John Shawe-Taylor! &quot;Upper and Lower Bounds on the Performance of Kernel PCA&quot; <a href="https://t.co/5FAYM8qU28">https://t.co/5FAYM8qU28</a> tl;dr: we contribute a theoretical analysis of kernel PCA. <a href="https://twitter.com/Inria_Lille?ref_src=twsrc%5Etfw">@Inria_Lille</a> <a href="https://twitter.com/Inria?ref_src=twsrc%5Etfw">@Inria</a> <a href="https://twitter.com/ai_ucl?ref_src=twsrc%5Etfw">@ai_ucl</a> <a href="https://twitter.com/DeepMind?ref_src=twsrc%5Etfw">@DeepMind</a> <a href="https://twitter.com/uclcs?ref_src=twsrc%5Etfw">@uclcs</a> <a href="https://twitter.com/ENS_ParisSaclay?ref_src=twsrc%5Etfw">@ENS_ParisSaclay</a> <a href="https://t.co/0UaVG2Xe8V">pic.twitter.com/0UaVG2Xe8V</a></p>&mdash; Benjamin Guedj (@bguedj) <a href="https://twitter.com/bguedj/status/1340969024201617408?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Data Leverage: A Framework for Empowering the Public in its Relationship  with Technology Companies

Nicholas Vincent, Hanlin Li, Nicole Tilly, Stevie Chancellor, Brent Hecht

- retweets: 30, favorites: 35 (12/22/2020 13:46:01)

- links: [abs](https://arxiv.org/abs/2012.09995) | [pdf](https://arxiv.org/pdf/2012.09995)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

Many powerful computing technologies rely on data contributions from the public. This dependency suggests a potential source of leverage: by reducing, stopping, redirecting, or otherwise manipulating data contributions, people can influence and impact the effectiveness of these technologies. In this paper, we synthesize emerging research that helps people better understand and action this \textit{data leverage}. Drawing on prior work in areas including machine learning, human-computer interaction, and fairness and accountability in computing, we present a framework for understanding data leverage that highlights new opportunities to empower the public in its relationships with technology companies. Our framework also points towards ways that policymakers can augment data leverage as a means of changing the balance of power between the public and tech companies.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Everyone contributes massive amounts of data to powerful computing systems, but only tech companies get a say in how these systems are used and who benefits economically. Our <a href="https://twitter.com/hashtag/FAccT2021?src=hash&amp;ref_src=twsrc%5Etfw">#FAccT2021</a> paper discusses how “data leverage” can give the public more power: <a href="https://t.co/XsnV8fH8hV">https://t.co/XsnV8fH8hV</a></p>&mdash; Nick Vincent (@nickmvincent) <a href="https://twitter.com/nickmvincent/status/1341067591402594304?ref_src=twsrc%5Etfw">December 21, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



