---
title: Hot Papers 2020-12-24
date: 2020-12-25T09:42:51.Z
template: "post"
draft: false
slug: "hot-papers-2020-12-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-12-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. Training data-efficient image transformers & distillation through  attention

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou

- retweets: 1797, favorites: 316 (12/25/2020 09:42:51)

- links: [abs](https://arxiv.org/abs/2012.12877) | [pdf](https://arxiv.org/pdf/2012.12877)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption by the larger community.   In this work, with an adequate training scheme, we produce a competitive convolution-free transformer by training on Imagenet only. We train it on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data. We share our code and models to accelerate community advances on this line of research.   Additionally, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 84.4% accuracy) and when transferring to other tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I&#39;m happy to share our latest work on training competitive Visual Transformers using only ImageNet.<br>Code and pre-trained models: <a href="https://t.co/8COKqmFWVv">https://t.co/8COKqmFWVv</a><br>Paper: <a href="https://t.co/9oYgjAhZz2">https://t.co/9oYgjAhZz2</a><br>Check it out! (1/2) <a href="https://t.co/8VoYLKKcmF">https://t.co/8VoYLKKcmF</a></p>&mdash; Francisco Massa (@fvsmassa) <a href="https://twitter.com/fvsmassa/status/1342072577506889728?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our DeiT transformers achieve convnets performance on ImageNet without extra-training data!<br><br>You can find all our results on the DeiT paper (<a href="https://t.co/L464Zm2uhw">https://t.co/L464Zm2uhw</a>) and also on <a href="https://twitter.com/paperswithcode?ref_src=twsrc%5Etfw">@paperswithcode</a>.<br><br>Our code is available on Github. (<a href="https://t.co/UHgGoldhHG">https://t.co/UHgGoldhHG</a>). <a href="https://t.co/8G2Bt8pXLL">pic.twitter.com/8G2Bt8pXLL</a></p>&mdash; Hugo Touvron (@HugoTouvron) <a href="https://twitter.com/HugoTouvron/status/1342143521801859072?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">画像認識モデルでConvを使わずTransformerのみ使うViTは大量の学習データ必要だったが、CNNの教師NNから蒸留トークンを介して蒸留するDeiTは通常の学習データと学習コストでCNNに匹敵する性能を達成でき通常のEfficientNetの速度/精度トレードオフで上回る。<a href="https://t.co/5p0CQn6ZLF">https://t.co/5p0CQn6ZLF</a></p>&mdash; Daisuke Okanohara (@hillbig) <a href="https://twitter.com/hillbig/status/1342250993464070146?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeiT - Transformer-based image classification model built for high performance and requiring less compute &amp; data. Uses distillation through attention and achieves 84.2 top-1 accuracy on the ImageNet benchmark trained on a single 8-GPU server over 3 days.<a href="https://t.co/NWHvVhzayK">https://t.co/NWHvVhzayK</a> <a href="https://t.co/1PaBn3kfiS">https://t.co/1PaBn3kfiS</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1342085353822420995?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Training data-efficient image transformers &amp; distillation through attention<br>pdf: <a href="https://t.co/52o44tGC5G">https://t.co/52o44tGC5G</a><br>abs: <a href="https://t.co/XohbkS7kq5">https://t.co/XohbkS7kq5</a><br>github: <a href="https://t.co/KpkMSwfKdp">https://t.co/KpkMSwfKdp</a> <a href="https://t.co/KP24f00vT4">pic.twitter.com/KP24f00vT4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341934234198544388?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Focal Frequency Loss for Generative Models

Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy

- retweets: 1366, favorites: 288 (12/25/2020 09:42:53)

- links: [abs](https://arxiv.org/abs/2012.12821) | [pdf](https://arxiv.org/pdf/2012.12821)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Despite the remarkable success of generative models in creating photorealistic images using deep neural networks, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we find that narrowing the frequency domain gap can ameliorate the image synthesis quality further. To this end, we propose the focal frequency loss, a novel objective function that brings optimization of generative models into the frequency domain. The proposed loss allows the model to dynamically focus on the frequency components that are hard to synthesize by down-weighting the easy frequencies. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent crux of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve various baselines in both perceptual quality and quantitative performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Focal Frequency Loss for Generative Models<br>pdf: <a href="https://t.co/tloafhKRch">https://t.co/tloafhKRch</a><br>abs: <a href="https://t.co/uecK5krkjk">https://t.co/uecK5krkjk</a><br>github: <a href="https://t.co/D5LyohKzMr">https://t.co/D5LyohKzMr</a> <a href="https://t.co/E7drwVAmD2">pic.twitter.com/E7drwVAmD2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341972869572677634?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="und" dir="ltr">🤔<a href="https://t.co/z64tditaQG">https://t.co/z64tditaQG</a> <a href="https://t.co/jhJW0iiASY">pic.twitter.com/jhJW0iiASY</a></p>&mdash; 深層学習ワクワク挑戦チーム (@mosko_mule) <a href="https://twitter.com/mosko_mule/status/1341973695250583558?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Vid2Actor: Free-viewpoint Animatable Person Synthesis from Video in the  Wild

Chung-Yi Weng, Brian Curless, Ira Kemelmacher-Shlizerman

- retweets: 182, favorites: 78 (12/25/2020 09:42:53)

- links: [abs](https://arxiv.org/abs/2012.12884) | [pdf](https://arxiv.org/pdf/2012.12884)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Given an "in-the-wild" video of a person, we reconstruct an animatable model of the person in the video. The output model can be rendered in any body pose to any camera view, via the learned controls, without explicit 3D mesh reconstruction. At the core of our method is a volumetric 3D human representation reconstructed with a deep network trained on input video, enabling novel pose/view synthesis. Our method is an advance over GAN-based image-to-image translation since it allows image synthesis for any pose and camera via the internal 3D representation, while at the same time it does not require a pre-rigged model or ground truth meshes for training, as in mesh-based learning. Experiments validate the design choices and yield results on synthetic data and on real videos of diverse people performing unconstrained activities (e.g. dancing or playing tennis). Finally, we demonstrate motion re-targeting and bullet-time rendering with the learned models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vid2Actor: Free-viewpoint Animatable Person Synthesis from Video in the Wild<br>pdf: <a href="https://t.co/b1QF1ZeZPv">https://t.co/b1QF1ZeZPv</a><br>abs: <a href="https://t.co/MuCNRa3SQL">https://t.co/MuCNRa3SQL</a><br>project page: <a href="https://t.co/eymPtwdzX7">https://t.co/eymPtwdzX7</a> <a href="https://t.co/J6IEXJs2rk">pic.twitter.com/J6IEXJs2rk</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341940800591187968?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic  Parsing

Xi Victoria Lin, Richard Socher, Caiming Xiong

- retweets: 120, favorites: 98 (12/25/2020 09:42:53)

- links: [abs](https://arxiv.org/abs/2012.12627) | [pdf](https://arxiv.org/pdf/2012.12627)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DB](https://arxiv.org/list/cs.DB/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present BRIDGE, a powerful sequential architecture for modeling dependencies between natural language questions and relational databases in cross-DB semantic parsing. BRIDGE represents the question and DB schema in a tagged sequence where a subset of the fields are augmented with cell values mentioned in the question. The hybrid sequence is encoded by BERT with minimal subsequent layers and the text-DB contextualization is realized via the fine-tuned deep attention in BERT. Combined with a pointer-generator decoder with schema-consistency driven search space pruning, BRIDGE attained state-of-the-art performance on popular cross-DB text-to-SQL benchmarks, Spider (71.1\% dev, 67.5\% test with ensemble model) and WikiSQL (92.6\% dev, 91.9\% test). Our analysis shows that BRIDGE effectively captures the desired cross-modal dependencies and has the potential to generalize to more text-DB related tasks. Our implementation is available at \url{https://github.com/salesforce/TabularSemanticParsing}.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Mapping natural language to complex SQL queries across different databases demands modeling of both the DB relations and SQL syntax structures, so how well can Seq2Seq style models perform on such tasks? <br><br>❄️ Paper: <a href="https://t.co/FCYhmdboFP">https://t.co/FCYhmdboFP</a><br>❄️ Code: <a href="https://t.co/ZaHkVAygU2">https://t.co/ZaHkVAygU2</a> <a href="https://t.co/6dmw8UtZ3B">pic.twitter.com/6dmw8UtZ3B</a></p>&mdash; Victoria X Lin (@VictoriaLinML) <a href="https://twitter.com/VictoriaLinML/status/1341928032362143745?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. A Survey on Visual Transformer

Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Zhaohui Yang, Yiman Zhang, Dacheng Tao

- retweets: 96, favorites: 56 (12/25/2020 09:42:53)

- links: [abs](https://arxiv.org/abs/2012.12556) | [pdf](https://arxiv.org/pdf/2012.12556)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Transformer is a type of deep neural network mainly based on self-attention mechanism which is originally applied in natural language processing field. Inspired by the strong representation ability of transformer, researchers propose to extend transformer for computer vision tasks. Transformer-based models show competitive and even better performance on various visual benchmarks compared to other network types such as convolutional networks and recurrent networks. In this paper we provide a literature review of these visual transformer models by categorizing them in different tasks and analyze the advantages and disadvantages of these methods. In particular, the main categories include the basic image classification, high-level vision, low-level vision and video processing. Self-attention in computer vision is also briefly revisited as self-attention is the base component in transformer. Efficient transformer methods are included for pushing transformer into real applications. Finally, we give a discussion about the further research directions for visual transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Survey on Visual Transformer <a href="https://t.co/Pz9eu4t7bT">https://t.co/Pz9eu4t7bT</a></p>&mdash; arXiv CS-CV (@arxiv_cscv) <a href="https://twitter.com/arxiv_cscv/status/1342087717581045760?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Efficient video annotation with visual interpolation and frame selection  guidance

A. Kuznetsova, A. Talati, Y. Luo, K. Simmons, V. Ferrari

- retweets: 56, favorites: 36 (12/25/2020 09:42:53)

- links: [abs](https://arxiv.org/abs/2012.12554) | [pdf](https://arxiv.org/pdf/2012.12554)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.HC](https://arxiv.org/list/cs.HC/recent)

We introduce a unified framework for generic video annotation with bounding boxes. Video annotation is a longstanding problem, as it is a tedious and time-consuming process. We tackle two important challenges of video annotation: (1) automatic temporal interpolation and extrapolation of bounding boxes provided by a human annotator on a subset of all frames, and (2) automatic selection of frames to annotate manually. Our contribution is two-fold: first, we propose a model that has both interpolating and extrapolating capabilities; second, we propose a guiding mechanism that sequentially generates suggestions for what frame to annotate next, based on the annotations made previously. We extensively evaluate our approach on several challenging datasets in simulation and demonstrate a reduction in terms of the number of manual bounding boxes drawn by 60% over linear interpolation and by 35% over an off-the-shelf tracker. Moreover, we also show 10% annotation time improvement over a state-of-the-art method for video annotation with bounding boxes [25]. Finally, we run human annotation experiments and provide extensive analysis of the results, showing that our approach reduces actual measured annotation time by 50% compared to commonly used linear interpolation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient video annotation with visual interpolation and frame selection guidance<br>pdf: <a href="https://t.co/bw40zAG7TD">https://t.co/bw40zAG7TD</a><br>abs: <a href="https://t.co/qFDqHX5vaB">https://t.co/qFDqHX5vaB</a> <a href="https://t.co/kLP1FK5Tpl">pic.twitter.com/kLP1FK5Tpl</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1341981822251634689?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Scalable Optical Learning Operator

Uğur Teğin, Mustafa Yıldırım, İlker Oğuz, Christophe Moser, Demetri Psaltis

- retweets: 28, favorites: 40 (12/25/2020 09:42:54)

- links: [abs](https://arxiv.org/abs/2012.12404) | [pdf](https://arxiv.org/pdf/2012.12404)
- [physics.optics](https://arxiv.org/list/physics.optics/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Today's heavy machine learning tasks are fueled by large datasets. Computing is performed with power hungry processors whose performance is ultimately limited by the data transfer to and from memory. Optics is one of the powerful means of communicating and processing information and there is intense current interest in optical information processing for realizing high-speed computations. Here we present and experimentally demonstrate an optical computing framework based on spatiotemporal effects in multimode fibers for a range of learning tasks from classifying COVID-19 X-ray lung images and speech recognition to predicting age from face images. The presented framework overcomes the energy scaling problem of existing systems without compromising speed. We leveraged simultaneous, linear, and nonlinear interaction of spatial modes as a computation engine. We numerically and experimentally showed the ability of the method to execute several different tasks with accuracy comparable to a digital implementation. Our results indicate that a powerful supercomputer would be required to duplicate the performance of the multimode fiber-based computer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Meet with our recent study on optical computing, SOLO <a href="https://t.co/PC8YjstQli">https://t.co/PC8YjstQli</a> We demonstrate an optical computing framework based on spatiotemporal effects in multimode fibers for a range of learning tasks from classifying COVID-19 cases and speech recognition to predicting age.</p>&mdash; Ugur Tegin (@u_tegin) <a href="https://twitter.com/u_tegin/status/1342023173714415616?ref_src=twsrc%5Etfw">December 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. TicketTalk: Toward human-level performance with end-to-end,  transaction-based dialog systems

Bill Byrne, Karthik Krishnamoorthi, Saravanan Ganesh, Mihir Sanjay Kale

- retweets: 37, favorites: 22 (12/25/2020 09:42:54)

- links: [abs](https://arxiv.org/abs/2012.12458) | [pdf](https://arxiv.org/pdf/2012.12458)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present a data-driven, end-to-end approach to transaction-based dialog systems that performs at near-human levels in terms of verbal response quality and factual grounding accuracy. We show that two essential components of the system produce these results: a sufficiently large and diverse, in-domain labeled dataset, and a neural network-based, pre-trained model that generates both verbal responses and API call predictions. In terms of data, we introduce TicketTalk, a movie ticketing dialog dataset with 23,789 annotated conversations. The movie ticketing conversations range from completely open-ended and unrestricted to more structured, both in terms of their knowledge base, discourse features, and number of turns. In qualitative human evaluations, model-generated responses trained on just 10,000 TicketTalk dialogs were rated to "make sense" 86.5 percent of the time, almost the same as human responses in the same contexts. Our simple, API-focused annotation schema results in a much easier labeling task making it faster and more cost effective. It is also the key component for being able to predict API calls accurately. We handle factual grounding by incorporating API calls in the training data, allowing our model to learn which actions to take and when. Trained on the same 10,000-dialog set, the model's API call predictions were rated to be correct 93.9 percent of the time in our evaluations, surpassing the ratings for the corresponding human labels. We show how API prediction and response generation scores improve as the dataset size incrementally increases from 5000 to 21,000 dialogs. Our analysis also clearly illustrates the benefits of pre-training. We are publicly releasing the TicketTalk dataset with this paper to facilitate future work on transaction-based dialogs.



