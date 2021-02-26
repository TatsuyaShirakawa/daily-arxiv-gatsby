---
title: Hot Papers 2021-02-25
date: 2021-02-26T10:52:45.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-25"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-25"
socialImage: "/media/flying-marine.jpg"

---

# 1. Zero-Shot Text-to-Image Generation

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever

- retweets: 12998, favorites: 79 (02/26/2021 10:52:45)

- links: [abs](https://arxiv.org/abs/2102.12092) | [pdf](https://arxiv.org/pdf/2102.12092)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks supplied during training. We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data. With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">今年1月にOpenAIが発表した激ヤバなニューラルネット「DALL-E」の論文がとうとう公開された。開発者も予想していなかった多種多様な画像を作り出すことができる。高い抽象度で珍しい概念を構成する能力も確認。さらに、画像から画像への変換をテキストで制御することも可能<a href="https://t.co/f2y6lbETCQ">https://t.co/f2y6lbETCQ</a> <a href="https://t.co/UInVjMJwtJ">pic.twitter.com/UInVjMJwtJ</a></p>&mdash; 小猫遊りょう（たかにゃし・りょう） (@jaguring1) <a href="https://twitter.com/jaguring1/status/1364776840787726336?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Image Completion via Inference in Deep Generative Models

William Harvey, Saeid Naderiparizi, Frank Wood

- retweets: 3812, favorites: 482 (02/26/2021 10:52:45)

- links: [abs](https://arxiv.org/abs/2102.12037) | [pdf](https://arxiv.org/pdf/2102.12037)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We consider image completion from the perspective of amortized inference in an image generative model. We leverage recent state of the art variational auto-encoder architectures that have been shown to produce photo-realistic natural images at non-trivial resolutions. Through amortized inference in such a model we can train neural artifacts that produce diverse, realistic image completions even when the vast majority of an image is missing. We demonstrate superior sample quality and diversity compared to prior art on the CIFAR-10 and FFHQ-256 datasets. We conclude by describing and demonstrating an application that requires an in-painting model with the capabilities ours exhibits: the use of Bayesian optimal experimental design to select the most informative sequence of small field of view x-rays for chest pathology detection.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical variational autoencoders are getting more powerful every day. This paper looks at ways to convert a VAE into an image completion generative model. It seems we no longer need GANs or adversarial losses for this level of realism anymore? <a href="https://t.co/1pL8QTAKsC">https://t.co/1pL8QTAKsC</a> <a href="https://t.co/wNJPh9CdN4">https://t.co/wNJPh9CdN4</a></p>&mdash; hardmaru (@hardmaru) <a href="https://twitter.com/hardmaru/status/1364817306803605508?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Image Completion via Inference in Deep Generative Models<br>pdf: <a href="https://t.co/aP7ITXgx2i">https://t.co/aP7ITXgx2i</a><br>abs: <a href="https://t.co/QGzjNztYkx">https://t.co/QGzjNztYkx</a> <a href="https://t.co/SYoQY1ccR3">pic.twitter.com/SYoQY1ccR3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364813980494135304?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction  without Convolutions

Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao

- retweets: 2593, favorites: 218 (02/26/2021 10:52:45)

- links: [abs](https://arxiv.org/abs/2102.12122) | [pdf](https://arxiv.org/pdf/2102.12122)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Although using convolutional neural networks (CNNs) as backbones achieves great successes in computer vision, this work investigates a simple backbone network useful for many dense prediction tasks without convolutions. Unlike the recently-proposed Transformer model (e.g., ViT) that is specially designed for image classification, we propose Pyramid Vision Transformer~(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to prior arts. (1) Different from ViT that typically has low-resolution outputs and high computational and memory cost, PVT can be not only trained on dense partitions of the image to achieve high output resolution, which is important for dense predictions but also using a progressive shrinking pyramid to reduce computations of large feature maps. (2) PVT inherits the advantages from both CNN and Transformer, making it a unified backbone in various vision tasks without convolutions by simply replacing CNN backbones. (3) We validate PVT by conducting extensive experiments, showing that it boosts the performance of many downstream tasks, e.g., object detection, semantic, and instance segmentation. For example, with a comparable number of parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing RetinNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future researches. Code is available at https://github.com/whai362/PVT.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions<br>pdf: <a href="https://t.co/zq8nL4Kpab">https://t.co/zq8nL4Kpab</a><br>abs: <a href="https://t.co/BuHoKo502s">https://t.co/BuHoKo502s</a><br>github: <a href="https://t.co/b1SirihVe6">https://t.co/b1SirihVe6</a> <a href="https://t.co/ijv8bVQCbj">pic.twitter.com/ijv8bVQCbj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364754681361268744?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Do Transformer Modifications Transfer Across Implementations and  Applications?

Sharan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry, Michael Matena, Karishma Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan, Yanqi Zhou, Wei Li, Nan Ding, Jake Marcus, Adam Roberts, Colin Raffel

- retweets: 1936, favorites: 432 (02/26/2021 10:52:45)

- links: [abs](https://arxiv.org/abs/2102.11972) | [pdf](https://arxiv.org/pdf/2102.11972)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

The research community has proposed copious modifications to the Transformer architecture since it was introduced over three years ago, relatively few of which have seen widespread adoption. In this paper, we comprehensively evaluate many of these modifications in a shared experimental setting that covers most of the common uses of the Transformer in natural language processing. Surprisingly, we find that most modifications do not meaningfully improve performance. Furthermore, most of the Transformer variants we found beneficial were either developed in the same codebase that we used or are relatively minor changes. We conjecture that performance improvements may strongly depend on implementation details and correspondingly make some recommendations for improving the generality of experimental results.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">Do Transformer Modifications Transfer Across Implementations and Applications?<a href="https://t.co/WcAaHdHJUP">https://t.co/WcAaHdHJUP</a><br>「Transformer出現から3年、夥しい亜種が発表されてきが、生き残ったのも、意味がある改良も驚くほどわずか。」から始まるG社の圧倒的比較論文。正解はG社のコードベースを使うことでした。 <a href="https://t.co/0bJsq1BI1D">pic.twitter.com/0bJsq1BI1D</a></p>&mdash; ワクワクさん（深層学習） (@mosko_mule) <a href="https://twitter.com/mosko_mule/status/1364874565776642050?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do Transformer Modifications Transfer Across Implementations and Applications?<br><br>Most modifications of Transformers do not meaningfully improve performance and may strongly depend on implementation details.<a href="https://t.co/24AO7f4pAr">https://t.co/24AO7f4pAr</a> <a href="https://t.co/IJ69JCOFwq">pic.twitter.com/IJ69JCOFwq</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1364755835801329665?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Teach Me to Explain: A Review of Datasets for Explainable NLP

Sarah Wiegreffe, Ana Marasović

- retweets: 1194, favorites: 138 (02/26/2021 10:52:46)

- links: [abs](https://arxiv.org/abs/2102.12060) | [pdf](https://arxiv.org/pdf/2102.12060)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Explainable NLP (ExNLP) has increasingly focused on collecting human-annotated explanations. These explanations are used downstream in three ways: as data augmentation to improve performance on a predictive task, as a loss signal to train models to produce explanations for their predictions, and as a means to evaluate the quality of model-generated explanations. In this review, we identify three predominant classes of explanations (highlights, free-text, and structured), organize the literature on annotating each type, point to what has been learned to date, and give recommendations for collecting ExNLP datasets in the future.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share our new preprint (with <a href="https://twitter.com/anmarasovic?ref_src=twsrc%5Etfw">@anmarasovic</a>) “Teach Me to Explain: A Review of Datasets for Explainable NLP”<br>Paper: <a href="https://t.co/P5mUz9JVmJ">https://t.co/P5mUz9JVmJ</a><br>Website: <a href="https://t.co/61gZfnZYp8">https://t.co/61gZfnZYp8</a><br><br>It’s half survey, half reflections for more standardized ExNLP dataset collection. Highlights:<br><br>1/6</p>&mdash; Sarah Wiegreffe (@sarahwiegreffe) <a href="https://twitter.com/sarahwiegreffe/status/1365057786250489858?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Modern Koopman Theory for Dynamical Systems

Steven L. Brunton, Marko Budišić, Eurika Kaiser, J. Nathan Kutz

- retweets: 828, favorites: 193 (02/26/2021 10:52:46)

- links: [abs](https://arxiv.org/abs/2102.12086) | [pdf](https://arxiv.org/pdf/2102.12086)
- [math.DS](https://arxiv.org/list/math.DS/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent) | [math.OC](https://arxiv.org/list/math.OC/recent)

The field of dynamical systems is being transformed by the mathematical tools and algorithms emerging from modern computing and data science. First-principles derivations and asymptotic reductions are giving way to data-driven approaches that formulate models in operator theoretic or probabilistic frameworks. Koopman spectral theory has emerged as a dominant perspective over the past decade, in which nonlinear dynamics are represented in terms of an infinite-dimensional linear operator acting on the space of all possible measurement functions of the system. This linear representation of nonlinear dynamics has tremendous potential to enable the prediction, estimation, and control of nonlinear systems with standard textbook methods developed for linear systems. However, obtaining finite-dimensional coordinate systems and embeddings in which the dynamics appear approximately linear remains a central open challenge. The success of Koopman analysis is due primarily to three key factors: 1) there exists rigorous theory connecting it to classical geometric approaches for dynamical systems, 2) the approach is formulated in terms of measurements, making it ideal for leveraging big-data and machine learning techniques, and 3) simple, yet powerful numerical algorithms, such as the dynamic mode decomposition (DMD), have been developed and extended to reduce Koopman theory to practice in real-world applications. In this review, we provide an overview of modern Koopman operator theory, describing recent theoretical and algorithmic developments and highlighting these methods with a diverse range of applications. We also discuss key advances and challenges in the rapidly growing field of machine learning that are likely to drive future developments and significantly transform the theoretical landscape of dynamical systems.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very excited to announce a new review paper on &quot;Modern Koopman Theory for Dynamical Systems&quot;<br><br>which was a great collaboration with Marko Budišić (<a href="https://twitter.com/dynamicalmarko?ref_src=twsrc%5Etfw">@dynamicalmarko</a>), Eurika Kaiser, and Nathan Kutz.<br><br>Check it out here: <a href="https://t.co/osbcJqG5aE">https://t.co/osbcJqG5aE</a><br><br>1/n <a href="https://t.co/A9da4xOIf1">pic.twitter.com/A9da4xOIf1</a></p>&mdash; Steven Brunton (@eigensteve) <a href="https://twitter.com/eigensteve/status/1365040453763014660?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="fi" dir="ltr">New review article on Koopmania: &quot;Modern Koopman Theory for Dynamical Systems&quot; (by Steven L. Brunton, Marko Budišić, Eurika Kaiser, J. Nathan Kutz): <a href="https://t.co/B3Rog5q5UT">https://t.co/B3Rog5q5UT</a></p>&mdash; DynamicalSystemsSIAM (@DynamicsSIAM) <a href="https://twitter.com/DynamicsSIAM/status/1364759733853294594?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. When Attention Meets Fast Recurrence: Training Language Models with  Reduced Compute

Tao Lei

- retweets: 166, favorites: 36 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12459) | [pdf](https://arxiv.org/pdf/2102.12459)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Large language models have become increasingly difficult to train because of the required computation time and cost. In this work, we present SRU++, a recurrent unit with optional built-in attention that exhibits state-of-the-art modeling capacity and training efficiency. On standard language modeling benchmarks such as enwik8 and Wiki-103 datasets, our model obtains better perplexity and bits-per-character (bpc) while using 2.5x-10x less training time and cost compared to top-performing Transformer models. Our results reaffirm that attention is not all we need and can be complementary to other sequential modeling modules. Moreover, fast recurrence with little attention can be a leading model architecture.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our work of SRU++: <a href="https://t.co/l1UisqvH17">https://t.co/l1UisqvH17</a><br><br>We show that fast RNNs with little attention not only achieve top results but also reduce the training cost greatly. This reaffirms previous work such as SHA-LSTM and shares an orthogonal idea to accelerating attention. <a href="https://t.co/9QqmqKQurQ">https://t.co/9QqmqKQurQ</a></p>&mdash; taolei (@taolei15949106) <a href="https://twitter.com/taolei15949106/status/1364980529007845381?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. A Framework for Integrating Gesture Generation Models into Interactive  Conversational Agents

Rajmund Nagy, Taras Kucherenko, Birger Moell, André Pereira, Hedvig Kjellström, Ulysses Bernardet

- retweets: 64, favorites: 40 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12302) | [pdf](https://arxiv.org/pdf/2102.12302)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Embodied conversational agents (ECAs) benefit from non-verbal behavior for natural and efficient interaction with users. Gesticulation - hand and arm movements accompanying speech - is an essential part of non-verbal behavior. Gesture generation models have been developed for several decades: starting with rule-based and ending with mainly data-driven methods. To date, recent end-to-end gesture generation methods have not been evaluated in a real-time interaction with users. We present a proof-of-concept framework, which is intended to facilitate evaluation of modern gesture generation models in interaction.   We demonstrate an extensible open-source framework that contains three components: 1) a 3D interactive agent; 2) a chatbot backend; 3) a gesticulating system. Each component can be replaced, making the proposed framework applicable for investigating the effect of different gesturing models in real-time interactions with different communication modalities, chatbot backends, or different agent appearances. The code and video are available at the project page https://nagyrajmund.github.io/project/gesturebot.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Framework for Integrating Gesture Generation Models into Interactive Conversational Agents<br>pdf: <a href="https://t.co/3tL6ucnnaF">https://t.co/3tL6ucnnaF</a><br>abs: <a href="https://t.co/xmVjpgtHaj">https://t.co/xmVjpgtHaj</a><br>project page: <a href="https://t.co/OboZ2yeFYA">https://t.co/OboZ2yeFYA</a><br>github:<a href="https://t.co/to9121o1hl">https://t.co/to9121o1hl</a> <a href="https://t.co/l07qtnrrhL">pic.twitter.com/l07qtnrrhL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364762283264638983?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Synthetic Returns for Long-Term Credit Assignment

David Raposo, Sam Ritter, Adam Santoro, Greg Wayne, Theophane Weber, Matt Botvinick, Hado van Hasselt, Francis Song

- retweets: 30, favorites: 48 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12425) | [pdf](https://arxiv.org/pdf/2102.12425)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

Since the earliest days of reinforcement learning, the workhorse method for assigning credit to actions over time has been temporal-difference (TD) learning, which propagates credit backward timestep-by-timestep. This approach suffers when delays between actions and rewards are long and when intervening unrelated events contribute variance to long-term returns. We propose state-associative (SA) learning, where the agent learns associations between states and arbitrarily distant future rewards, then propagates credit directly between the two. In this work, we use SA-learning to model the contribution of past states to the current reward. With this model we can predict each state's contribution to the far future, a quantity we call "synthetic returns". TD-learning can then be applied to select actions that maximize these synthetic returns (SRs). We demonstrate the effectiveness of augmenting agents with SRs across a range of tasks on which TD-learning alone fails. We show that the learned SRs are interpretable: they spike for states that occur after critical actions are taken. Finally, we show that our IMPALA-based SR agent solves Atari Skiing -- a game with a lengthy reward delay that posed a major hurdle to deep-RL agents -- 25 times faster than the published state-of-the-art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New DeepMind paper &quot;Synthetic Returns for Long-Term Credit Assignment&quot; (<a href="https://t.co/3fmHhLtQdw">https://t.co/3fmHhLtQdw</a>) seems very interesting. It is a little confusing to abbreviate &quot;Synthetic Returns&quot; as SRs though, since that already means &quot;Successor Representation&quot; in the RL context. 😅</p>&mdash; Arthur Juliani (@awjuliani) <a href="https://twitter.com/awjuliani/status/1365079620798738435?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. A Straightforward Framework For Video Retrieval Using CLIP

Jesús Andrés Portillo-Quintero, José Carlos Ortiz-Bayliss, Hugo Terashima-Marín

- retweets: 32, favorites: 45 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12443) | [pdf](https://arxiv.org/pdf/2102.12443)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Video Retrieval is a challenging task where a text query is matched to a video or vice versa. Most of the existing approaches for addressing such a problem rely on annotations made by the users. Although simple, this approach is not always feasible in practice. In this work, we explore the application of the language-image model, CLIP, to obtain video representations without the need for said annotations. This model was explicitly trained to learn a common space where images and text can be compared. Using various techniques described in this document, we extended its application to videos, obtaining state-of-the-art results on the MSR-VTT and MSVD benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Straightforward Framework For Video Retrieval Using CLIP<br>pdf: <a href="https://t.co/XCToU20rYU">https://t.co/XCToU20rYU</a><br>abs: <a href="https://t.co/iDHWvlKyDo">https://t.co/iDHWvlKyDo</a> <a href="https://t.co/86bHwvAoxF">pic.twitter.com/86bHwvAoxF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364822463775932416?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Deep Video Prediction for Time Series Forecasting

Zhen Zeng, Tucker Balch, Manuela Veloso

- retweets: 25, favorites: 31 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12061) | [pdf](https://arxiv.org/pdf/2102.12061)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [econ.EM](https://arxiv.org/list/econ.EM/recent)

Time series forecasting is essential for decision making in many domains. In this work, we address the challenge of predicting prices evolution among multiple potentially interacting financial assets. A solution to this problem has obvious importance for governments, banks, and investors. Statistical methods such as Auto Regressive Integrated Moving Average (ARIMA) are widely applied to these problems. In this paper, we propose to approach economic time series forecasting of multiple financial assets in a novel way via video prediction. Given past prices of multiple potentially interacting financial assets, we aim to predict the prices evolution in the future. Instead of treating the snapshot of prices at each time point as a vector, we spatially layout these prices in 2D as an image, such that we can harness the power of CNNs in learning a latent representation for these financial assets. Thus, the history of these prices becomes a sequence of images, and our goal becomes predicting future images. We build on a state-of-the-art video prediction method for forecasting future images. Our experiments involve the prediction task of the price evolution of nine financial assets traded in U.S. stock markets. The proposed method outperforms baselines including ARIMA, Prophet, and variations of the proposed method, demonstrating the benefits of harnessing the power of CNNs in the problem of economic time series forecasting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep Video Prediction for Time Series Forecasting<br>pdf: <a href="https://t.co/uWM7ScihxR">https://t.co/uWM7ScihxR</a><br>abs: <a href="https://t.co/28OOyN97bJ">https://t.co/28OOyN97bJ</a> <a href="https://t.co/WipJuHBMES">pic.twitter.com/WipJuHBMES</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1364758775949844481?ref_src=twsrc%5Etfw">February 25, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Bridging Breiman's Brook: From Algorithmic Modeling to Statistical  Learning

Lucas Mentch, Giles Hooker

- retweets: 42, favorites: 9 (02/26/2021 10:52:47)

- links: [abs](https://arxiv.org/abs/2102.12328) | [pdf](https://arxiv.org/pdf/2102.12328)
- [stat.OT](https://arxiv.org/list/stat.OT/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In 2001, Leo Breiman wrote of a divide between "data modeling" and "algorithmic modeling" cultures. Twenty years later this division feels far more ephemeral, both in terms of assigning individuals to camps, and in terms of intellectual boundaries. We argue that this is largely due to the "data modelers" incorporating algorithmic methods into their toolbox, particularly driven by recent developments in the statistical understanding of Breiman's own Random Forest methods. While this can be simplistically described as "Breiman won", these same developments also expose the limitations of the prediction-first philosophy that he espoused, making careful statistical analysis all the more important. This paper outlines these exciting recent developments in the random forest literature which, in our view, occurred as a result of a necessary blending of the two ways of thinking Breiman originally described. We also ask what areas statistics and statisticians might currently overlook.



