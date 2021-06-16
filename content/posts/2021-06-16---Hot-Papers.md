---
title: Hot Papers 2021-06-16
date: 2021-06-17T07:07:08.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-16"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-16"
socialImage: "/media/flying-marine.jpg"

---

# 1. BEiT: BERT Pre-Training of Image Transformers

Hangbo Bao, Li Dong, Furu Wei

- retweets: 2350, favorites: 288 (06/17/2021 07:07:08)

- links: [abs](https://arxiv.org/abs/2106.08254) | [pdf](https://arxiv.org/pdf/2106.08254)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%). The code and pretrained models are available at https://aka.ms/beit.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">BEIT: BERT Pre-Training of Image Transformers<br>pdf: <a href="https://t.co/WiFZIiErLt">https://t.co/WiFZIiErLt</a><br>abs: <a href="https://t.co/Ld2067ltiV">https://t.co/Ld2067ltiV</a><br><br>large-size BEIT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised<br>pre-training on ImageNet-22K (85.2%) <a href="https://t.co/abMaWZ1aZ8">pic.twitter.com/abMaWZ1aZ8</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404966151541739522?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Self-Supervised Learning with Kernel Dependence Maximization

Yazhe Li, Roman Pogodin, Danica J. Sutherland, Arthur Gretton

- retweets: 780, favorites: 176 (06/17/2021 07:07:08)

- links: [abs](https://arxiv.org/abs/2106.08320) | [pdf](https://arxiv.org/pdf/2106.08320)
- [stat.ML](https://arxiv.org/list/stat.ML/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We approach self-supervised learning of image representations from a statistical dependence perspective, proposing Self-Supervised Learning with the Hilbert-Schmidt Independence Criterion (SSL-HSIC). SSL-HSIC maximizes dependence between representations of transformed versions of an image and the image identity, while minimizing the kernelized variance of those features. This self-supervised learning framework yields a new understanding of InfoNCE, a variational lower bound on the mutual information (MI) between different transformations. While the MI itself is known to have pathologies which can result in meaningless representations being learned, its bound is much better behaved: we show that it implicitly approximates SSL-HSIC (with a slightly different regularizer). Our approach also gives us insight into BYOL, since SSL-HSIC similarly learns local neighborhoods of samples. SSL-HSIC allows us to directly optimize statistical dependence in time linear in the batch size, without restrictive data assumptions or indirect mutual information estimators. Trained with or without a target network, SSL-HSIC matches the current state-of-the-art for standard linear evaluation on ImageNet, semi-supervised learning and transfer to other classification and vision tasks such as semantic segmentation, depth estimation and object recognition.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Your contrastive self-supervised learning is secretly a kernel method....<a href="https://t.co/QEwhMR7Nz6">https://t.co/QEwhMR7Nz6</a><br>with <a href="https://twitter.com/yazhe_li?ref_src=twsrc%5Etfw">@yazhe_li</a> <a href="https://twitter.com/rmnpogodin?ref_src=twsrc%5Etfw">@rmnpogodin</a> <a href="https://twitter.com/d_j_sutherland?ref_src=twsrc%5Etfw">@d_j_sutherland</a></p>&mdash; Arthur Gretton (@ArthurGretton) <a href="https://twitter.com/ArthurGretton/status/1405108119068364803?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Controlling Neural Networks with Rule Representations

Sungyong Seo, Sercan O. Arik, Jinsung Yoon, Xiang Zhang, Kihyuk Sohn, Tomas Pfister

- retweets: 676, favorites: 117 (06/17/2021 07:07:10)

- links: [abs](https://arxiv.org/abs/2106.07804) | [pdf](https://arxiv.org/pdf/2106.07804)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We propose a novel training method to integrate rules into deep learning, in a way their strengths are controllable at inference. Deep Neural Networks with Controllable Rule Representations (DeepCTRL) incorporates a rule encoder into the model coupled with a rule-based objective, enabling a shared representation for decision making. DeepCTRL is agnostic to data type and model architecture. It can be applied to any kind of rule defined for inputs and outputs. The key aspect of DeepCTRL is that it does not require retraining to adapt the rule strength -- at inference, the user can adjust it based on the desired operation point on accuracy vs. rule verification ratio. In real-world domains where incorporating rules is critical -- such as Physics, Retail and Healthcare -- we show the effectiveness of DeepCTRL in teaching rules for deep learning. DeepCTRL improves the trust and reliability of the trained models by significantly increasing their rule verification ratio, while also providing accuracy gains at downstream tasks. Additionally, DeepCTRL enables novel use cases such as hypothesis testing of the rules on data samples, and unsupervised adaptation based on shared rules between datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Controlling Neural Networks with Rule Representations<br>pdf: <a href="https://t.co/dZnV6cZsYC">https://t.co/dZnV6cZsYC</a><br>abs: <a href="https://t.co/agPEJ5tBze">https://t.co/agPEJ5tBze</a> <a href="https://t.co/GEIq3ve78M">pic.twitter.com/GEIq3ve78M</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404971191358660610?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Revisiting the Calibration of Modern Neural Networks

Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic

- retweets: 576, favorites: 189 (06/17/2021 07:07:10)

- links: [abs](https://arxiv.org/abs/2106.07998) | [pdf](https://arxiv.org/pdf/2106.07998)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Accurate estimation of predictive uncertainty (model calibration) is essential for the safe application of neural networks. Many instances of miscalibration in modern neural networks have been reported, suggesting a trend that newer, more accurate models produce poorly calibrated predictions. Here, we revisit this question for recent state-of-the-art image classification models. We systematically relate model calibration and accuracy, and find that the most recent models, notably those not using convolutions, are among the best calibrated. Trends observed in prior model generations, such as decay of calibration with distribution shift or model size, are less pronounced in recent architectures. We also show that model size and amount of pretraining do not fully explain these differences, suggesting that architecture is a major determinant of calibration properties.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New paper: Revisiting the Calibration of Modern Neural Networks (<a href="https://t.co/VmmiWrfmrh">https://t.co/VmmiWrfmrh</a>). We studied the calibration of MLP-Mixer, Vision Transformers, BiT, and many others. Non-convolutional models are doing surprisingly well! 1/5 <a href="https://t.co/PRwtz3lT4d">pic.twitter.com/PRwtz3lT4d</a></p>&mdash; Matthias Minderer (@MJLM3) <a href="https://twitter.com/MJLM3/status/1405230885901897731?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Revisiting the Calibration of Modern Neural Networks<br>pdf: <a href="https://t.co/GMaqc0tyj0">https://t.co/GMaqc0tyj0</a><br>abs: <a href="https://t.co/0ParfICt6H">https://t.co/0ParfICt6H</a><br><br>a large study of the calibration of recent state-of-the-art image models and its relationship with accuracy <a href="https://t.co/ckYW7IDa0Y">pic.twitter.com/ckYW7IDa0Y</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405003551311552515?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Learning Equivariant Energy Based Models with Equivariant Stein  Variational Gradient Descent

Priyank Jaini, Lars Holdijk, Max Welling

- retweets: 625, favorites: 126 (06/17/2021 07:07:10)

- links: [abs](https://arxiv.org/abs/2106.07832) | [pdf](https://arxiv.org/pdf/2106.07832)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We focus on the problem of efficient sampling and learning of probability densities by incorporating symmetries in probabilistic models. We first introduce Equivariant Stein Variational Gradient Descent algorithm -- an equivariant sampling method based on Stein's identity for sampling from densities with symmetries. Equivariant SVGD explicitly incorporates symmetry information in a density through equivariant kernels which makes the resultant sampler efficient both in terms of sample complexity and the quality of generated samples. Subsequently, we define equivariant energy based models to model invariant densities that are learned using contrastive divergence. By utilizing our equivariant SVGD for training equivariant EBMs, we propose new ways of improving and scaling up training of energy based models. We apply these equivariant energy models for modelling joint densities in regression and classification tasks for image datasets, many-body particle systems and molecular structure generation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our latest work on incorporating symmetries in samplers and energy models by Learning Equivariant Energy Based Models with Equivariant Stein Variational Gradient Descent (<a href="https://t.co/8aL44iT1Ce">https://t.co/8aL44iT1Ce</a>) w. the amazing <a href="https://twitter.com/HoldijkLars?ref_src=twsrc%5Etfw">@HoldijkLars</a> (eq. contribution) &amp; <a href="https://twitter.com/wellingmax?ref_src=twsrc%5Etfw">@wellingmax</a> <br>🧵👇 <a href="https://t.co/IXDCWc6ck4">pic.twitter.com/IXDCWc6ck4</a></p>&mdash; Priyank Jaini (@priyankjaini) <a href="https://twitter.com/priyankjaini/status/1405027643775455238?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis

Jaesung Tae, Hyeongju Kim, Younggun Lee

- retweets: 414, favorites: 74 (06/17/2021 07:07:11)

- links: [abs](https://arxiv.org/abs/2106.07886) | [pdf](https://arxiv.org/pdf/2106.07886)
- [cs.SD](https://arxiv.org/list/cs.SD/recent)

Recent developments in deep learning have significantly improved the quality of synthesized singing voice audio. However, prominent neural singing voice synthesis systems suffer from slow inference speed due to their autoregressive design. Inspired by MLP-Mixer, a novel architecture introduced in the vision literature for attention-free image classification, we propose MLP Singer, a parallel Korean singing voice synthesis system. To the best of our knowledge, this is the first work that uses an entirely MLP-based architecture for voice synthesis. Listening tests demonstrate that MLP Singer outperforms a larger autoregressive GAN-based system, both in terms of audio quality and synthesis speed. In particular, MLP Singer achieves a real-time factor of up to 200 and 3400 on CPUs and GPUs respectively, enabling order of magnitude faster generation on both environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MLP Singer: Towards Rapid Parallel Korean Singing Voice Synthesis<br>pdf: <a href="https://t.co/Z6fAfyV7zv">https://t.co/Z6fAfyV7zv</a><br>abs: <a href="https://t.co/OUM7iTfhin">https://t.co/OUM7iTfhin</a><br>project page: <a href="https://t.co/Qs89Ohm3f3">https://t.co/Qs89Ohm3f3</a> <a href="https://t.co/9FZghaXF25">pic.twitter.com/9FZghaXF25</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404963740609650688?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. ARTA: Collection and Classification of Ambiguous Requests and Thoughtful  Actions

Shohei Tanaka, Koichiro Yoshino, Katsuhito Sudoh, Satoshi Nakamura

- retweets: 336, favorites: 55 (06/17/2021 07:07:11)

- links: [abs](https://arxiv.org/abs/2106.07999) | [pdf](https://arxiv.org/pdf/2106.07999)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Human-assisting systems such as dialogue systems must take thoughtful, appropriate actions not only for clear and unambiguous user requests, but also for ambiguous user requests, even if the users themselves are not aware of their potential requirements. To construct such a dialogue agent, we collected a corpus and developed a model that classifies ambiguous user requests into corresponding system actions. In order to collect a high-quality corpus, we asked workers to input antecedent user requests whose pre-defined actions could be regarded as thoughtful. Although multiple actions could be identified as thoughtful for a single user request, annotating all combinations of user requests and system actions is impractical. For this reason, we fully annotated only the test data and left the annotation of the training data incomplete. In order to train the classification model on such training data, we applied the positive/unlabeled (PU) learning method, which assumes that only a part of the data is labeled with positive examples. The experimental results show that the PU learning method achieved better performance than the general positive/negative (PN) learning method to classify thoughtful actions given an ambiguous user request.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">SIGDIAL に採択された論文を arXiv にて公開しました．<br>「ここの景色最高だね」という発話に対して「カメラを起動しましょうか？」のように気の利いた応答を返す対話エージェントに関する研究です．<a href="https://t.co/IBeI9uuWLu">https://t.co/IBeI9uuWLu</a><br>併せて論文中で使用したコーパスを公開しました．<a href="https://t.co/bSo99429gy">https://t.co/bSo99429gy</a></p>&mdash; Shohei Tanaka (@shohei_ta_ds7) <a href="https://twitter.com/shohei_ta_ds7/status/1405030702458966016?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Revisiting Model Stitching to Compare Neural Representations

Yamini Bansal, Preetum Nakkiran, Boaz Barak

- retweets: 240, favorites: 96 (06/17/2021 07:07:11)

- links: [abs](https://arxiv.org/abs/2106.07682) | [pdf](https://arxiv.org/pdf/2106.07682)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We revisit and extend model stitching (Lenc & Vedaldi 2015) as a methodology to study the internal representations of neural networks. Given two trained and frozen models $A$ and $B$, we consider a "stitched model'' formed by connecting the bottom-layers of $A$ to the top-layers of $B$, with a simple trainable layer between them. We argue that model stitching is a powerful and perhaps under-appreciated tool, which reveals aspects of representations that measures such as centered kernel alignment (CKA) cannot. Through extensive experiments, we use model stitching to obtain quantitative verifications for intuitive statements such as "good networks learn similar representations'', by demonstrating that good networks of the same architecture, but trained in very different ways (e.g.: supervised vs. self-supervised learning), can be stitched to each other without drop in performance. We also give evidence for the intuition that "more is better'' by showing that representations learnt with (1) more data, (2) bigger width, or (3) more training time can be "plugged in'' to weaker models to improve performance. Finally, our experiments reveal a new structural property of SGD which we call "stitching connectivity'', akin to mode-connectivity: typical minima reached by SGD can all be stitched to each other with minimal change in accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">1/7 Do neural nets with different architecture, training objectives, and data, learn similar representations? <br>With <a href="https://twitter.com/whybansal?ref_src=twsrc%5Etfw">@whybansal</a> and <a href="https://twitter.com/PreetumNakkiran?ref_src=twsrc%5Etfw">@PreetumNakkiran</a> we use &quot;stitching&quot; (Lenc-Vedaldi 2015) to study this question.<a href="https://t.co/CLRPWEBBfC">https://t.co/CLRPWEBBfC</a></p>&mdash; Boaz Barak (@boazbaraktcs) <a href="https://twitter.com/boazbaraktcs/status/1405154772920016900?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Question Answering Infused Pre-training of General-Purpose  Contextualized Representations

Robin Jia, Mike Lewis, Luke Zettlemoyer

- retweets: 230, favorites: 92 (06/17/2021 07:07:11)

- links: [abs](https://arxiv.org/abs/2106.08190) | [pdf](https://arxiv.org/pdf/2106.08190)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

This paper proposes a pre-training objective based on question answering (QA) for learning general-purpose contextual representations, motivated by the intuition that the representation of a phrase in a passage should encode all questions that the phrase can answer in context. We accomplish this goal by training a bi-encoder QA model, which independently encodes passages and questions, to match the predictions of a more accurate cross-encoder model on 80 million synthesized QA pairs. By encoding QA-relevant information, the bi-encoder's token-level representations are useful for non-QA downstream tasks without extensive (or in some cases, any) fine-tuning. We show large improvements over both RoBERTa-large and previous state-of-the-art results on zero-shot and few-shot paraphrase detection on four datasets, few-shot named entity recognition on two datasets, and zero-shot sentiment analysis on three datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share QuIP: Question Answering Infused Pre-training. We use bi-encoder QA to pre-train more contextualized representations, leading to better zero/few-shot learning on non-QA tasks. Joint work with <a href="https://twitter.com/ml_perception?ref_src=twsrc%5Etfw">@ml_perception</a> and <a href="https://twitter.com/LukeZettlemoyer?ref_src=twsrc%5Etfw">@LukeZettlemoyer</a> <a href="https://t.co/XiA8WpjUxI">https://t.co/XiA8WpjUxI</a> 1/7</p>&mdash; Robin Jia (@robinomial) <a href="https://twitter.com/robinomial/status/1404991750079406084?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Reverse Engineering of Generative Models: Inferring Model  Hyperparameters from Generated Images

Vishal Asnani, Xi Yin, Tal Hassner, Xiaoming Liu

- retweets: 196, favorites: 57 (06/17/2021 07:07:11)

- links: [abs](https://arxiv.org/abs/2106.07873) | [pdf](https://arxiv.org/pdf/2106.07873)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

State-of-the-art (SOTA) Generative Models (GMs) can synthesize photo-realistic images that are hard for humans to distinguish from genuine photos. We propose to perform reverse engineering of GMs to infer the model hyperparameters from the images generated by these models. We define a novel problem, "model parsing", as estimating GM network architectures and training loss functions by examining their generated images -- a task seemingly impossible for human beings. To tackle this problem, we propose a framework with two components: a Fingerprint Estimation Network (FEN), which estimates a GM fingerprint from a generated image by training with four constraints to encourage the fingerprint to have desired properties, and a Parsing Network (PN), which predicts network architecture and loss functions from the estimated fingerprints. To evaluate our approach, we collect a fake image dataset with $100$K images generated by $100$ GMs. Extensive experiments show encouraging results in parsing the hyperparameters of the unseen models. Finally, our fingerprint estimation can be leveraged for deepfake detection and image attribution, as we show by reporting SOTA results on both the recent Celeb-DF and image attribution benchmarks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Reverse Engineering of Generative Models: Inferring Model Hyperparameters from Generated Images<br>pdf: <a href="https://t.co/ZI3FlqaBln">https://t.co/ZI3FlqaBln</a><br>abs: <a href="https://t.co/Rd9nXMrvFi">https://t.co/Rd9nXMrvFi</a> <a href="https://t.co/As71ZvibGS">pic.twitter.com/As71ZvibGS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405002175705763841?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. CathAI: Fully Automated Interpretation of Coronary Angiograms Using  Neural Networks

Robert Avram, Jeffrey E. Olgin, Alvin Wan, Zeeshan Ahmed, Louis Verreault-Julien, Sean Abreau, Derek Wan, Joseph E. Gonzalez, Derek Y. So, Krishan Soni, Geoffrey H. Tison

- retweets: 141, favorites: 40 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.07708) | [pdf](https://arxiv.org/pdf/2106.07708)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Coronary heart disease (CHD) is the leading cause of adult death in the United States and worldwide, and for which the coronary angiography procedure is the primary gateway for diagnosis and clinical management decisions. The standard-of-care for interpretation of coronary angiograms depends upon ad-hoc visual assessment by the physician operator. However, ad-hoc visual interpretation of angiograms is poorly reproducible, highly variable and bias prone. Here we show for the first time that fully-automated angiogram interpretation to estimate coronary artery stenosis is possible using a sequence of deep neural network algorithms. The algorithmic pipeline we developed--called CathAI--achieves state-of-the art performance across the sequence of tasks required to accomplish automated interpretation of unselected, real-world angiograms. CathAI (Algorithms 1-2) demonstrated positive predictive value, sensitivity and F1 score of >=90% to identify the projection angle overall and >=93% for left or right coronary artery angiogram detection, the primary anatomic structures of interest. To predict obstructive coronary artery stenosis (>=70% stenosis), CathAI (Algorithm 4) exhibited an area under the receiver operating characteristic curve (AUC) of 0.862 (95% CI: 0.843-0.880). When externally validated in a healthcare system in another country, CathAI AUC was 0.869 (95% CI: 0.830-0.907) to predict obstructive coronary artery stenosis. Our results demonstrate that multiple purpose-built neural networks can function in sequence to accomplish the complex series of tasks required for automated analysis of real-world angiograms. Deployment of CathAI may serve to increase standardization and reproducibility in coronary stenosis assessment, while providing a robust foundation to accomplish future tasks for algorithmic angiographic interpretation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We’re excited to share our work using <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a> to perform fully automated estimation of coronary stenosis from real-world coronary angiograms, the central procedure for coronary heart disease <a href="https://t.co/PzI4Ee8mOU">https://t.co/PzI4Ee8mOU</a>. More at <a href="https://t.co/7fT5zN2gbr">https://t.co/7fT5zN2gbr</a> <a href="https://twitter.com/RobertAvramMD?ref_src=twsrc%5Etfw">@RobertAvramMD</a> <a href="https://t.co/FSdO0N8gRW">pic.twitter.com/FSdO0N8gRW</a></p>&mdash; Geoff Tison (@GeoffTison) <a href="https://twitter.com/GeoffTison/status/1405000171734269952?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">After 3 years of hard work, we published the preprint of <a href="https://twitter.com/hashtag/CathAI?src=hash&amp;ref_src=twsrc%5Etfw">#CathAI</a>  <a href="https://t.co/2ppjK2Gk0J">https://t.co/2ppjK2Gk0J</a>. This is the first proof of concept to show that <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a> can perform the automated reading of <a href="https://twitter.com/hashtag/coronary?src=hash&amp;ref_src=twsrc%5Etfw">#coronary</a> <a href="https://twitter.com/hashtag/angiograms?src=hash&amp;ref_src=twsrc%5Etfw">#angiograms</a>. <a href="https://t.co/k2y30VTL5W">https://t.co/k2y30VTL5W</a> <a href="https://twitter.com/UCSFCardiology?ref_src=twsrc%5Etfw">@UCSFCardiology</a> <a href="https://twitter.com/GeoffTison?ref_src=twsrc%5Etfw">@GeoffTison</a></p>&mdash; Robert Avram - robertmd.eth (@RobertAvramMD) <a href="https://twitter.com/RobertAvramMD/status/1404975949221822464?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Tree-Values: selective inference for regression trees

Anna C. Neufeld, Lucy L. Gao, Daniela M. Witten

- retweets: 100, favorites: 50 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.07816) | [pdf](https://arxiv.org/pdf/2106.07816)
- [stat.ME](https://arxiv.org/list/stat.ME/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

We consider conducting inference on the output of the Classification and Regression Tree (CART) [Breiman et al., 1984] algorithm. A naive approach to inference that does not account for the fact that the tree was estimated from the data will not achieve standard guarantees, such as Type 1 error rate control and nominal coverage. Thus, we propose a selective inference framework for conducting inference on a fitted CART tree. In a nutshell, we condition on the fact that the tree was estimated from the data. We propose a test for the difference in the mean response between a pair of terminal nodes that controls the selective Type 1 error rate, and a confidence interval for the mean response within a single terminal node that attains the nominal selective coverage. Efficient algorithms for computing the necessary conditioning sets are provided. We apply these methods in simulation and to a dataset involving the association between portion control interventions and caloric intake.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Super excited to announce our new pre-print available on Arxiv! <a href="https://t.co/V5XTKhsJO6">https://t.co/V5XTKhsJO6</a>. We (<a href="https://twitter.com/daniela_witten?ref_src=twsrc%5Etfw">@daniela_witten</a>, <a href="https://twitter.com/lucylgao?ref_src=twsrc%5Etfw">@lucylgao</a>, and I) introduce “treevalues”, a framework for selective inference on <br>CART regression trees. In honor of the occasion, I’m making my first twitter thread! 1/n</p>&mdash; Anna Neufeld (@AnnaCNeufeld) <a href="https://twitter.com/AnnaCNeufeld/status/1405219949602242564?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Color2Style: Real-Time Exemplar-Based Image Colorization with  Self-Reference Learning and Deep Feature Modulation

Hengyuan Zhao, Wenhao Wu, Yihao Liu, Dongliang He

- retweets: 112, favorites: 35 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.08017) | [pdf](https://arxiv.org/pdf/2106.08017)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.MM](https://arxiv.org/list/cs.MM/recent)

Legacy black-and-white photos are riddled with people's nostalgia and glorious memories of the past. To better relive the elapsed frozen moments, in this paper, we present a deep exemplar-based image colorization approach named Color2Style to resurrect these grayscale image media by filling them with vibrant colors. Generally, for exemplar-based colorization, unsupervised and unpaired training are usually adopted, due to the difficulty of obtaining input and ground truth image pairs. To train an exemplar-based colorization model, current algorithms usually strive to achieve two procedures: i) retrieving a large number of reference images with high similarity in advance, which is inevitably time-consuming and tedious; ii) designing complicated modules to transfer the colors of the reference image to the grayscale image, by calculating and leveraging the deep semantic correspondence between them (e.g., non-local operation). Contrary to the previous methods, we solve and simplify the above two steps in one end-to-end learning procedure. First, we adopt a self-augmented self-reference training scheme, where the reference image is generated by graphical transformations from the original colorful one whereby the training can be formulated in a paired manner. Second, instead of computing complex and inexplicable correspondence maps, our method exploits a simple yet effective deep feature modulation (DFM) module, which injects the color embeddings extracted from the reference image into the deep representations of the input grayscale image. Such design is much more lightweight and intelligible, achieving appealing performance with real-time processing speed. Moreover, our model does not require multifarious loss functions and regularization terms like existing methods, but only two widely used loss functions. Codes and models will be available at https://github.com/zhaohengyuan1/Color2Style.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Color2Style: Real-Time Exemplar-Based Image Colorization with Self-Reference Learning and Deep Feature Modulation<br>pdf: <a href="https://t.co/Tw8TNddwot">https://t.co/Tw8TNddwot</a><br>abs: <a href="https://t.co/jVESmeKndg">https://t.co/jVESmeKndg</a> <a href="https://t.co/nHssKHy0Yy">pic.twitter.com/nHssKHy0Yy</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404973183388860416?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Dynamic Head: Unifying Object Detection Heads with Attentions

Xiyang Dai, Yinpeng Chen, Bin Xiao, Dongdong Chen, Mengchen Liu, Lu Yuan, Lei Zhang

- retweets: 98, favorites: 47 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.08322) | [pdf](https://arxiv.org/pdf/2106.08322)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The complex nature of combining localization and classification in object detection has resulted in the flourished development of methods. Previous works tried to improve the performance in various object detection heads but failed to present a unified view. In this paper, we present a novel dynamic head framework to unify object detection heads with attentions. By coherently combining multiple self-attention mechanisms between feature levels for scale-awareness, among spatial locations for spatial-awareness, and within output channels for task-awareness, the proposed approach significantly improves the representation ability of object detection heads without any computational overhead. Further experiments demonstrate that the effectiveness and efficiency of the proposed dynamic head on the COCO benchmark. With a standard ResNeXt-101-DCN backbone, we largely improve the performance over popular object detectors and achieve a new state-of-the-art at 54.0 AP. Furthermore, with latest transformer backbone and extra data, we can push current best COCO result to a new record at 60.6 AP. The code will be released at https://github.com/microsoft/DynamicHead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dynamic Head: Unifying Object Detection Heads with Attentions<br>pdf: <a href="https://t.co/j0MIA845TX">https://t.co/j0MIA845TX</a><br>abs: <a href="https://t.co/9tOqt8gLgf">https://t.co/9tOqt8gLgf</a><br><br>with latest transformer backbone and extra data, can push current best COCO result to a new record at 60.6 AP <a href="https://t.co/EsCEF03gbH">pic.twitter.com/EsCEF03gbH</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404966846055452675?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram  Discriminators for High-Fidelity Waveform Generation

Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, Juntae Kim

- retweets: 100, favorites: 33 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.07889) | [pdf](https://arxiv.org/pdf/2106.07889)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent)

Most neural vocoders employ band-limited mel-spectrograms to generate waveforms. If full-band spectral features are used as the input, the vocoder can be provided with as much acoustic information as possible. However, in some models employing full-band mel-spectrograms, an over-smoothing problem occurs as part of which non-sharp spectrograms are generated. To address this problem, we propose UnivNet, a neural vocoder that synthesizes high-fidelity waveforms in real time. Inspired by works in the field of voice activity detection, we added a multi-resolution spectrogram discriminator that employs multiple linear spectrogram magnitudes computed using various parameter sets. Using full-band mel-spectrograms as input, we expect to generate high-resolution signals by adding a discriminator that employs spectrograms of multiple resolutions as the input. In an evaluation on a dataset containing information on hundreds of speakers, UnivNet obtained the best objective and subjective results among competing models for both seen and unseen speakers. These results, including the best subjective score for text-to-speech, demonstrate the potential for fast adaptation to new speakers without a need for training from scratch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation<br>pdf: <a href="https://t.co/mG2UzKggvK">https://t.co/mG2UzKggvK</a><br>abs: <a href="https://t.co/LyJdtqNh4q">https://t.co/LyJdtqNh4q</a><br>project page: <a href="https://t.co/mQ1BzDxPcC">https://t.co/mQ1BzDxPcC</a> <a href="https://t.co/VRi01eANRC">pic.twitter.com/VRi01eANRC</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404979580046393347?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Keep CALM and Improve Visual Feature Attribution

Jae Myung Kim, Junsuk Choe, Zeynep Akata, Seong Joon Oh

- retweets: 81, favorites: 43 (06/17/2021 07:07:12)

- links: [abs](https://arxiv.org/abs/2106.07861) | [pdf](https://arxiv.org/pdf/2106.07861)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

The class activation mapping, or CAM, has been the cornerstone of feature attribution methods for multiple vision tasks. Its simplicity and effectiveness have led to wide applications in the explanation of visual predictions and weakly-supervised localization tasks. However, CAM has its own shortcomings. The computation of attribution maps relies on ad-hoc calibration steps that are not part of the training computational graph, making it difficult for us to understand the real meaning of the attribution values. In this paper, we improve CAM by explicitly incorporating a latent variable encoding the location of the cue for recognition in the formulation, thereby subsuming the attribution map into the training computational graph. The resulting model, class activation latent mapping, or CALM, is trained with the expectation-maximization algorithm. Our experiments show that CALM identifies discriminative attributes for image classifiers more accurately than CAM and other visual attribution baselines. CALM also shows performance improvements over prior arts on the weakly-supervised object localization benchmarks. Our code is available at https://github.com/naver-ai/calm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Keep CALM and Improve Visual Feature Attribution<br>pdf: <a href="https://t.co/I90yEPbjse">https://t.co/I90yEPbjse</a><br>abs: <a href="https://t.co/JKIlsnNnGn">https://t.co/JKIlsnNnGn</a><br>github: <a href="https://t.co/MP1F8LXvY0">https://t.co/MP1F8LXvY0</a><br><br>identifies discriminative attributes for image classifiers more accurately than CAM and other visual attribution baselines <a href="https://t.co/50DBMoVYai">pic.twitter.com/50DBMoVYai</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404983014380158976?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. A General Purpose Transpiler for Fully Homomorphic Encryption

Shruthi Gorantala, Rob Springer, Sean Purser-Haskell, William Lam, Royce Wilson, Asra Ali, Eric P. Astor, Itai Zukerman, Sam Ruth, Christoph Dibak, Phillipp Schoppmann, Sasha Kulankhina, Alain Forget, David Marn, Cameron Tew, Rafael Misoczki, Bernat Guillen, Xinyu Ye, Dennis Kraft, Damien Desfontaines, Aishe Krishnamurthy, Miguel Guevara, Irippuge Milinda Perera, Yurii Sushko, Bryant Gipson

- retweets: 76, favorites: 36 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07893) | [pdf](https://arxiv.org/pdf/2106.07893)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.PL](https://arxiv.org/list/cs.PL/recent)

Fully homomorphic encryption (FHE) is an encryption scheme which enables computation on encrypted data without revealing the underlying data. While there have been many advances in the field of FHE, developing programs using FHE still requires expertise in cryptography. In this white paper, we present a fully homomorphic encryption transpiler that allows developers to convert high-level code (e.g., C++) that works on unencrypted data into high-level code that operates on encrypted data. Thus, our transpiler makes transformations possible on encrypted data.   Our transpiler builds on Google's open-source XLS SDK (https://github.com/google/xls) and uses an off-the-shelf FHE library, TFHE (https://tfhe.github.io/tfhe/), to perform low-level FHE operations. The transpiler design is modular, which means the underlying FHE library as well as the high-level input and output languages can vary. This modularity will help accelerate FHE research by providing an easy way to compare arbitrary programs in different FHE schemes side-by-side. We hope this lays the groundwork for eventual easy adoption of FHE by software developers. As a proof-of-concept, we are releasing an experimental transpiler (https://github.com/google/fully-homomorphic-encryption/tree/main/transpiler) as open-source software.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper describing our general-purpose transpiler for fully homomorphic encryption is now on arXiv ✨<br><br>➡️ <a href="https://t.co/NNqyRalxye">https://t.co/NNqyRalxye</a> ⬅️</p>&mdash; Ted, ε-indistinguishable from not being there (@TedOnPrivacy) <a href="https://twitter.com/TedOnPrivacy/status/1405163931212128260?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. An enriched category theory of language: from syntax to semantics

Tai-Danae Bradley, John Terilla, Yiannis Vlassopoulos

- retweets: 80, favorites: 31 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07890) | [pdf](https://arxiv.org/pdf/2106.07890)
- [math.CT](https://arxiv.org/list/math.CT/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Given a piece of text, the ability to generate a coherent extension of it implies some sophistication, including a knowledge of grammar and semantics. In this paper, we propose a mathematical framework for passing from probability distributions on extensions of given texts to an enriched category containing semantic information. Roughly speaking, we model probability distributions on texts as a category enriched over the unit interval. Objects of this category are expressions in language and hom objects are conditional probabilities that one expression is an extension of another. This category is syntactical: it describes what goes with what. We then pass to the enriched category of unit interval-valued copresheaves on this syntactical category to find semantic information.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tai-Danae Bradley, John Terilla, Yiannis Vlassopoulos: An enriched category theory of language: from syntax to semantics <a href="https://t.co/MIyuxUCVIK">https://t.co/MIyuxUCVIK</a> <a href="https://t.co/PFcX4jkIkR">https://t.co/PFcX4jkIkR</a></p>&mdash; arXiv math.CT Category Theory (@mathCTbot) <a href="https://twitter.com/mathCTbot/status/1404979879339171841?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. HUMAP: Hierarchical Uniform Manifold Approximation and Projection

Wilson E. Marcílio-Jr, Danilo M. Eler, Fernando V. Paulovich, Rafael M. Martins

- retweets: 72, favorites: 35 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07718) | [pdf](https://arxiv.org/pdf/2106.07718)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Dimensionality reduction (DR) techniques help analysts to understand patterns in high-dimensional spaces. These techniques, often represented by scatter plots, are employed in diverse science domains and facilitate similarity analysis among clusters and data samples. For datasets containing many granularities or when analysis follows the information visualization mantra, hierarchical DR techniques are the most suitable approach since they present major structures beforehand and details on demand. However, current hierarchical DR techniques are not fully capable of addressing literature problems because they do not preserve the projection mental map across hierarchical levels or are not suitable for most data types. This work presents HUMAP, a novel hierarchical dimensionality reduction technique designed to be flexible on preserving local and global structures and preserve the mental map throughout hierarchical exploration. We provide empirical evidence of our technique's superiority compared with current hierarchical approaches and show two case studies to demonstrate its strengths.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/7) Announcing our recent work, HUMAP. It&#39;s a hierarchical dimensionality reduction technique based on <a href="https://twitter.com/leland_mcinnes?ref_src=twsrc%5Etfw">@leland_mcinnes</a>&#39;s <a href="https://twitter.com/hashtag/UMAP?src=hash&amp;ref_src=twsrc%5Etfw">#UMAP</a>.<br><br>It focuses on high-level information and features drill-down operations for details.<br><br>Preprint: <a href="https://t.co/SVVRvkza3p">https://t.co/SVVRvkza3p</a><br>Code: <a href="https://t.co/UUwrZRZfuQ">https://t.co/UUwrZRZfuQ</a></p>&mdash; Wilson Marcílio-Jr (@EstecioJunior) <a href="https://twitter.com/EstecioJunior/status/1405008759026556930?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. GeoMol: Torsional Geometric Generation of Molecular 3D Conformer  Ensembles

Octavian-Eugen Ganea, Lagnajit Pattanaik, Connor W. Coley, Regina Barzilay, Klavs F. Jensen, William H. Green, Tommi S. Jaakkola

- retweets: 49, favorites: 50 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07802) | [pdf](https://arxiv.org/pdf/2106.07802)
- [physics.chem-ph](https://arxiv.org/list/physics.chem-ph/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Prediction of a molecule's 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. Existing generative models have several drawbacks including lack of modeling important molecular geometry elements (e.g. torsion angles), separate optimization stages prone to error accumulation, and the need for structure fine-tuning based on approximate classical force-fields or computationally expensive methods such as metadynamics with approximate quantum mechanics calculations at each geometry. We propose GeoMol--an end-to-end, non-autoregressive and SE(3)-invariant machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, avoiding unnecessary over-parameterization of the geometric degrees of freedom (e.g. one angle per non-terminal bond). Such local predictions suffice both for the training loss computation, as well as for the full deterministic conformer assembly (at test time). We devise a non-adversarial optimal transport based loss function to promote diverse conformer generation. GeoMol predominantly outperforms popular open-source, commercial, or state-of-the-art machine learning (ML) models, while achieving significant speed-ups. We expect such differentiable 3D structure generators to significantly impact molecular modeling and related applications.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very happy about our recent work on predicting the (low-energy) 3D conformation ensembles of drug-like molecules from their chemical graphs<a href="https://t.co/1Um3drWvFs">https://t.co/1Um3drWvFs</a><br>w/ great collaborators <a href="https://twitter.com/lucky_pattanaik?ref_src=twsrc%5Etfw">@lucky_pattanaik</a>  <a href="https://twitter.com/cwcoley?ref_src=twsrc%5Etfw">@cwcoley</a>. <a href="https://twitter.com/BarzilayRegina?ref_src=twsrc%5Etfw">@BarzilayRegina</a>, W. Green, K. Jensen, T. Jaakkola. 1/2</p>&mdash; Octavian Ganea (@oooctavian) <a href="https://twitter.com/oooctavian/status/1405011104129064962?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. MICo: Learning improved representations via sampling-based state  similarity for Markov decision processes

Pablo Samuel Castro, Tyler Kastner, Prakash Panangaden, Mark Rowland

- retweets: 72, favorites: 27 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.08229) | [pdf](https://arxiv.org/pdf/2106.08229)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We present a new behavioural distance over the state space of a Markov decision process, and demonstrate the use of this distance as an effective means of shaping the learnt representations of deep reinforcement learning agents. While existing notions of state similarity are typically difficult to learn at scale due to high computational cost and lack of sample-based algorithms, our newly-proposed distance addresses both of these issues. In addition to providing detailed theoretical analysis, we provide empirical evidence that learning this distance alongside the value function yields structured and informative representations, including strong results on the Arcade Learning Environment benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Very happy to share our MICo paper!<br><br>We present a new behavioural distance over the state space of an MDP, and show how it can shape the learnt representations of deep RL agents.<br><br>Paper: <a href="https://t.co/MXN0t4ytha">https://t.co/MXN0t4ytha</a><br>Blog: <a href="https://t.co/pw3B1D8XpF">https://t.co/pw3B1D8XpF</a><br>Code: <a href="https://t.co/OGSTOEM51x">https://t.co/OGSTOEM51x</a><br><br>1/🧵 <a href="https://t.co/9zQOtBBvru">pic.twitter.com/9zQOtBBvru</a></p>&mdash; Pablo Samuel Castro (@pcastr) <a href="https://twitter.com/pcastr/status/1405158221074022401?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. Targeted Data Acquisition for Evolving Negotiation Agents

Minae Kwon, Siddharth Karamcheti, Mariano-Florentino Cuellar, Dorsa Sadigh

- retweets: 56, favorites: 33 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07728) | [pdf](https://arxiv.org/pdf/2106.07728)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.MA](https://arxiv.org/list/cs.MA/recent)

Successful negotiators must learn how to balance optimizing for self-interest and cooperation. Yet current artificial negotiation agents often heavily depend on the quality of the static datasets they were trained on, limiting their capacity to fashion an adaptive response balancing self-interest and cooperation. For this reason, we find that these agents can achieve either high utility or cooperation, but not both. To address this, we introduce a targeted data acquisition framework where we guide the exploration of a reinforcement learning agent using annotations from an expert oracle. The guided exploration incentivizes the learning agent to go beyond its static dataset and develop new negotiation strategies. We show that this enables our agents to obtain higher-reward and more Pareto-optimal solutions when negotiating with both simulated and human partners compared to standard supervised learning and reinforcement learning methods. This trend additionally holds when comparing agents using our targeted data acquisition framework to variants of agents trained with a mix of supervised learning and reinforcement learning, or to agents using tailored reward functions that explicitly optimize for utility and Pareto-optimality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share our <a href="https://twitter.com/hashtag/ICML2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICML2021</a> paper “Targeted Data Acquisition for Evolving Negotiation Agents” with the amazing <a href="https://twitter.com/siddkaramcheti?ref_src=twsrc%5Etfw">@siddkaramcheti</a>, Mariano-Florentino Cuéllar, and <a href="https://twitter.com/DorsaSadigh?ref_src=twsrc%5Etfw">@DorsaSadigh</a>!<br><br>Paper: <a href="https://t.co/u0rLPUhxoA">https://t.co/u0rLPUhxoA</a><br>Talk: <a href="https://t.co/Y89SvXUxUS">https://t.co/Y89SvXUxUS</a><br><br>🧵👇 [1/6] <a href="https://t.co/RBD9vC0MhJ">pic.twitter.com/RBD9vC0MhJ</a></p>&mdash; Minae Kwon (@MinaeKwon) <a href="https://twitter.com/MinaeKwon/status/1405200923828244482?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 23. Text Generation with Efficient (Soft) Q-Learning

Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu

- retweets: 49, favorites: 27 (06/17/2021 07:07:13)

- links: [abs](https://arxiv.org/abs/2106.07704) | [pdf](https://arxiv.org/pdf/2106.07704)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Maximum likelihood estimation (MLE) is the predominant algorithm for training text generation models. This paradigm relies on direct supervision examples, which is not applicable to many applications, such as generating adversarial attacks or generating prompts to control language models. Reinforcement learning (RL) on the other hand offers a more flexible solution by allowing users to plug in arbitrary task metrics as reward. Yet previous RL algorithms for text generation, such as policy gradient (on-policy RL) and Q-learning (off-policy RL), are often notoriously inefficient or unstable to train due to the large sequence space and the sparse reward received only at the end of sequences. In this paper, we introduce a new RL formulation for text generation from the soft Q-learning perspective. It further enables us to draw from the latest RL advances, such as path consistency learning, to combine the best of on-/off-policy updates, and learn effectively from sparse reward. We apply the approach to a wide range of tasks, including learning from noisy/negative examples, adversarial attacks, and prompt generation. Experiments show our approach consistently outperforms both task-specialized algorithms and the previous RL methods. On standard supervised tasks where MLE prevails, our approach also achieves competitive performance and stability by training text generation from scratch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Text Generation with Efficient (Soft) Q-Learning<br>pdf: <a href="https://t.co/qLEY3bpzie">https://t.co/qLEY3bpzie</a><br>abs: <a href="https://t.co/Jw1hGfStTP">https://t.co/Jw1hGfStTP</a><br>github: <a href="https://t.co/kppHalMPeb">https://t.co/kppHalMPeb</a> <a href="https://t.co/wFRnDqF3Li">pic.twitter.com/wFRnDqF3Li</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1405173929589805065?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. Communicating Natural Programs to Humans and Machines

Samuel Acquaviva, Yewen Pu, Marta Kryven, Catherine Wong, Gabrielle E Ecanow, Maxwell Nye, Theodoros Sechopoulos, Michael Henry Tessler, Joshua B. Tenenbaum

- retweets: 24, favorites: 45 (06/17/2021 07:07:14)

- links: [abs](https://arxiv.org/abs/2106.07824) | [pdf](https://arxiv.org/pdf/2106.07824)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

The Abstraction and Reasoning Corpus (ARC) is a set of tasks that tests an agent's ability to flexibly solve novel problems. While most ARC tasks are easy for humans, they are challenging for state-of-the-art AI. How do we build intelligent systems that can generalize to novel situations and understand human instructions in domains such as ARC? We posit that the answer may be found by studying how humans communicate to each other in solving these tasks. We present LARC, the Language-annotated ARC: a collection of natural language descriptions by a group of human participants, unfamiliar both with ARC and with each other, who instruct each other on how to solve ARC tasks. LARC contains successful instructions for 88\% of the ARC tasks. We analyze the collected instructions as `natural programs', finding that most natural program concepts have analogies in typical computer programs. However, unlike how one precisely programs a computer, we find that humans both anticipate and exploit ambiguities to communicate effectively. We demonstrate that a state-of-the-art program synthesis technique, which leverages the additional language annotations, outperforms its language-free counterpart.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Communicating Natural Programs to Humans and Machines<br>pdf: <a href="https://t.co/XUqHAxze2d">https://t.co/XUqHAxze2d</a><br>abs: <a href="https://t.co/R28I9OZppw">https://t.co/R28I9OZppw</a> <a href="https://t.co/k3mmN87LKx">pic.twitter.com/k3mmN87LKx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1404968263595999236?ref_src=twsrc%5Etfw">June 16, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



