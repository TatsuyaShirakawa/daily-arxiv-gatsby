---
title: Hot Papers 2021-06-23
date: 2021-06-24T08:07:40.Z
template: "post"
draft: false
slug: "hot-papers-2021-06-23"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-06-23"
socialImage: "/media/flying-marine.jpg"

---

# 1. Revisiting Deep Learning Models for Tabular Data

Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko

- retweets: 2692, favorites: 393 (06/24/2021 08:07:40)

- links: [abs](https://arxiv.org/abs/2106.11959) | [pdf](https://arxiv.org/pdf/2106.11959)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The necessity of deep learning for tabular data is still an unanswered question addressed by a large number of research efforts. The recent literature on tabular DL proposes several deep architectures reported to be superior to traditional "shallow" models like Gradient Boosted Decision Trees. However, since existing works often use different benchmarks and tuning protocols, it is unclear if the proposed models universally outperform GBDT. Moreover, the models are often not compared to each other, therefore, it is challenging to identify the best deep model for practitioners.   In this work, we start from a thorough review of the main families of DL models recently developed for tabular data. We carefully tune and evaluate them on a wide range of datasets and reveal two significant findings. First, we show that the choice between GBDT and DL models highly depends on data and there is still no universally superior solution. Second, we demonstrate that a simple ResNet-like architecture is a surprisingly effective baseline, which outperforms most of the sophisticated models from the DL literature. Finally, we design a simple adaptation of the Transformer architecture for tabular data that becomes a new strong DL baseline and reduces the gap between GBDT and DL models on datasets where GBDT dominates.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Revisiting Deep Learning for Tabular Data<br><br>This recent paper reviews some recent deep learning models developed for tabular data.<br><br>The authors propose a Transformer model for tabular data that achieves state-of-the-art performance among DL solutions.<a href="https://t.co/aMqD0FAzdH">https://t.co/aMqD0FAzdH</a> <a href="https://t.co/c1ozxNrpRo">pic.twitter.com/c1ozxNrpRo</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1407755365593272327?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Revisiting Deep Learning Models for Tabular Data<br>pdf: <a href="https://t.co/G4J8TRyRBt">https://t.co/G4J8TRyRBt</a><br>abs: <a href="https://t.co/tMUhp2IZwW">https://t.co/tMUhp2IZwW</a><br>github: <a href="https://t.co/qeue47mnWD">https://t.co/qeue47mnWD</a><br><br>proposed a attention-based architecture that outperforms ResNet on many tasks <a href="https://t.co/aBCZ3dD4VV">pic.twitter.com/aBCZ3dD4VV</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407499718998081547?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The rate at which &quot;DL for tabular data papers&quot; are uploaded is quite impressive. &quot;Regularization is all you need&quot; was yesterday. Today, we are already &quot;Revisiting Deep Learning Models for Tabular Data&quot; <a href="https://t.co/hqorAVCnD9">https://t.co/hqorAVCnD9</a> <a href="https://t.co/eigfJcLQnY">https://t.co/eigfJcLQnY</a></p>&mdash; Sebastian Raschka (@rasbt) <a href="https://twitter.com/rasbt/status/1407532529633087491?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Randomness In Neural Network Training: Characterizing The Impact of  Tooling

Donglin Zhuang, Xingyao Zhang, Shuaiwen Leon Song, Sara Hooker

- retweets: 2408, favorites: 230 (06/24/2021 08:07:41)

- links: [abs](https://arxiv.org/abs/2106.11872) | [pdf](https://arxiv.org/pdf/2106.11872)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

The quest for determinism in machine learning has disproportionately focused on characterizing the impact of noise introduced by algorithmic design choices. In this work, we address a less well understood and studied question: how does our choice of tooling introduce randomness to deep neural network training. We conduct large scale experiments across different types of hardware, accelerators, state of art networks, and open-source datasets, to characterize how tooling choices contribute to the level of non-determinism in a system, the impact of said non-determinism, and the cost of eliminating different sources of noise.   Our findings are surprising, and suggest that the impact of non-determinism in nuanced. While top-line metrics such as top-1 accuracy are not noticeably impacted, model performance on certain parts of the data distribution is far more sensitive to the introduction of randomness. Our results suggest that deterministic tooling is critical for AI safety. However, we also find that the cost of ensuring determinism varies dramatically between neural network architectures and hardware types, e.g., with overhead up to $746\%$, $241\%$, and $196\%$ on a spectrum of widely used GPU accelerator architectures, relative to non-deterministic training. The source code used in this paper is available at https://github.com/usyd-fsalab/NeuralNetworkRandomness.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How do hardware, software and algorithm  contribute to non-determinism in deep neural networks? <br><br>What is the downstream impact and the cost of ensuring determinism?<br><br>Very excited to share new work led by <a href="https://twitter.com/Donglin07326431?ref_src=twsrc%5Etfw">@Donglin07326431</a> w  Xingyao Zhang, <a href="https://twitter.com/Leon75421958?ref_src=twsrc%5Etfw">@Leon75421958</a> <a href="https://t.co/INnTbe4OEm">https://t.co/INnTbe4OEm</a> <a href="https://t.co/IZcyKb6voW">pic.twitter.com/IZcyKb6voW</a></p>&mdash; Sara Hooker (@sarahookr) <a href="https://twitter.com/sarahookr/status/1407698660670853120?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Dangers of Bayesian Model Averaging under Covariate Shift

Pavel Izmailov, Patrick Nicholson, Sanae Lotfi, Andrew Gordon Wilson

- retweets: 1530, favorites: 254 (06/24/2021 08:07:41)

- links: [abs](https://arxiv.org/abs/2106.11905) | [pdf](https://arxiv.org/pdf/2106.11905)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Approximate Bayesian inference for neural networks is considered a robust alternative to standard training, often providing good performance on out-of-distribution data. However, Bayesian neural networks (BNNs) with high-fidelity approximate inference via full-batch Hamiltonian Monte Carlo achieve poor generalization under covariate shift, even underperforming classical estimation. We explain this surprising result, showing how a Bayesian model average can in fact be problematic under covariate shift, particularly in cases where linear dependencies in the input features cause a lack of posterior contraction. We additionally show why the same issue does not affect many approximate inference procedures, or classical maximum a-posteriori (MAP) training. Finally, we propose novel priors that improve the robustness of BNNs to many sources of covariate shift.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dangers of Bayesian Model Averaging under Covariate Shift <a href="https://t.co/7V4Y15xi6P">https://t.co/7V4Y15xi6P</a><br><br>We show how Bayesian neural nets can generalize *extremely* poorly under covariate shift, why it happens and how to fix it!<br><br>With Patrick Nicholson, <a href="https://twitter.com/LotfiSanae?ref_src=twsrc%5Etfw">@LotfiSanae</a> and <a href="https://twitter.com/andrewgwils?ref_src=twsrc%5Etfw">@andrewgwils</a> <br><br>1/10 <a href="https://t.co/kR0I9YZSog">pic.twitter.com/kR0I9YZSog</a></p>&mdash; Pavel Izmailov (@Pavel_Izmailov) <a href="https://twitter.com/Pavel_Izmailov/status/1407522681516331016?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Despite its popularity in the covariate shift setting, Bayesian model averaging can surprisingly hurt OOD generalization! <a href="https://t.co/AqbH4f29FG">https://t.co/AqbH4f29FG</a> 1/5 <a href="https://t.co/OawZs6AV7n">https://t.co/OawZs6AV7n</a> <a href="https://t.co/SiPHp8mQrm">pic.twitter.com/SiPHp8mQrm</a></p>&mdash; Andrew Gordon Wilson (@andrewgwils) <a href="https://twitter.com/andrewgwils/status/1407524419665338368?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Normalized Avatar Synthesis Using StyleGAN and Perceptual Refinement

Huiwen Luo, Koki Nagano, Han-Wei Kung, Mclean Goldwhite, Qingguo Xu, Zejian Wang, Lingyu Wei, Liwen Hu, Hao Li

- retweets: 650, favorites: 122 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11423) | [pdf](https://arxiv.org/pdf/2106.11423)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

We introduce a highly robust GAN-based framework for digitizing a normalized 3D avatar of a person from a single unconstrained photo. While the input image can be of a smiling person or taken in extreme lighting conditions, our method can reliably produce a high-quality textured model of a person's face in neutral expression and skin textures under diffuse lighting condition. Cutting-edge 3D face reconstruction methods use non-linear morphable face models combined with GAN-based decoders to capture the likeness and details of a person but fail to produce neutral head models with unshaded albedo textures which is critical for creating relightable and animation-friendly avatars for integration in virtual environments. The key challenges for existing methods to work is the lack of training and ground truth data containing normalized 3D faces. We propose a two-stage approach to address this problem. First, we adopt a highly robust normalized 3D face generator by embedding a non-linear morphable face model into a StyleGAN2 network. This allows us to generate detailed but normalized facial assets. This inference is then followed by a perceptual refinement step that uses the generated assets as regularization to cope with the limited available training samples of normalized faces. We further introduce a Normalized Face Dataset, which consists of a combination photogrammetry scans, carefully selected photographs, and generated fake people with neutral expressions in diffuse lighting conditions. While our prepared dataset contains two orders of magnitude less subjects than cutting edge GAN-based 3D facial reconstruction methods, we show that it is possible to produce high-quality normalized face models for very challenging unconstrained input images, and demonstrate superior performance to the current state-of-the-art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Normalized Avatar Synthesis Using StyleGAN and Perceptual Refinement<br>pdf: <a href="https://t.co/e87LOKwylg">https://t.co/e87LOKwylg</a><br><br>StyleGAN2-based digitization approach using a non-linear 3DMM, generates high-quality normalized textured 3D face models from challenging unconstrained input photos <a href="https://t.co/VgZgL4gAbR">pic.twitter.com/VgZgL4gAbR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407505604432699394?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Counterexample to cut-elimination in cyclic proof system for first-order  logic with inductive definitions

Yukihiro Masuoka, Makoto Tatsuta

- retweets: 106, favorites: 23 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11798) | [pdf](https://arxiv.org/pdf/2106.11798)
- [cs.LO](https://arxiv.org/list/cs.LO/recent) | [math.LO](https://arxiv.org/list/math.LO/recent)

A cyclic proof system is a proof system whose proof figure is a tree with cycles. The cut-elimination in a proof system is fundamental. It is conjectured that the cut-elimination in the cyclic proof system for first-order logic with inductive definitions does not hold. This paper shows that the conjecture is correct by giving a sequent not provable without the cut rule but provable in the cyclic proof system.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">My first pre-print......<a href="https://t.co/lWIDaj11Mk">https://t.co/lWIDaj11Mk</a></p>&mdash; YukihiroMASUOKA (@Yukihiro0036) <a href="https://twitter.com/Yukihiro0036/status/1407523159566282755?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Towards Reducing Labeling Cost in Deep Object Detection

Ismail Elezi, Zhiding Yu, Anima Anandkumar, Laura Leal-Taixe, Jose M. Alvarez

- retweets: 72, favorites: 57 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11921) | [pdf](https://arxiv.org/pdf/2106.11921)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep neural networks have reached very high accuracy on object detection but their success hinges on large amounts of labeled data. To reduce the dependency on labels, various active-learning strategies have been proposed, typically based on the confidence of the detector. However, these methods are biased towards best-performing classes and can lead to acquired datasets that are not good representatives of the data in the testing set. In this work, we propose a unified framework for active learning, that considers both the uncertainty and the robustness of the detector, ensuring that the network performs accurately in all classes. Furthermore, our method is able to pseudo-label the very confident predictions, suppressing a potential distribution drift while further boosting the performance of the model. Experiments show that our method comprehensively outperforms a wide range of active-learning methods on PASCAL VOC07+12 and MS-COCO, having up to a 7.7% relative improvement, or up to 82% reduction in labeling cost.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Towards Reducing Labeling Cost in Deep Object Detection<br>pdf: <a href="https://t.co/t9jNnhrw8E">https://t.co/t9jNnhrw8E</a><br><br>consistently outperforms a wide range of active-learning methods, yielding up to a 7.7% relative improvement in mAP, or up to a 82% reduction in labeling cost <a href="https://t.co/JNJauyl8tc">pic.twitter.com/JNJauyl8tc</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407547301200007169?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Unsupervised Object-Level Representation Learning from Scene Images

Jiahao Xie, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, Chen Change Loy

- retweets: 81, favorites: 43 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11952) | [pdf](https://arxiv.org/pdf/2106.11952)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Contrastive self-supervised learning has largely narrowed the gap to supervised pre-training on ImageNet. However, its success highly relies on the object-centric priors of ImageNet, i.e., different augmented views of the same image correspond to the same object. Such a heavily curated constraint becomes immediately infeasible when pre-trained on more complex scene images with many objects. To overcome this limitation, we introduce Object-level Representation Learning (ORL), a new self-supervised learning framework towards scene images. Our key insight is to leverage image-level self-supervised pre-training as the prior to discover object-level semantic correspondence, thus realizing object-level representation learning from scene images. Extensive experiments on COCO show that ORL significantly improves the performance of self-supervised learning on scene images, even surpassing supervised ImageNet pre-training on several downstream tasks. Furthermore, ORL improves the downstream performance when more unlabeled scene images are available, demonstrating its great potential of harnessing unlabeled data in the wild. We hope our approach can motivate future research on more general-purpose unsupervised representation learning from scene data. Project page: https://www.mmlab-ntu.com/project/orl/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Unsupervised Object-Level Representation Learning from Scene Images<br>pdf: <a href="https://t.co/jYccf0MAvu">https://t.co/jYccf0MAvu</a><br>project page: <a href="https://t.co/U1pZ5NtIbe">https://t.co/U1pZ5NtIbe</a><br><br>improves the performance of SSL on scene images,<br>even surpassing supervised ImageNet pre-training on several downstream tasks <a href="https://t.co/9xGVd5uD0J">pic.twitter.com/9xGVd5uD0J</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407506986032369664?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. BARTScore: Evaluating Generated Text as Text Generation

Weizhe Yuan, Graham Neubig, Pengfei Liu

- retweets: 65, favorites: 26 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11520) | [pdf](https://arxiv.org/pdf/2106.11520)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

A wide variety of NLP applications, such as machine translation, summarization, and dialog, involve text generation. One major challenge for these applications is how to evaluate whether such generated texts are actually fluent, accurate, or effective. In this work, we conceptualize the evaluation of generated text as a text generation problem, modeled using pre-trained sequence-to-sequence models. The general idea is that models trained to convert the generated text to/from a reference output or the source text will achieve higher scores when the generated text is better. We operationalize this idea using BART, an encoder-decoder based pre-trained model, and propose a metric BARTScore with a number of variants that can be flexibly applied in an unsupervised fashion to evaluation of text from different perspectives (e.g. informativeness, fluency, or factuality). BARTScore is conceptually simple and empirically effective. It can outperform existing top-scoring metrics in 16 of 22 test settings, covering evaluation of 16 datasets (e.g., machine translation, text summarization) and 7 different perspectives (e.g., informativeness, factuality). Code to calculate BARTScore is available at https://github.com/neulab/BARTScore, and we have released an interactive leaderboard for meta-evaluation at http://explainaboard.nlpedia.ai/leaderboard/task-meval/ on the ExplainaBoard platform, which allows us to interactively understand the strengths, weaknesses, and complementarity of each metric.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">BARTScore: Evaluating Generated Text as Text Generation<br>pdf: <a href="https://t.co/Zr8IXGKkKC">https://t.co/Zr8IXGKkKC</a><br>demo: <a href="https://t.co/n0ZuXsaeO0">https://t.co/n0ZuXsaeO0</a><br>outperforms existing top-scoring metrics in 16 of 22 test settings, covering evaluation of 16 datasets<br>and 7 different perspectives <a href="https://t.co/CH6rYRtZlI">pic.twitter.com/CH6rYRtZlI</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407511951316504576?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. DeepMesh: Differentiable Iso-Surface Extraction

Benoit Guillard, Edoardo Remelli, Artem Lukoianov, Stephan Richter, Timur Bagautdinov, Pierre Baque, Pascal Fua

- retweets: 49, favorites: 33 (06/24/2021 08:07:42)

- links: [abs](https://arxiv.org/abs/2106.11795) | [pdf](https://arxiv.org/pdf/2106.11795)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Geometric Deep Learning has recently made striking progress with the advent of continuous Deep Implicit Fields. They allow for detailed modeling of watertight surfaces of arbitrary topology while not relying on a 3D Euclidean grid, resulting in a learnable parameterization that is unlimited in resolution. Unfortunately, these methods are often unsuitable for applications that require an explicit mesh-based surface representation because converting an implicit field to such a representation relies on the Marching Cubes algorithm, which cannot be differentiated with respect to the underlying implicit field. In this work, we remove this limitation and introduce a differentiable way to produce explicit surface mesh representations from Deep Implicit Fields. Our key insight is that by reasoning on how implicit field perturbations impact local surface geometry, one can ultimately differentiate the 3D location of surface samples with respect to the underlying deep implicit field. We exploit this to define DeepMesh -- end-to-end differentiable mesh representation that can vary its topology. We use two different applications to validate our theoretical insight: Single view 3D Reconstruction via Differentiable Rendering and Physically-Driven Shape Optimization. In both cases our end-to-end differentiable parameterization gives us an edge over state-of-the-art algorithms.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We can parametrize differentiably 3D meshes whose topology can change. In earlier work, we showed this for meshes represented by SDFs. We have now extended this result to a much broader class of implicit functions. <a href="https://t.co/g8vywEYqpi">https://t.co/g8vywEYqpi</a> <a href="https://twitter.com/hashtag/deeplearning?src=hash&amp;ref_src=twsrc%5Etfw">#deeplearning</a> <a href="https://twitter.com/hashtag/computervision?src=hash&amp;ref_src=twsrc%5Etfw">#computervision</a> <a href="https://t.co/QCmA97mb1H">pic.twitter.com/QCmA97mb1H</a></p>&mdash; Pascal Fua (@FuaPv) <a href="https://twitter.com/FuaPv/status/1407640311673786369?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. DocFormer: End-to-End Transformer for Document Understanding

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, R. Manmatha

- retweets: 49, favorites: 32 (06/24/2021 08:07:43)

- links: [abs](https://arxiv.org/abs/2106.11539) | [pdf](https://arxiv.org/pdf/2106.11539)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present DocFormer -- a multi-modal transformer based architecture for the task of Visual Document Understanding (VDU). VDU is a challenging problem which aims to understand documents in their varied formats (forms, receipts etc.) and layouts. In addition, DocFormer is pre-trained in an unsupervised fashion using carefully designed tasks which encourage multi-modal interaction. DocFormer uses text, vision and spatial features and combines them using a novel multi-modal self-attention layer. DocFormer also shares learned spatial embeddings across modalities which makes it easy for the model to correlate text to visual tokens and vice versa. DocFormer is evaluated on 4 different datasets each with strong baselines. DocFormer achieves state-of-the-art results on all of them, sometimes beating models 4x its size (in no. of parameters).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DocFormer: End-to-End Transformer for Document Understanding<br>pdf: <a href="https://t.co/GwVkCjUMVl">https://t.co/GwVkCjUMVl</a><br>abs: <a href="https://t.co/BnY2nnegdC">https://t.co/BnY2nnegdC</a><br><br>a multi-modal end-to-end trainable transformer based model for various Visual Document Understanding tasks <a href="https://t.co/ZbOlxOij6i">pic.twitter.com/ZbOlxOij6i</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1407503383741800452?ref_src=twsrc%5Etfw">June 23, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



