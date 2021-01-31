---
title: Hot Papers 2021-01-29
date: 2021-01-31T09:38:27.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-29"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-29"
socialImage: "/media/flying-marine.jpg"

---

# 1. TT-Rec: Tensor Train Compression for Deep Learning Recommendation Models

Chunxing Yin, Bilge Acun, Xing Liu, Carole-Jean Wu

- retweets: 11972, favorites: 0 (01/31/2021 09:38:27)

- links: [abs](https://arxiv.org/abs/2101.11714) | [pdf](https://arxiv.org/pdf/2101.11714)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

The memory capacity of embedding tables in deep learning recommendation models (DLRMs) is increasing dramatically from tens of GBs to TBs across the industry. Given the fast growth in DLRMs, novel solutions are urgently needed, in order to enable fast and efficient DLRM innovations. At the same time, this must be done without having to exponentially increase infrastructure capacity demands. In this paper, we demonstrate the promising potential of Tensor Train decomposition for DLRMs (TT-Rec), an important yet under-investigated context. We design and implement optimized kernels (TT-EmbeddingBag) to evaluate the proposed TT-Rec design. TT-EmbeddingBag is 3 times faster than the SOTA TT implementation. The performance of TT-Rec is further optimized with the batched matrix multiplication and caching strategies for embedding vector lookup operations. In addition, we present mathematically and empirically the effect of weight initialization distribution on DLRM accuracy and propose to initialize the tensor cores of TT-Rec following the sampled Gaussian distribution. We evaluate TT-Rec across three important design space dimensions -- memory capacity, accuracy, and timing performance -- by training MLPerf-DLRM with Criteo's Kaggle and Terabyte data sets. TT-Rec achieves 117 times and 112 times model size compression, for Kaggle and Terabyte, respectively. This impressive model size reduction can come with no accuracy nor training time overhead as compared to the uncompressed baseline.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Introducing TT-Rec, a new method to dramatically shrink memory-intensive Deep Learning Recommendation Models and make them easier to deploy at scale. <a href="https://t.co/IiGmIakkZZ">https://t.co/IiGmIakkZZ</a> <a href="https://t.co/pjtGhZhOG3">pic.twitter.com/pjtGhZhOG3</a></p>&mdash; Facebook AI (@facebookai) <a href="https://twitter.com/facebookai/status/1355261469282209799?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Playable Video Generation

Willi Menapace, Stéphane Lathuilière, Sergey Tulyakov, Aliaksandr Siarohin, Elisa Ricci

- retweets: 1597, favorites: 195 (01/31/2021 09:38:27)

- links: [abs](https://arxiv.org/abs/2101.12195) | [pdf](https://arxiv.org/pdf/2101.12195)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

This paper introduces the unsupervised learning problem of playable video generation (PVG). In PVG, we aim at allowing a user to control the generated video by selecting a discrete action at every time step as when playing a video game. The difficulty of the task lies both in learning semantically consistent actions and in generating realistic videos conditioned on the user input. We propose a novel framework for PVG that is trained in a self-supervised manner on a large dataset of unlabelled videos. We employ an encoder-decoder architecture where the predicted action labels act as bottleneck. The network is constrained to learn a rich action space using, as main driving loss, a reconstruction loss on the generated video. We demonstrate the effectiveness of the proposed approach on several datasets with wide environment variety. Further details, code and examples are available on our project page willi-menapace.github.io/playable-video-generation-website.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Playable Video Generation<br>pdf: <a href="https://t.co/oG8e19Kwng">https://t.co/oG8e19Kwng</a><br>abs: <a href="https://t.co/UrpiREYGOi">https://t.co/UrpiREYGOi</a><br>project page: <a href="https://t.co/U3bL8cl6g3">https://t.co/U3bL8cl6g3</a><br>github: <a href="https://t.co/qJ8jsNsAoE">https://t.co/qJ8jsNsAoE</a><br>demo: <a href="https://t.co/TT4ZisXZKN">https://t.co/TT4ZisXZKN</a> <a href="https://t.co/cSHfC3Qxga">pic.twitter.com/cSHfC3Qxga</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1355006110927892485?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. On the Origin of Implicit Regularization in Stochastic Gradient Descent

Samuel L. Smith, Benoit Dherin, David G. T. Barrett, Soham De

- retweets: 580, favorites: 158 (01/31/2021 09:38:27)

- links: [abs](https://arxiv.org/abs/2101.12176) | [pdf](https://arxiv.org/pdf/2101.12176)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

For infinitesimal learning rates, stochastic gradient descent (SGD) follows the path of gradient flow on the full batch loss function. However moderately large learning rates can achieve higher test accuracies, and this generalization benefit is not explained by convergence bounds, since the learning rate which maximizes test accuracy is often larger than the learning rate which minimizes training loss. To interpret this phenomenon we prove that for SGD with random shuffling, the mean SGD iterate also stays close to the path of gradient flow if the learning rate is small and finite, but on a modified loss. This modified loss is composed of the original loss function and an implicit regularizer, which penalizes the norms of the minibatch gradients. Under mild assumptions, when the batch size is small the scale of the implicit regularization term is proportional to the ratio of the learning rate to the batch size. We verify empirically that explicitly including the implicit regularizer in the loss can enhance the test accuracy when the learning rate is small.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The Stochastic Gradient Descent we use in practice, SGD with Random Shuffling, is not a Stochastic Differential Equation when the learning rate is small. <br><br>Instead, it follows the path of gradient flow on a regularized loss: <a href="https://t.co/qaegQKTm1A">https://t.co/qaegQKTm1A</a><br><br>(Mea Culpa at ICLR 2021)</p>&mdash; Samuel L Smith (@SamuelMLSmith) <a href="https://twitter.com/SamuelMLSmith/status/1355103454650372096?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. DOC2PPT: Automatic Presentation Slides Generation from Scientific  Documents

Tsu-Jui Fu, William Yang Wang, Daniel McDuff, Yale Song

- retweets: 538, favorites: 109 (01/31/2021 09:38:27)

- links: [abs](https://arxiv.org/abs/2101.11796) | [pdf](https://arxiv.org/pdf/2101.11796)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Creating presentation materials requires complex multimodal reasoning skills to summarize key concepts and arrange them in a logical and visually pleasing manner. Can machines learn to emulate this laborious process? We present a novel task and approach for document-to-slide generation. Solving this involves document summarization, image and text retrieval, slide structure, and layout prediction to arrange key elements in a form suitable for presentation. We propose a hierarchical sequence-to-sequence approach to tackle our task in an end-to-end manner. Our approach exploits the inherent structures within documents and slides and incorporates paraphrasing and layout prediction modules to generate slides. To help accelerate research in this domain, we release a dataset about 6K paired documents and slide decks used in our experiments. We show that our approach outperforms strong baselines and produces slides with rich content and aligned imagery.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents<br>pdf: <a href="https://t.co/idlfLBQTVQ">https://t.co/idlfLBQTVQ</a><br>abs: <a href="https://t.co/auily8fH6P">https://t.co/auily8fH6P</a><br>project page: <a href="https://t.co/CDtIu151hS">https://t.co/CDtIu151hS</a> <a href="https://t.co/mzZmTJn4q3">pic.twitter.com/mzZmTJn4q3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1354991755721117697?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents. <a href="https://t.co/70Y7x2eVFd">https://t.co/70Y7x2eVFd</a> <a href="https://t.co/lqcta3qOrL">pic.twitter.com/lqcta3qOrL</a></p>&mdash; arxiv (@arxiv_org) <a href="https://twitter.com/arxiv_org/status/1355488219505192962?ref_src=twsrc%5Etfw">January 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Automated femur segmentation from computed tomography images using a  deep neural network

P.A. Bjornsson, B. Helgason, H. Palsson, S. Sigurdsson, V. Gudnason, L.M. Ellingsen

- retweets: 587, favorites: 23 (01/31/2021 09:38:28)

- links: [abs](https://arxiv.org/abs/2101.11742) | [pdf](https://arxiv.org/pdf/2101.11742)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Osteoporosis is a common bone disease that occurs when the creation of new bone does not keep up with the loss of old bone, resulting in increased fracture risk. Adults over the age of 50 are especially at risk and see their quality of life diminished because of limited mobility, which can lead to isolation and depression. We are developing a robust screening method capable of identifying individuals predisposed to hip fracture to address this clinical challenge. The method uses finite element analysis and relies on segmented computed tomography (CT) images of the hip. Presently, the segmentation of the proximal femur requires manual input, which is a tedious task, prone to human error, and severely limits the practicality of the method in a clinical context. Here we present a novel approach for segmenting the proximal femur that uses a deep convolutional neural network to produce accurate, automated, robust, and fast segmentations of the femur from CT scans. The network architecture is based on the renowned u-net, which consists of a downsampling path to extract increasingly complex features of the input patch and an upsampling path to convert the acquired low resolution image into a high resolution one. Skipped connections allow us to recover critical spatial information lost during downsampling. The model was trained on 30 manually segmented CT images and was evaluated on 200 ground truth manual segmentations. Our method delivers a mean Dice similarity coefficient (DSC) and 95th percentile Hausdorff distance (HD95) of 0.990 and 0.981 mm, respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Automated femur segmentation from computed tomography images using a deep neural network.<a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> <a href="https://twitter.com/hashtag/ML?src=hash&amp;ref_src=twsrc%5Etfw">#ML</a> <a href="https://twitter.com/hashtag/BigData?src=hash&amp;ref_src=twsrc%5Etfw">#BigData</a> <a href="https://twitter.com/hashtag/Analytics?src=hash&amp;ref_src=twsrc%5Etfw">#Analytics</a> <a href="https://twitter.com/hashtag/Python?src=hash&amp;ref_src=twsrc%5Etfw">#Python</a> <a href="https://twitter.com/hashtag/Rstats?src=hash&amp;ref_src=twsrc%5Etfw">#Rstats</a> <a href="https://twitter.com/hashtag/100DaysOfCode?src=hash&amp;ref_src=twsrc%5Etfw">#100DaysOfCode</a> <a href="https://twitter.com/hashtag/devcommunity?src=hash&amp;ref_src=twsrc%5Etfw">#devcommunity</a> <a href="https://twitter.com/hashtag/linux?src=hash&amp;ref_src=twsrc%5Etfw">#linux</a> <a href="https://twitter.com/hashtag/serverless?src=hash&amp;ref_src=twsrc%5Etfw">#serverless</a> <a href="https://twitter.com/hashtag/iot?src=hash&amp;ref_src=twsrc%5Etfw">#iot</a> <a href="https://twitter.com/hashtag/womenwhocode?src=hash&amp;ref_src=twsrc%5Etfw">#womenwhocode</a> <a href="https://twitter.com/hashtag/programming?src=hash&amp;ref_src=twsrc%5Etfw">#programming</a> <a href="https://twitter.com/hashtag/DataScience?src=hash&amp;ref_src=twsrc%5Etfw">#DataScience</a> <a href="https://twitter.com/hashtag/MachineLearning?src=hash&amp;ref_src=twsrc%5Etfw">#MachineLearning</a> <a href="https://twitter.com/hashtag/AI?src=hash&amp;ref_src=twsrc%5Etfw">#AI</a><a href="https://t.co/8NOW5kItic">https://t.co/8NOW5kItic</a> <a href="https://t.co/DTALk2Dl21">pic.twitter.com/DTALk2Dl21</a></p>&mdash; Marcus Borba (@marcusborba) <a href="https://twitter.com/marcusborba/status/1355500120587034626?ref_src=twsrc%5Etfw">January 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Tokens-to-Token ViT: Training Vision Transformers from Scratch on  ImageNet

Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Francis EH Tay, Jiashi Feng, Shuicheng Yan

- retweets: 240, favorites: 95 (01/31/2021 09:38:28)

- links: [abs](https://arxiv.org/abs/2101.11986) | [pdf](https://arxiv.org/pdf/2101.11986)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Transformers, which are popular for language modeling, have been explored for solving vision tasks recently, e.g., the Vision Transformers (ViT) for image classification. The ViT model splits each image into a sequence of tokens with fixed length and then applies multiple Transformer layers to model their global relation for classification. However, ViT achieves inferior performance compared with CNNs when trained from scratch on a midsize dataset (e.g., ImageNet). We find it is because: 1) the simple tokenization of input images fails to model the important local structure (e.g., edges, lines) among neighboring pixels, leading to its low training sample efficiency; 2) the redundant attention backbone design of ViT leads to limited feature richness in fixed computation budgets and limited training samples.   To overcome such limitations, we propose a new Tokens-To-Token Vision Transformers (T2T-ViT), which introduces 1) a layer-wise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure presented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformers motivated by CNN architecture design after extensive study. Notably, T2T-ViT reduces the parameter counts and MACs of vanilla ViT by 200\%, while achieving more than 2.5\% improvement when trained from scratch on ImageNet. It also outperforms ResNets and achieves comparable performance with MobileNets when directly training on ImageNet. For example, T2T-ViT with ResNet50 comparable size can achieve 80.7\% top-1 accuracy on ImageNet. (Code: https://github.com/yitu-opensource/T2T-ViT)

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet<br>pdf: <a href="https://t.co/iI0BFr1BjI">https://t.co/iI0BFr1BjI</a><br>abs: <a href="https://t.co/cmanacFhth">https://t.co/cmanacFhth</a><br>github: <a href="https://t.co/4iAQLsRnLV">https://t.co/4iAQLsRnLV</a> <a href="https://t.co/pfxaI1jjWS">pic.twitter.com/pfxaI1jjWS</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1354971643471396864?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. The Role of Syntactic Planning in Compositional Image Captioning

Emanuele Bugliarello, Desmond Elliott

- retweets: 156, favorites: 41 (01/31/2021 09:38:28)

- links: [abs](https://arxiv.org/abs/2101.11911) | [pdf](https://arxiv.org/pdf/2101.11911)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Image captioning has focused on generalizing to images drawn from the same distribution as the training set, and not to the more challenging problem of generalizing to different distributions of images. Recently, Nikolaus et al. (2019) introduced a dataset to assess compositional generalization in image captioning, where models are evaluated on their ability to describe images with unseen adjective-noun and noun-verb compositions. In this work, we investigate different methods to improve compositional generalization by planning the syntactic structure of a caption. Our experiments show that jointly modeling tokens and syntactic tags enhances generalization in both RNN- and Transformer-based models, while also improving performance on standard metrics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we use syntax to improve generalization in image captioning models? Yes!<br>Check out our (w/ <a href="https://twitter.com/delliott?ref_src=twsrc%5Etfw">@delliott</a>) new paper:<br><br>“The Role of Syntactic Planning in Compositional Image Captioning”<br><br>📄 <a href="https://t.co/OAZy8473jW">https://t.co/OAZy8473jW</a><br>🗣️ to appear at <a href="https://twitter.com/hashtag/EACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#EACL2021</a> <a href="https://t.co/ztFju0Jyxz">pic.twitter.com/ztFju0Jyxz</a></p>&mdash; Emanuele Bugliarello (@ebugliarello) <a href="https://twitter.com/ebugliarello/status/1355262106396983298?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. WallStreetBets: Positions or Ban

Christian Boylston, Beatriz Palacios, Plamen Tassev, Amy Bruckman

- retweets: 144, favorites: 49 (01/31/2021 09:38:28)

- links: [abs](https://arxiv.org/abs/2101.12110) | [pdf](https://arxiv.org/pdf/2101.12110)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

r/wallstreetbets (WallStreetBets or WSB) is a subreddit devoted to irreverent memes and high-risk options trading. As of March 30, 2020, the subreddit boasts a usership of nearly 1.1 millions subscribers and self-describes as "if 4chan found a Bloomberg terminal." This paper will utilize Amy Jo Kim's community design principles along with social psychology theory as frameworks to understand how this chaotic, oftentimes offensive community has developed one of the largest and most loyal user bases on the platform. We will further argue that humor plays a vital role in promoting in-group cohesion and in providing an unconventional third place for traders (and thinly veiled gamblers) to seek support from each other in the form of vulgar, yet good-humored taunting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share: a qualitative study of r/wallstreetbets, class project from Design of Online Communities class 2020 by <a href="https://twitter.com/beapalaces?ref_src=twsrc%5Etfw">@beapalaces</a> Christian Boylston and Plamen Tassev. (Not peer reviewed, but got an A!) <a href="https://t.co/VgCdwsm2AJ">https://t.co/VgCdwsm2AJ</a></p>&mdash; Amy Bruckman (@asbruckman) <a href="https://twitter.com/asbruckman/status/1355141353261441024?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. Object Detection Made Simpler by Eliminating Heuristic NMS

Qiang Zhou, Chaohui Yu, Chunhua Shen, Zhibin Wang, Hao Li

- retweets: 103, favorites: 41 (01/31/2021 09:38:28)

- links: [abs](https://arxiv.org/abs/2101.11782) | [pdf](https://arxiv.org/pdf/2101.11782)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We show a simple NMS-free, end-to-end object detection framework, of which the network is a minimal modification to a one-stage object detector such as the FCOS detection model [Tian et al. 2019]. We attain on par or even improved detection accuracy compared with the original one-stage detector. It performs detection at almost the same inference speed, while being even simpler in that now the post-processing NMS (non-maximum suppression) is eliminated during inference. If the network is capable of identifying only one positive sample for prediction for each ground-truth object instance in an image, then NMS would become unnecessary. This is made possible by attaching a compact PSS head for automatic selection of the single positive sample for each instance (see Fig. 1). As the learning objective involves both one-to-many and one-to-one label assignments, there is a conflict in the labels of some training examples, making the learning challenging. We show that by employing a stop-gradient operation, we can successfully tackle this issue and train the detector. On the COCO dataset, our simple design achieves superior performance compared to both the FCOS baseline detector with NMS post-processing and the recent end-to-end NMS-free detectors. Our extensive ablation studies justify the rationale of the design choices.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Object Detection Made Simpler by Eliminating Heuristic NMS<a href="https://t.co/GEcR0MJuMx">https://t.co/GEcR0MJuMx</a> <a href="https://t.co/ya8KrE7Bup">pic.twitter.com/ya8KrE7Bup</a></p>&mdash; phalanx (@ZFPhalanx) <a href="https://twitter.com/ZFPhalanx/status/1354987990808883217?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. A Machine Learning Challenge for Prognostic Modelling in Head and Neck  Cancer Using Multi-modal Data

Michal Kazmierski, Mattea Welch, Sejin Kim, Chris McIntosh, Princess Margaret Head, Neck Cancer Group, Katrina Rey-McIntyre, Shao Hui Huang, Tirth Patel, Tony Tadic, Michael Milosevic, Fei-Fei Liu, Andrew Hope, Scott Bratman, Benjamin Haibe-Kains

- retweets: 114, favorites: 26 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11935) | [pdf](https://arxiv.org/pdf/2101.11935)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Accurate prognosis for an individual patient is a key component of precision oncology. Recent advances in machine learning have enabled the development of models using a wider range of data, including imaging. Radiomics aims to extract quantitative predictive and prognostic biomarkers from routine medical imaging, but evidence for computed tomography radiomics for prognosis remains inconclusive. We have conducted an institutional machine learning challenge to develop an accurate model for overall survival prediction in head and neck cancer using clinical data etxracted from electronic medical records and pre-treatment radiological images, as well as to evaluate the true added benefit of radiomics for head and neck cancer prognosis. Using a large, retrospective dataset of 2,552 patients and a rigorous evaluation framework, we compared 12 different submissions using imaging and clinical data, separately or in combination. The winning approach used non-linear, multitask learning on clinical data and tumour volume, achieving high prognostic accuracy for 2-year and lifetime survival prediction and outperforming models relying on clinical data only, engineered radiomics and deep learning. Combining all submissions in an ensemble model resulted in improved accuracy, with the highest gain from a image-based deep learning model. Our results show the potential of machine learning and simple, informative prognostic factors in combination with large datasets as a tool to guide personalized cancer care.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Latest <a href="https://twitter.com/bhklab?ref_src=twsrc%5Etfw">@bhklab</a> <a href="https://twitter.com/hashtag/Radiomics?src=hash&amp;ref_src=twsrc%5Etfw">#Radiomics</a> study describing a massive head-and-neck cancer dataset from <a href="https://twitter.com/RadMedPM?ref_src=twsrc%5Etfw">@RadMedPM</a>  <a href="https://twitter.com/pmcancercentre?ref_src=twsrc%5Etfw">@pmcancercentre</a>  <a href="https://twitter.com/UHN?ref_src=twsrc%5Etfw">@UHN</a> <a href="https://twitter.com/UofT?ref_src=twsrc%5Etfw">@UofT</a> and its use to develop the best prognostic modelling approach using engineered and <a href="https://twitter.com/hashtag/ArtificialIntelligence?src=hash&amp;ref_src=twsrc%5Etfw">#ArtificialIntelligence</a> <a href="https://twitter.com/hashtag/DeepLearning?src=hash&amp;ref_src=twsrc%5Etfw">#DeepLearning</a> features<a href="https://t.co/P9LkkisJCI">https://t.co/P9LkkisJCI</a> <a href="https://t.co/eCmAptEVNP">pic.twitter.com/eCmAptEVNP</a></p>&mdash; Benjamin Haibe-Kains (@bhaibeka) <a href="https://twitter.com/bhaibeka/status/1355205415802433536?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Website Fingerprinting on Early QUIC Traffic

Pengwei Zhan, Liming Wang, Yi Tang

- retweets: 63, favorites: 75 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11871) | [pdf](https://arxiv.org/pdf/2101.11871)
- [cs.CR](https://arxiv.org/list/cs.CR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NI](https://arxiv.org/list/cs.NI/recent)

Cryptographic protocols have been widely used to protect the user's privacy and avoid exposing private information. QUIC (Quick UDP Internet Connections), as an alternative to traditional HTTP, demonstrates its unique transmission characteristics: based on UDP for encrypted resource transmission, accelerating web page rendering. However, existing encrypted transmission schemes based on TCP are vulnerable to website fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited websites by eavesdropping on the transmission channel. Whether QUIC protocol can effectively resisting to such attacks is worth investigating. In this work, we demonstrated the extreme vulnerability of QUIC under WFP attacks by comparing attack results under well-designed conditions. We also study the transferability of features, which enable the adversary to use proven effective features on a special protocol attacking a new protocol. This study shows that QUIC is more vulnerable to WFP attacks than HTTPS in the early traffic scenario but is similar in the normal scenario. The maximum attack accuracy on QUIC is 56.8 % and 73 % higher than on HTTPS utilizing Simple features and Transfer features. The insecurity characteristic of QUIC explains the dramatic gap. We also find that features are transferable between protocols, and the feature importance is partially inherited on normal traffic due to the relatively fixed browser rendering sequence and the similar request-response model of protocols. However, the transferability is inefficient when on early traffic, as QUIC and HTTPS show significantly different vulnerability when considering early traffic. We also show that attack accuracy on QUIC could reach 95.4 % with only 40 packets and just using simple features, whereas only 60.7 % when on HTTPS.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">QUIC much easier to fingerprint than HTTPS <a href="https://t.co/nACtOXukIc">https://t.co/nACtOXukIc</a></p>&mdash; Hacker News (@newsycombinator) <a href="https://twitter.com/newsycombinator/status/1355562278939205634?ref_src=twsrc%5Etfw">January 30, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Disembodied Machine Learning: On the Illusion of Objectivity in NLP

Zeerak Waseem, Smarika Lulz, Joachim Bingel, Isabelle Augenstein

- retweets: 82, favorites: 22 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11974) | [pdf](https://arxiv.org/pdf/2101.11974)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Machine Learning seeks to identify and encode bodies of knowledge within provided datasets. However, data encodes subjective content, which determines the possible outcomes of the models trained on it. Because such subjectivity enables marginalisation of parts of society, it is termed (social) `bias' and sought to be removed. In this paper, we contextualise this discourse of bias in the ML community against the subjective choices in the development process. Through a consideration of how choices in data and model development construct subjectivity, or biases that are represented in a model, we argue that addressing and mitigating biases is near-impossible. This is because both data and ML models are objects for which meaning is made in each step of the development pipeline, from data selection over annotation to model training and analysis. Accordingly, we find the prevalent discourse of bias limiting in its ability to address social marginalisation. We recommend to be conscientious of this, and to accept that de-biasing methods only correct for a fraction of biases.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Many papers on <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> ethics, including the famous stochastic parrots 🦜,  only describe what the problems are and do not explicate their normative and philosophical assumptions. Not these folks <a href="https://t.co/Q5C4JFqWXx">https://t.co/Q5C4JFqWXx</a>  <a href="https://twitter.com/ZeerakW?ref_src=twsrc%5Etfw">@ZeerakW</a> <a href="https://twitter.com/IAugenstein?ref_src=twsrc%5Etfw">@IAugenstein</a> 👏1/3</p>&mdash; Jindřich Libovický (@jlibovicky) <a href="https://twitter.com/jlibovicky/status/1355180966273298433?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Syntactic Nuclei in Dependency Parsing -- A Multilingual Exploration

Ali Basirat, Joakim Nivre

- retweets: 49, favorites: 28 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11959) | [pdf](https://arxiv.org/pdf/2101.11959)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Standard models for syntactic dependency parsing take words to be the elementary units that enter into dependency relations. In this paper, we investigate whether there are any benefits from enriching these models with the more abstract notion of nucleus proposed by Tesni\`{e}re. We do this by showing how the concept of nucleus can be defined in the framework of Universal Dependencies and how we can use composition functions to make a transition-based dependency parser aware of this concept. Experiments on 12 languages show that nucleus composition gives small but significant improvements in parsing accuracy. Further analysis reveals that the improvement mainly concerns a small number of dependency relations, including nominal modifiers, relations of coordination, main predicates, and direct objects.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to share two of our papers accepted to <a href="https://twitter.com/hashtag/EACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#EACL2021</a>: Attention Can Reflect Syntactic Structure (If You Let It) (<a href="https://t.co/ID8Xris1AI">https://t.co/ID8Xris1AI</a>) and Syntactic Nuclei in Dependency Parsing -- A Multilingual Exploration (<a href="https://t.co/5LmfpfIfDp">https://t.co/5LmfpfIfDp</a>). See you all there!</p>&mdash; Uppsala NLP (@uppsala_nlp) <a href="https://twitter.com/uppsala_nlp/status/1355132298287599618?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Does Typological Blinding Impede Cross-Lingual Sharing?

Johannes Bjerva, Isabelle Augenstein

- retweets: 30, favorites: 38 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11888) | [pdf](https://arxiv.org/pdf/2101.11888)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Bridging the performance gap between high- and low-resource languages has been the focus of much previous work. Typological features from databases such as the World Atlas of Language Structures (WALS) are a prime candidate for this, as such data exists even for very low-resource languages. However, previous work has only found minor benefits from using typological information. Our hypothesis is that a model trained in a cross-lingual setting will pick up on typological cues from the input data, thus overshadowing the utility of explicitly using such features. We verify this hypothesis by blinding a model to typological information, and investigate how cross-lingual sharing and performance is impacted. Our model is based on a cross-lingual architecture in which the latent weights governing the sharing between languages is learnt during training. We show that (i) preventing this model from exploiting typology severely reduces performance, while a control experiment reaffirms that (ii) encouraging sharing according to typology somewhat improves performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Looking forward to sharing some joint work with <a href="https://twitter.com/IAugenstein?ref_src=twsrc%5Etfw">@IAugenstein</a> <a href="https://twitter.com/CopeNLU?ref_src=twsrc%5Etfw">@CopeNLU</a>  on computational typology at <a href="https://twitter.com/eaclmeeting?ref_src=twsrc%5Etfw">@eaclmeeting</a> <a href="https://twitter.com/hashtag/EACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#EACL2021</a> ! We show that &#39;blinding&#39; a model to task-relevant typological features, can affect cross-lingual NLP performance <a href="https://t.co/yo12tDlsUj">https://t.co/yo12tDlsUj</a> <a href="https://t.co/EOuYZeFucJ">pic.twitter.com/EOuYZeFucJ</a></p>&mdash; Johannes Bjerva (@johannesbjerva) <a href="https://twitter.com/johannesbjerva/status/1355048725064937475?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Multi-Modal Aesthetic Assessment for MObile Gaming Image

Zhenyu Lei, Yejing Xie, Suiyi Ling, Andreas Pastor, Junle Wang, Patrick Le Callet

- retweets: 42, favorites: 18 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.11700) | [pdf](https://arxiv.org/pdf/2101.11700)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

With the proliferation of various gaming technology, services, game styles, and platforms, multi-dimensional aesthetic assessment of the gaming contents is becoming more and more important for the gaming industry. Depending on the diverse needs of diversified game players, game designers, graphical developers, etc. in particular conditions, multi-modal aesthetic assessment is required to consider different aesthetic dimensions/perspectives. Since there are different underlying relationships between different aesthetic dimensions, e.g., between the `Colorfulness' and `Color Harmony', it could be advantageous to leverage effective information attached in multiple relevant dimensions. To this end, we solve this problem via multi-task learning. Our inclination is to seek and learn the correlations between different aesthetic relevant dimensions to further boost the generalization performance in predicting all the aesthetic dimensions. Therefore, the `bottleneck' of obtaining good predictions with limited labeled data for one individual dimension could be unplugged by harnessing complementary sources of other dimensions, i.e., augment the training data indirectly by sharing training information across dimensions. According to experimental results, the proposed model outperforms state-of-the-art aesthetic metrics significantly in predicting four gaming aesthetic dimensions.




# 16. Vx2Text: End-to-End Learning of Video-Based Text Generation From  Multimodal Inputs

Xudong Lin, Gedas Bertasius, Jue Wang, Shih-Fu Chang, Devi Parikh, Lorenzo Torresani

- retweets: 24, favorites: 35 (01/31/2021 09:38:29)

- links: [abs](https://arxiv.org/abs/2101.12059) | [pdf](https://arxiv.org/pdf/2101.12059)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

We present \textsc{Vx2Text}, a framework for text generation from multimodal inputs consisting of video plus text, speech, or audio. In order to leverage transformer networks, which have been shown to be effective at modeling language, each modality is first converted into a set of language embeddings by a learnable tokenizer. This allows our approach to perform multimodal fusion in the language space, thus eliminating the need for ad-hoc cross-modal fusion modules. To address the non-differentiability of tokenization on continuous inputs (e.g., video or audio), we utilize a relaxation scheme that enables end-to-end training. Furthermore, unlike prior encoder-only models, our network includes an autoregressive decoder to generate open-ended text from the multimodal embeddings fused by the language encoder. This renders our approach fully generative and makes it directly applicable to different "video+$x$ to text" problems without the need to design specialized network heads for each task. The proposed framework is not only conceptually simple but also remarkably effective: experiments demonstrate that our approach based on a single architecture outperforms the state-of-the-art on three video-based text-generation tasks -- captioning, question answering and audio-visual scene-aware dialog.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs<br>pdf: <a href="https://t.co/NiyI5u8ong">https://t.co/NiyI5u8ong</a><br>abs: <a href="https://t.co/Y1f8zHmkOf">https://t.co/Y1f8zHmkOf</a> <a href="https://t.co/59GuKOLoc2">pic.twitter.com/59GuKOLoc2</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1354977762776383489?ref_src=twsrc%5Etfw">January 29, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



