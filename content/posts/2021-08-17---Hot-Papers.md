---
title: Hot Papers 2021-08-17
date: 2021-08-18T07:33:40.Z
template: "post"
draft: false
slug: "hot-papers-2021-08-17"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-08-17"
socialImage: "/media/flying-marine.jpg"

---

# 1. On the Opportunities and Risks of Foundation Models

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi

- retweets: 9565, favorites: 7 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.07258) | [pdf](https://arxiv.org/pdf/2108.07258)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

AI is undergoing a paradigm shift with the rise of models (e.g., BERT, DALL-E, GPT-3) that are trained on broad data at scale and are adaptable to a wide range of downstream tasks. We call these models foundation models to underscore their critically central yet incomplete character. This report provides a thorough account of the opportunities and risks of foundation models, ranging from their capabilities (e.g., language, vision, robotics, reasoning, human interaction) and technical principles (e.g., model architectures, training procedures, data, systems, security, evaluation, theory) to their applications (e.g., law, healthcare, education) and societal impact (e.g., inequity, misuse, economic and environmental impact, legal and ethical considerations). Though foundation models are based on conventional deep learning and transfer learning, their scale results in new emergent capabilities, and their effectiveness across so many tasks incentivizes homogenization. Homogenization provides powerful leverage but demands caution, as the defects of the foundation model are inherited by all the adapted models downstream. Despite the impending widespread deployment of foundation models, we currently lack a clear understanding of how they work, when they fail, and what they are even capable of due to their emergent properties. To tackle these questions, we believe much of the critical research on foundation models will require deep interdisciplinary collaboration commensurate with their fundamentally sociotechnical nature.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stanford&#39;s ~entire AI Department has just released a 200 page 100 author Neural Scaling Laws Manifesto.<br><br>They&#39;re pivoting to positioning themselves as #1 at academic ML Scaling (e.g. GPT-4) research.<br><br>&quot;On the Opportunities and Risks of Foundation Models&quot;<a href="https://t.co/rFNh0m2CmB">https://t.co/rFNh0m2CmB</a> <a href="https://t.co/B6i0zbGLGU">pic.twitter.com/B6i0zbGLGU</a></p>&mdash; Ethan Caballero (@ethancaballero) <a href="https://twitter.com/ethancaballero/status/1427679507062923268?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Learning Open-World Object Proposals without Learning to Classify

Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon, Weicheng Kuo

- retweets: 504, favorites: 127 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.06753) | [pdf](https://arxiv.org/pdf/2108.06753)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Object proposals have become an integral preprocessing steps of many vision pipelines including object detection, weakly supervised detection, object discovery, tracking, etc. Compared to the learning-free methods, learning-based proposals have become popular recently due to the growing interest in object detection. The common paradigm is to learn object proposals from data labeled with a set of object regions and their corresponding categories. However, this approach often struggles with novel objects in the open world that are absent in the training set. In this paper, we identify that the problem is that the binary classifiers in existing proposal methods tend to overfit to the training categories. Therefore, we propose a classification-free Object Localization Network (OLN) which estimates the objectness of each region purely by how well the location and shape of a region overlap with any ground-truth object (e.g., centerness and IoU). This simple strategy learns generalizable objectness and outperforms existing proposals on cross-category generalization on COCO, as well as cross-dataset evaluation on RoboNet, Object365, and EpicKitchens. Finally, we demonstrate the merit of OLN for long-tail object detection on large vocabulary dataset, LVIS, where we notice clear improvement in rare and common categories.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning Open-World Object Proposals without Learning to Classify<br>pdf: <a href="https://t.co/UxOBMGqpir">https://t.co/UxOBMGqpir</a><br>abs: <a href="https://t.co/mFo2d8JdGJ">https://t.co/mFo2d8JdGJ</a> <a href="https://t.co/cu8d5JTx1c">pic.twitter.com/cu8d5JTx1c</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427466655622389762?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. SOTR: Segmenting Objects with Transformers

Ruohao Guo, Dantong Niu, Liao Qu, Zhenbo Li

- retweets: 270, favorites: 115 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.06747) | [pdf](https://arxiv.org/pdf/2108.06747)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Most recent transformer-based models show impressive performance on vision tasks, even better than Convolution Neural Networks (CNN). In this work, we present a novel, flexible, and effective transformer-based model for high-quality instance segmentation. The proposed method, Segmenting Objects with TRansformers (SOTR), simplifies the segmentation pipeline, building on an alternative CNN backbone appended with two parallel subtasks: (1) predicting per-instance category via transformer and (2) dynamically generating segmentation mask with the multi-level upsampling module. SOTR can effectively extract lower-level feature representations and capture long-range context dependencies by Feature Pyramid Network (FPN) and twin transformer, respectively. Meanwhile, compared with the original transformer, the proposed twin transformer is time- and resource-efficient since only a row and a column attention are involved to encode pixels. Moreover, SOTR is easy to be incorporated with various CNN backbones and transformer model variants to make considerable improvements for the segmentation accuracy and training convergence. Extensive experiments show that our SOTR performs well on the MS COCO dataset and surpasses state-of-the-art instance segmentation approaches. We hope our simple but strong framework could serve as a preferment baseline for instance-level recognition. Our code is available at https://github.com/easton-cau/SOTR.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SOTR: Segmenting Objects with Transformers<br>pdf: <a href="https://t.co/eplIKD4mgZ">https://t.co/eplIKD4mgZ</a><br>abs: <a href="https://t.co/ARAaQ7VJAe">https://t.co/ARAaQ7VJAe</a><br>github: <a href="https://t.co/XlVZrJh25P">https://t.co/XlVZrJh25P</a><br><br>performs well on the MS COCO dataset and surpasses sota instance segmentation approaches <a href="https://t.co/06tH3XPtKQ">pic.twitter.com/06tH3XPtKQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427437974149599235?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and  Intra-modal Knowledge Integration

Yuhao Cui, Zhou Yu, Chunqi Wang, Zhongzhou Zhao, Ji Zhang, Meng Wang, Jun Yu

- retweets: 72, favorites: 33 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.07073) | [pdf](https://arxiv.org/pdf/2108.07073)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

Vision-and-language pretraining (VLP) aims to learn generic multimodal representations from massive image-text pairs. While various successful attempts have been proposed, learning fine-grained semantic alignments between image-text pairs plays a key role in their approaches. Nevertheless, most existing VLP approaches have not fully utilized the intrinsic knowledge within the image-text pairs, which limits the effectiveness of the learned alignments and further restricts the performance of their models. To this end, we introduce a new VLP method called ROSITA, which integrates the cross- and intra-modal knowledge in a unified scene graph to enhance the semantic alignments. Specifically, we introduce a novel structural knowledge masking (SKM) strategy to use the scene graph structure as a priori to perform masked language (region) modeling, which enhances the semantic alignments by eliminating the interference information within and across modalities. Extensive ablation studies and comprehensive analysis verifies the effectiveness of ROSITA in semantic alignments. Pretrained with both in-domain and out-of-domain datasets, ROSITA significantly outperforms existing state-of-the-art VLP methods on three typical vision-and-language tasks over six benchmark datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration<br>pdf: <a href="https://t.co/ha8jCqGSxF">https://t.co/ha8jCqGSxF</a><br>abs: <a href="https://t.co/1ZlwQ6a2vN">https://t.co/1ZlwQ6a2vN</a><br>github: <a href="https://t.co/kYhbj9cU3L">https://t.co/kYhbj9cU3L</a> <a href="https://t.co/bq3CCxt5Ff">pic.twitter.com/bq3CCxt5Ff</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427468564924141570?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Who's Waldo? Linking People Across Text and Images

Claire Yuqing Cui, Apoorv Khandelwal, Yoav Artzi, Noah Snavely, Hadar Averbuch-Elor

- retweets: 36, favorites: 52 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.07253) | [pdf](https://arxiv.org/pdf/2108.07253)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a task and benchmark dataset for person-centric visual grounding, the problem of linking between people named in a caption and people pictured in an image. In contrast to prior work in visual grounding, which is predominantly object-based, our new task masks out the names of people in captions in order to encourage methods trained on such image-caption pairs to focus on contextual cues (such as rich interactions between multiple people), rather than learning associations between names and appearances. To facilitate this task, we introduce a new dataset, Who's Waldo, mined automatically from image-caption data on Wikimedia Commons. We propose a Transformer-based method that outperforms several strong baselines on this task, and are releasing our data to the research community to spur work on contextual models that consider both vision and language.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Who’s Waldo? Linking People Across Text and Images<br>pdf: <a href="https://t.co/iiktNH0yey">https://t.co/iiktNH0yey</a><br>abs: <a href="https://t.co/y65WaRRkVd">https://t.co/y65WaRRkVd</a><br><br>present a task, dataset, and method for linking people across images and text <a href="https://t.co/XMlBRiqg9m">pic.twitter.com/XMlBRiqg9m</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427437320341438464?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Spectral Detection of Simplicial Communities via Hodge Laplacians

Sanjukta Krishnagopal, Ginestra Bianconi

- retweets: 56, favorites: 31 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.06547) | [pdf](https://arxiv.org/pdf/2108.06547)
- [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

While the study of graphs has been very popular, simplicial complexes are relatively new in the network science community. Despite being are a source of rich information, graphs are limited to pairwise interactions. However, several real world networks such as social networks, neuronal networks etc. involve simultaneous interactions between more than two nodes. Simplicial complexes provide a powerful mathematical way to model such interactions. Now, the spectrum of the graph Laplacian is known to be indicative of community structure, with nonzero eigenvectors encoding the identity of communities. Here, we propose that the spectrum of the Hodge Laplacian, a higher-order Laplacian applied to simplicial complexes, encodes simplicial communities. We formulate an algorithm to extract simplicial communities (of arbitrary dimension). We apply this algorithm on simplicial complex benchmarks and on real data including social networks and language-networks, where higher-order relationships are intrinsic. Additionally, datasets for simplicial complexes are scarce. Hence, we introduce a method of optimally generating a simplicial complex from its network backbone through estimating the \textit{true} higher-order relationships when its community structure is known. We do so by using the adjusted mutual information to identify the configuration that best matches the expected data partition. Lastly, we demonstrate an example of persistent simplicial communities inspired by the field of persistence homology.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Curious to know which Zachary-Karate-Club members had an higher-order interaction? Find out by looking at our work today in the arxiv  <a href="https://t.co/3ocyBxX1JO">https://t.co/3ocyBxX1JO</a>. Many thanks to 𝗦𝗮𝗻𝗷𝘂𝗸𝘁𝗮 𝗞𝗿𝗶𝘀𝗵𝗻𝗮𝗴𝗼𝗽𝗮𝗹 for the wonderful collaboration! <a href="https://t.co/zq9XXtuLWh">pic.twitter.com/zq9XXtuLWh</a></p>&mdash; Ginestra Bianconi (@gin_bianconi) <a href="https://twitter.com/gin_bianconi/status/1427621342329851913?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Constrained Iterative LQG for Real-Time Chance-ConstrainedGaussian  Belief Space Planning

Jianyu Chen, Yutaka Shimizu, Liting Sun, Masayoshi Tomizuka, Wei Zhan

- retweets: 25, favorites: 62 (08/18/2021 07:33:40)

- links: [abs](https://arxiv.org/abs/2108.06533) | [pdf](https://arxiv.org/pdf/2108.06533)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [eess.SY](https://arxiv.org/list/eess.SY/recent)

Motion planning under uncertainty is of significant importance for safety-critical systems such as autonomous vehicles. Such systems have to satisfy necessary constraints (e.g., collision avoidance) with potential uncertainties coming from either disturbed system dynamics or noisy sensor measurements. However, existing motion planning methods cannot efficiently find the robust optimal solutions under general nonlinear and non-convex settings. In this paper, we formulate such problem as chance-constrained Gaussian belief space planning and propose the constrained iterative Linear Quadratic Gaussian (CILQG) algorithm as a real-time solution. In this algorithm, we iteratively calculate a Gaussian approximation of the belief and transform the chance-constraints. We evaluate the effectiveness of our method in simulations of autonomous driving planning tasks with static and dynamic obstacles. Results show that CILQG can handle uncertainties more appropriately and has faster computation time than baseline methods.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">IROSに出した論文がArxivに上がりました！ロボット(特に自動運転)の確率的な経路計画してる方やILQR, ILQG,最適制御などの単語を聞くと震えがとまらない方の目に入れば嬉しいです！<a href="https://t.co/27ypLpgJ2e">https://t.co/27ypLpgJ2e</a></p>&mdash; じゃん (@purewater0901) <a href="https://twitter.com/purewater0901/status/1427490421131792387?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Online Multi-Granularity Distillation for GAN Compression

Yuxi Ren, Jie Wu, Xuefeng Xiao, Jianchao Yang

- retweets: 42, favorites: 25 (08/18/2021 07:33:41)

- links: [abs](https://arxiv.org/abs/2108.06908) | [pdf](https://arxiv.org/pdf/2108.06908)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative Adversarial Networks (GANs) have witnessed prevailing success in yielding outstanding images, however, they are burdensome to deploy on resource-constrained devices due to ponderous computational costs and hulking memory usage. Although recent efforts on compressing GANs have acquired remarkable results, they still exist potential model redundancies and can be further compressed. To solve this issue, we propose a novel online multi-granularity distillation (OMGD) scheme to obtain lightweight GANs, which contributes to generating high-fidelity images with low computational demands. We offer the first attempt to popularize single-stage online distillation for GAN-oriented compression, where the progressively promoted teacher generator helps to refine the discriminator-free based student generator. Complementary teacher generators and network layers provide comprehensive and multi-granularity concepts to enhance visual fidelity from diverse dimensions. Experimental results on four benchmark datasets demonstrate that OMGD successes to compress 40x MACs and 82.5X parameters on Pix2Pix and CycleGAN, without loss of image quality. It reveals that OMGD provides a feasible solution for the deployment of real-time image translation on resource-constrained devices. Our code and models are made public at: https://github.com/bytedance/OMGD.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Online Multi-Granularity Distillation for GAN Compression<br>pdf: <a href="https://t.co/mYPCYIks8B">https://t.co/mYPCYIks8B</a><br>abs: <a href="https://t.co/uJ5cbfGPnL">https://t.co/uJ5cbfGPnL</a><br>github: <a href="https://t.co/N5XY817RGD">https://t.co/N5XY817RGD</a> <a href="https://t.co/PDhTy3d9wK">pic.twitter.com/PDhTy3d9wK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1427446161376436230?ref_src=twsrc%5Etfw">August 17, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. WikiChurches: A Fine-Grained Dataset of Architectural Styles with  Real-World Challenges

Björn Barz, Joachim Denzler

- retweets: 42, favorites: 11 (08/18/2021 07:33:41)

- links: [abs](https://arxiv.org/abs/2108.06959) | [pdf](https://arxiv.org/pdf/2108.06959)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We introduce a novel dataset for architectural style classification, consisting of 9,485 images of church buildings. Both images and style labels were sourced from Wikipedia. The dataset can serve as a benchmark for various research fields, as it combines numerous real-world challenges: fine-grained distinctions between classes based on subtle visual features, a comparatively small sample size, a highly imbalanced class distribution, a high variance of viewpoints, and a hierarchical organization of labels, where only some images are labeled at the most precise level. In addition, we provide 631 bounding box annotations of characteristic visual features for 139 churches from four major categories. These annotations can, for example, be useful for research on fine-grained classification, where additional expert knowledge about distinctive object parts is often available. Images and annotations are available at: https://doi.org/10.5281/zenodo.5166987



