---
title: Hot Papers 2020-09-24
date: 2020-09-25T08:00:47.Z
template: "post"
draft: false
slug: "hot-papers-2020-09-24"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2020-09-24"
socialImage: "/media/flying-marine.jpg"

---

# 1. Tasks, stability, architecture, and compute: Training more effective  learned optimizers, and using them to train themselves

Luke Metz, Niru Maheswaranathan, C. Daniel Freeman, Ben Poole, Jascha Sohl-Dickstein

- retweets: 18348, favorites: 0 (09/25/2020 08:00:47)

- links: [abs](https://arxiv.org/abs/2009.11243) | [pdf](https://arxiv.org/pdf/2009.11243)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Much as replacing hand-designed features with learned functions has revolutionized how we solve perceptual tasks, we believe learned algorithms will transform how we train models. In this work we focus on general-purpose learned optimizers capable of training a wide variety of problems with no user-specified hyperparameters. We introduce a new, neural network parameterized, hierarchical optimizer with access to additional features such as validation loss to enable automatic regularization. Most learned optimizers have been trained on only a single task, or a small number of tasks. We train our optimizers on thousands of tasks, making use of orders of magnitude more compute, resulting in optimizers that generalize better to unseen tasks. The learned optimizers not only perform well, but learn behaviors that are distinct from existing first order optimizers. For instance, they generate update steps that have implicit regularization and adapt as the problem hyperparameters (e.g. batch size) or architecture (e.g. neural network width) change. Finally, these learned optimizers show evidence of being useful for out of distribution tasks such as training themselves from scratch.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have a new paper on learned optimizers! We used thousands of tasks (and a lot of compute 😬) to train general purpose learned optimizers that perform well on never-before-seen tasks, and can even train new versions of themselves.<a href="https://t.co/LQf6o3Fwq7">https://t.co/LQf6o3Fwq7</a><br>1/8</p>&mdash; Luke Metz (@Luke_Metz) <a href="https://twitter.com/Luke_Metz/status/1308951548979011585?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Dataset Cartography: Mapping and Diagnosing Datasets with Training  Dynamics

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, Yejin Choi

- retweets: 1038, favorites: 219 (09/25/2020 08:00:47)

- links: [abs](https://arxiv.org/abs/2009.10795) | [pdf](https://arxiv.org/pdf/2009.10795)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Large datasets have become commonplace in NLP research. However, the increased emphasis on data quantity has made it challenging to assess the quality of data. We introduce "Data Maps"---a model-based tool to characterize and diagnose datasets. We leverage a largely ignored source of information: the behavior of the model on individual instances during training (training dynamics) for building data maps. This yields two intuitive measures for each example---the model's confidence in the true class, and the variability of this confidence across epochs, in a single run of training. Experiments on four datasets show that these model-dependent measures reveal three distinct regions in the data map, each with pronounced characteristics. First, our data maps show the presence of "ambiguous" regions with respect to the model, which contribute the most towards out-of-distribution generalization. Second, the most populous regions in the data are "easy to learn" for the model, and play an important role in model optimization. Finally, data maps uncover a region with instances that the model finds "hard to learn"; these often correspond to labeling errors. Our results indicate that a shift in focus from quantity to quality of data could lead to robust models and improved out-of-distribution generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dataset cartography:  a new way to look at your training dataset, derived from model training dynamics with respect to each instance.  Forthcoming EMNLP paper by <a href="https://twitter.com/swabhz?ref_src=twsrc%5Etfw">@swabhz</a> <a href="https://twitter.com/royschwartz02?ref_src=twsrc%5Etfw">@royschwartz02</a> <a href="https://twitter.com/NickLourie?ref_src=twsrc%5Etfw">@NickLourie</a> <a href="https://twitter.com/yizhongwyz?ref_src=twsrc%5Etfw">@yizhongwyz</a> <a href="https://twitter.com/HannaHajishirzi?ref_src=twsrc%5Etfw">@HannaHajishirzi</a> <a href="https://twitter.com/nlpnoah?ref_src=twsrc%5Etfw">@nlpnoah</a> <a href="https://twitter.com/YejinChoinka?ref_src=twsrc%5Etfw">@YejinChoinka</a>  <a href="https://t.co/MNu6Kbha0J">https://t.co/MNu6Kbha0J</a></p>&mdash; Noah A Smith (@nlpnoah) <a href="https://twitter.com/nlpnoah/status/1308994306062135296?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">As datasets have grown larger, data exploration has become increasingly challenging. Our new work on Dataset Cartography, at <a href="https://twitter.com/emnlp2020?ref_src=twsrc%5Etfw">@emnlp2020</a> with <a href="https://twitter.com/royschwartz02?ref_src=twsrc%5Etfw">@royschwartz02</a>, <a href="https://twitter.com/NickLourie?ref_src=twsrc%5Etfw">@NickLourie</a>, <a href="https://twitter.com/yizhongwyz?ref_src=twsrc%5Etfw">@yizhongwyz</a>, <a href="https://twitter.com/HannaHajishirzi?ref_src=twsrc%5Etfw">@HannaHajishirzi</a>, <a href="https://twitter.com/nlpnoah?ref_src=twsrc%5Etfw">@nlpnoah</a>, <a href="https://twitter.com/YejinChoinka?ref_src=twsrc%5Etfw">@YejinChoinka</a> offers a solution 🗺️<br>Paper: <a href="https://t.co/9JsYrxeACa">https://t.co/9JsYrxeACa</a> 1/n <a href="https://t.co/1hItp5yOx2">pic.twitter.com/1hItp5yOx2</a></p>&mdash; Swabha Swayamdipta (@swabhz) <a href="https://twitter.com/swabhz/status/1309217889568854016?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Message Passing for Hyper-Relational Knowledge Graphs

Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, Jens Lehmann

- retweets: 242, favorites: 33 (09/25/2020 08:00:47)

- links: [abs](https://arxiv.org/abs/2009.10847) | [pdf](https://arxiv.org/pdf/2009.10847)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Hyper-relational knowledge graphs (KGs) (e.g., Wikidata) enable associating additional key-value pairs along with the main triple to disambiguate, or restrict the validity of a fact. In this work, we propose a message passing based graph encoder - StarE capable of modeling such hyper-relational KGs. Unlike existing approaches, StarE can encode an arbitrary number of additional information (qualifiers) along with the main triple while keeping the semantic roles of qualifiers and triples intact. We also demonstrate that existing benchmarks for evaluating link prediction (LP) performance on hyper-relational KGs suffer from fundamental flaws and thus develop a new Wikidata-based dataset - WD50K. Our experiments demonstrate that StarE based LP model outperforms existing approaches across multiple benchmarks. We also confirm that leveraging qualifiers is vital for link prediction with gains up to 25 MRR points compared to triple-based representations.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We&#39;re releasing everything on StarE ⭐️ - a <a href="https://twitter.com/hashtag/GNN?src=hash&amp;ref_src=twsrc%5Etfw">#GNN</a> encoder for hyper-relational <a href="https://twitter.com/hashtag/KnowledgeGraph?src=hash&amp;ref_src=twsrc%5Etfw">#KnowledgeGraph</a> techniques like RDF* and LPG. Have fun 😊<br><br>Blog: <a href="https://t.co/OV2FprqzJt">https://t.co/OV2FprqzJt</a><br>Paper: <a href="https://t.co/K49xGI6MI7">https://t.co/K49xGI6MI7</a><br>Code: <a href="https://t.co/mrwyvXoPMf">https://t.co/mrwyvXoPMf</a><br>Report <a href="https://twitter.com/weights_biases?ref_src=twsrc%5Etfw">@weights_biases</a> : <a href="https://t.co/KFsxZyiw31">https://t.co/KFsxZyiw31</a> <a href="https://t.co/NfAndJ5eSs">https://t.co/NfAndJ5eSs</a></p>&mdash; Michael Galkin (@michael_galkin) <a href="https://twitter.com/michael_galkin/status/1309080340154273793?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Harnessing Multilinguality in Unsupervised Machine Translation for Rare  Languages

Xavier Garcia, Aditya Siddhant, Orhan Firat, Ankur P. Parikh

- retweets: 198, favorites: 72 (09/25/2020 08:00:47)

- links: [abs](https://arxiv.org/abs/2009.11201) | [pdf](https://arxiv.org/pdf/2009.11201)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Unsupervised translation has reached impressive performance on resource-rich language pairs such as English-French and English-German. However, early studies have shown that in more realistic settings involving low-resource, rare languages, unsupervised translation performs poorly, achieving less than 3.0 BLEU. In this work, we show that multilinguality is critical to making unsupervised systems practical for low-resource settings. In particular, we present a single model for 5 low-resource languages (Gujarati, Kazakh, Nepali, Sinhala, and Turkish) to and from English directions, which leverages monolingual and auxiliary parallel data from other high-resource language pairs via a three-stage training scheme. We outperform all current state-of-the-art unsupervised baselines for these languages, achieving gains of up to 14.4 BLEU. Additionally, we outperform a large collection of supervised WMT submissions for various language pairs as well as match the performance of the current state-of-the-art supervised model for Nepali-English. We conduct a series of ablation studies to establish the robustness of our model under different degrees of data quality, as well as to analyze the factors which led to the superior performance of the proposed approach over traditional unsupervised models.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out our multilingual unsupervised translation work! Theory + SOTA results. Led by <a href="https://twitter.com/xgarcia238?ref_src=twsrc%5Etfw">@xgarcia238</a> (1/4)<br><br>1. Multilingual View of Unsupervised MT  - Findings of EMNLP 2020  (<a href="https://t.co/oibhq2FDZ4">https://t.co/oibhq2FDZ4</a> )<br><br>2. Multilingual Unsupervised MT for Rare Languages (<a href="https://t.co/PkQAlH7lcq">https://t.co/PkQAlH7lcq</a> ) <a href="https://t.co/FcfMCeRJ7y">pic.twitter.com/FcfMCeRJ7y</a></p>&mdash; Ankur Parikh (@ank_parikh) <a href="https://twitter.com/ank_parikh/status/1309140734386343936?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. The cost of coordination can exceed the benefit of collaboration in  performing complex tasks

Vince J. Straub, Milena Tsvetkova, Taha Yasseri

- retweets: 159, favorites: 71 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.11038) | [pdf](https://arxiv.org/pdf/2009.11038)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.GT](https://arxiv.org/list/cs.GT/recent) | [nlin.AO](https://arxiv.org/list/nlin.AO/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Collective decision-making is ubiquitous when observing the behavior of intelligent agents, including humans. However, there are inconsistencies in our theoretical understanding of whether there is a collective advantage from interacting with group members of varying levels of competence in solving problems of varying complexity. Moreover, most existing experiments have relied on highly stylized tasks, reducing the generality of their results. The present study narrows the gap between experimental control and realistic settings, reporting the results from an analysis of collective problem-solving in the context of a real-world citizen science task environment in which individuals with manipulated differences in task-relevant training collaborated on the Wildcam Gorongosa task, hosted by The Zooniverse. We find that dyads gradually improve in performance but do not experience a collective benefit compared to individuals in most situations; rather, the cost of team coordination to efficiency and speed is consistently larger than the leverage of having a partner, even if they are expertly trained. It is only in terms of accuracy in the most complex tasks that having an additional expert significantly improves performance upon that of non-experts. Our findings have important theoretical and applied implications for collective problem-solving: to improve efficiency, one could prioritize providing task-relevant training and relying on trained experts working alone over interaction and to improve accuracy, one could target the expertise of selectively trained individuals.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Prefer to finish a task alone rather than w someone else when it&#39;s too complex? You should do that if you&#39;re good! After 3 years of work finally the science is in!<a href="https://t.co/O53mAUMgG1">https://t.co/O53mAUMgG1</a><br>The cost of coordination can exceed the benefit of collaboration in performing complex tasks <a href="https://t.co/JhfXFmkRuc">pic.twitter.com/JhfXFmkRuc</a></p>&mdash; Taha Yasseri (@TahaYasseri) <a href="https://twitter.com/TahaYasseri/status/1309061358902751233?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Scene Graph to Image Generation with Contextualized Object Layout  Refinement

Maor Ivgi, Yaniv Benny, Avichai Ben-David, Jonathan Berant, Lior Wolf

- retweets: 110, favorites: 38 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.10939) | [pdf](https://arxiv.org/pdf/2009.10939)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generating high-quality images from scene graphs, that is, graphs that describe multiple entities in complex relations, is a challenging task that attracted substantial interest recently. Prior work trained such models by using supervised learning, where the goal is to produce the exact target image layout for each scene graph. It relied on predicting object locations and shapes independently and in parallel. However, scene graphs are underspecified, and thus the same scene graph often occurs with many target images in the training data. This leads to generated images with high inter-object overlap, empty areas, blurry objects, and overall compromised quality. In this work, we propose a method that alleviates these issues by generating all object layouts together and reducing the reliance on such supervision. Our model predicts layouts directly from embeddings (without predicting intermediate boxes) by gradually upsampling, refining and contextualizing object layouts. It is trained with a novel adversarial loss, that optimizes the interaction between object pairs. This improves coverage and removes overlaps, while maintaining sensible contours and respecting objects relations. We empirically show on the COCO-STUFF dataset that our proposed approach substantially improves the quality of generated layouts as well as the overall image quality. Our evaluation shows that we improve layout coverage by almost 20 points, and drop object overlap to negligible amounts. This leads to better image generation, relation fulfillment and objects quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Scene Graph to Image Generation with Contextualized Object Layout Refinement<br>pdf: <a href="https://t.co/So4Ux1OXQA">https://t.co/So4Ux1OXQA</a><br>abs: <a href="https://t.co/Xd02vZTGHH">https://t.co/Xd02vZTGHH</a> <a href="https://t.co/VRv2kdRHwu">pic.twitter.com/VRv2kdRHwu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1308945306227224577?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. On Data Augmentation for Extreme Multi-label Classification

Danqing Zhang, Tao Li, Haiyang Zhang, Bing Yin

- retweets: 83, favorites: 53 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.10778) | [pdf](https://arxiv.org/pdf/2009.10778)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper, we focus on data augmentation for the extreme multi-label classification (XMC) problem. One of the most challenging issues of XMC is the long tail label distribution where even strong models suffer from insufficient supervision. To mitigate such label bias, we propose a simple and effective augmentation framework and a new state-of-the-art classifier. Our augmentation framework takes advantage of the pre-trained GPT-2 model to generate label-invariant perturbations of the input texts to augment the existing training data. As a result, it present substantial improvements over baseline models. Our contributions are two-factored: (1) we introduce a new state-of-the-art classifier that uses label attention with RoBERTa and combine it with our augmentation framework for further improvement; (2) we present a broad study on how effective are different augmentation methods in the XMC task.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Using language-model-based and rule-based data augmentation to deal with extremely unbalanced data scenarios.<br><br>This is common when dealing with multi-label text classification.<a href="https://t.co/HkHGJMpepx">https://t.co/HkHGJMpepx</a> <a href="https://t.co/U5CuZAHEfC">pic.twitter.com/U5CuZAHEfC</a></p>&mdash; elvis (@omarsar0) <a href="https://twitter.com/omarsar0/status/1309102454018117637?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal  Transformers

Jaemin Cho, Jiasen Lu, Dustin Schwenk, Hannaneh Hajishirzi, Aniruddha Kembhavi

- retweets: 72, favorites: 60 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.11278) | [pdf](https://arxiv.org/pdf/2009.11278)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Mirroring the success of masked language models, vision-and-language counterparts like ViLBERT, LXMERT and UNITER have achieved state of the art performance on a variety of multimodal discriminative tasks like visual question answering and visual grounding. Recent work has also successfully adapted such models towards the generative task of image captioning. This begs the question: Can these models go the other way and generate images from pieces of text? Our analysis of a popular representative from this model family - LXMERT - finds that it is unable to generate rich and semantically meaningful imagery with its current training setup. We introduce X-LXMERT, an extension to LXMERT with training refinements including: discretizing visual representations, using uniform masking with a large range of masking ratios and aligning the right pre-training datasets to the right objectives which enables it to paint. X-LXMERT's image generation capabilities rival state of the art generative models while its question answering and captioning abilities remains comparable to LXMERT. Finally, we demonstrate the generality of these training refinements by adding image generation capabilities into UNITER to produce X-UNITER.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers<br>pdf: <a href="https://t.co/nxyYF4LTDy">https://t.co/nxyYF4LTDy</a><br>abs: <a href="https://t.co/CL4SDy0DjZ">https://t.co/CL4SDy0DjZ</a><br>project page: <a href="https://t.co/gOqLdAxAJK">https://t.co/gOqLdAxAJK</a> <a href="https://t.co/8ycgxyPpkP">pic.twitter.com/8ycgxyPpkP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1308934290386845697?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. KoBE: Knowledge-Based Machine Translation Evaluation

Zorik Gekhman, Roee Aharoni, Genady Beryozkin, Markus Freitag, Wolfgang Macherey

- retweets: 58, favorites: 67 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.11027) | [pdf](https://arxiv.org/pdf/2009.11027)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We propose a simple and effective method for machine translation evaluation which does not require reference translations. Our approach is based on (1) grounding the entity mentions found in each source sentence and candidate translation against a large-scale multilingual knowledge base, and (2) measuring the recall of the grounded entities found in the candidate vs. those found in the source. Our approach achieves the highest correlation with human judgements on 9 out of the 18 language pairs from the WMT19 benchmark for evaluation without references, which is the largest number of wins for a single evaluation method on this task. On 4 language pairs, we also achieve higher correlation with human judgements than BLEU. To foster further research, we release a dataset containing 1.8 million grounded entity mentions across 18 language pairs from the WMT19 metrics track data.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Turns out that using multilingual entity linking, we can automatically evaluate machine translation without any references! New paper with Zorik Gekhman, Genady Beryozkin, Markus Freitag and Wolfgang Macherey, to appear in Findings of EMNLP: <a href="https://t.co/87QTZWV0bS">https://t.co/87QTZWV0bS</a> <a href="https://twitter.com/GoogleAI?ref_src=twsrc%5Etfw">@GoogleAI</a> <a href="https://t.co/wiuKL9BXkg">pic.twitter.com/wiuKL9BXkg</a></p>&mdash; roeeaharoni (@roeeaharoni) <a href="https://twitter.com/roeeaharoni/status/1309057171473260551?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Few-shot Font Generation with Localized Style Representations and  Factorization

Song Park, Sanghyuk Chun, Junbum Cha, Bado Lee, Hyunjung Shim

- retweets: 81, favorites: 25 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.11042) | [pdf](https://arxiv.org/pdf/2009.11042)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Automatic few-shot font generation is in high demand because manual designs are expensive and sensitive to the expertise of designers. Existing few-shot font generation methods aim to learn to disentangle the style and content element from a few reference glyphs, and mainly focus on a universal style representation for each font style. However, such approach limits the model in representing diverse local styles, and thus makes it unsuitable to the most complicated letter system, e.g., Chinese, whose characters consist of a varying number of components (often called "radical") with a highly complex structure. In this paper, we propose a novel font generation method by learning localized styles, namely component-wise style representations, instead of universal styles. The proposed style representations enable us to synthesize complex local details in text designs. However, learning component-wise styles solely from reference glyphs is infeasible in the few-shot font generation scenario, when a target script has a large number of components, e.g., over 200 for Chinese. To reduce the number of reference glyphs, we simplify component-wise styles by a product of component factor and style factor, inspired by low-rank matrix factorization. Thanks to the combination of strong representation and a compact factorization strategy, our method shows remarkably better few-shot font generation results (with only 8 reference glyph images) than other state-of-the-arts, without utilizing strong locality supervision, e.g., location of each component, skeleton, or strokes. The source code is available at https://github.com/clovaai/lffont.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-shot Font Generation with Localized Style Representations and Factorization<br>pdf: <a href="https://t.co/wVmkoK4Zgf">https://t.co/wVmkoK4Zgf</a><br>abs: <a href="https://t.co/FzWufs2TlA">https://t.co/FzWufs2TlA</a><br>github: <a href="https://t.co/KJ2BCzpwMX">https://t.co/KJ2BCzpwMX</a> <a href="https://t.co/fwHUilnISu">pic.twitter.com/fwHUilnISu</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1308930576305528832?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot

Jingtong Su, Yihang Chen, Tianle Cai, Tianhao Wu, Ruiqi Gao, Liwei Wang, Jason D. Lee

- retweets: 49, favorites: 32 (09/25/2020 08:00:48)

- links: [abs](https://arxiv.org/abs/2009.11094) | [pdf](https://arxiv.org/pdf/2009.11094)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Network pruning is a method for reducing test-time computational resource requirements with minimal performance degradation. Conventional wisdom of pruning algorithms suggests that: (1) Pruning methods exploit information from training data to find good subnetworks; (2) The architecture of the pruned network is crucial for good performance. In this paper, we conduct sanity checks for the above beliefs on several recent unstructured pruning methods and surprisingly find that: (1) A set of methods which aims to find good subnetworks of the randomly-initialized network (which we call "initial tickets"), hardly exploits any information from the training data; (2) For the pruned networks obtained by these methods, randomly changing the preserved weights in each layer, while keeping the total number of preserved weights unchanged per layer, does not affect the final performance. These findings inspire us to choose a series of simple \emph{data-independent} prune ratios for each layer, and randomly prune each layer accordingly to get a subnetwork (which we call "random tickets"). Experimental results show that our zero-shot random tickets outperforms or attains similar performance compared to existing "initial tickets". In addition, we identify one existing pruning method that passes our sanity checks. We hybridize the ratios in our random ticket with this method and propose a new method called "hybrid tickets", which achieves further improvement.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Do existing pruning methods really exploit the info from data? Are the architectures of the pruned networks really matter for the performance? We propose sanity checks on pruning methods and find a great part of existing methods does not rely on these! <a href="https://t.co/6Wk9DparBL">https://t.co/6Wk9DparBL</a> <a href="https://t.co/Wp2DJznQD0">pic.twitter.com/Wp2DJznQD0</a></p>&mdash; Tianle Cai (@tianle_cai) <a href="https://twitter.com/tianle_cai/status/1308938327735713793?ref_src=twsrc%5Etfw">September 24, 2020</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



