---
title: Hot Papers 2021-05-21
date: 2021-05-22T08:44:22.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-21"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-21"
socialImage: "/media/flying-marine.jpg"

---

# 1. Measuring Coding Challenge Competence With APPS

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, Jacob Steinhardt

- retweets: 36288, favorites: 0 (05/22/2021 08:44:22)

- links: [abs](https://arxiv.org/abs/2105.09938) | [pdf](https://arxiv.org/pdf/2105.09938)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

While programming is one of the most broadly applicable skills in modern society, modern machine learning models still cannot code solutions to basic problems. It can be difficult to accurately assess code generation performance, and there has been surprisingly little work on evaluating code generation in a way that is both flexible and rigorous. To meet this challenge, we introduce APPS, a benchmark for code generation. Unlike prior work in more restricted settings, our benchmark measures the ability of models to take an arbitrary natural language specification and generate Python code fulfilling this specification. Similar to how companies assess candidate software developers, we then evaluate models by checking their generated code on test cases. Our benchmark includes 10,000 problems, which range from having simple one-line solutions to being substantial algorithmic challenges. We fine-tune large language models on both GitHub and our training set, and we find that the prevalence of syntax errors is decreasing exponentially. Recent models such as GPT-Neo can pass approximately 15% of the test cases of introductory problems, so we find that machine learning models are beginning to learn how to code. As the social significance of automatic code generation increases over the coming years, our benchmark can provide an important measure for tracking advancements.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can Transformers crack the coding interview? We collected 10,000 programming problems to find out. GPT-3 isn&#39;t very good, but new models like GPT-Neo are starting to be able to solve introductory coding challenges.<br><br>paper: <a href="https://t.co/90HrYv4QW9">https://t.co/90HrYv4QW9</a><br>dataset: <a href="https://t.co/NGrS7M3VPx">https://t.co/NGrS7M3VPx</a> <a href="https://t.co/XkghbY4eLp">pic.twitter.com/XkghbY4eLp</a></p>&mdash; Dan Hendrycks (@DanHendrycks) <a href="https://twitter.com/DanHendrycks/status/1395536919774121984?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. DeepDebug: Fixing Python Bugs Using Stack Traces, Backtranslation, and  Code Skeletons

Dawn Drain, Colin B. Clement, Guillermo Serrato, Neel Sundaresan

- retweets: 1122, favorites: 149 (05/22/2021 08:44:22)

- links: [abs](https://arxiv.org/abs/2105.09352) | [pdf](https://arxiv.org/pdf/2105.09352)
- [cs.SE](https://arxiv.org/list/cs.SE/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The joint task of bug localization and program repair is an integral part of the software development process. In this work we present DeepDebug, an approach to automated debugging using large, pretrained transformers. We begin by training a bug-creation model on reversed commit data for the purpose of generating synthetic bugs. We apply these synthetic bugs toward two ends. First, we directly train a backtranslation model on all functions from 200K repositories. Next, we focus on 10K repositories for which we can execute tests, and create buggy versions of all functions in those repositories that are covered by passing tests. This provides us with rich debugging information such as stack traces and print statements, which we use to finetune our model which was pretrained on raw source code. Finally, we strengthen all our models by expanding the context window beyond the buggy function itself, and adding a skeleton consisting of that function's parent class, imports, signatures, docstrings, and method bodies, in order of priority. On the QuixBugs benchmark, we increase the total number of fixes found by over 50%, while also decreasing the false positive rate from 35% to 5% and decreasing the timeout from six hours to one minute. On our own benchmark of executable tests, our model fixes 68% of all bugs on its first attempt without using traces, and after adding traces it fixes 75% on first attempt. We will open-source our framework and validation set for evaluating on executable tests.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DeepDebug: Fixing Python Bugs Using Stack Traces,<br>Backtranslation, and Code Skeletons<br>pdf: <a href="https://t.co/9sSMlSgTJK">https://t.co/9sSMlSgTJK</a><br>abs: <a href="https://t.co/1jQm4r7PTe">https://t.co/1jQm4r7PTe</a><br><br>an approach to automated debugging using large, pretrained transformers <a href="https://t.co/Lvxvqde28A">pic.twitter.com/Lvxvqde28A</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395541622889369607?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. KLUE: Korean Language Understanding Evaluation

Sungjoon Park, Jihyung Moon, Sungdong Kim, Won Ik Cho, Jiyoon Han, Jangwon Park, Chisung Song, Junseong Kim, Yongsook Song, Taehwan Oh, Joohong Lee, Juhyun Oh, Sungwon Lyu, Younghoon Jeong, Inkwon Lee, Sangwoo Seo, Dongjun Lee, Hyunwoo Kim, Myeonghwa Lee, Seongbo Jang, Seungwon Do, Sunkyoung Kim, Kyungtae Lim, Jongwon Lee, Kyumin Park, Jamin Shin, Seonghyun Kim, Lucy Park, Alice Oh, Jungwoo Ha, Kyunghyun Cho Alice Oh Jungwoo Ha Kyunghyun Cho

- retweets: 716, favorites: 229 (05/22/2021 08:44:22)

- links: [abs](https://arxiv.org/abs/2105.09680) | [pdf](https://arxiv.org/pdf/2105.09680)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

We introduce Korean Language Understanding Evaluation (KLUE) benchmark. KLUE is a collection of 8 Korean natural language understanding (NLU) tasks, including Topic Classification, Semantic Textual Similarity, Natural Language Inference, Named Entity Recognition, Relation Extraction, Dependency Parsing, Machine Reading Comprehension, and Dialogue State Tracking. We build all of the tasks from scratch from diverse source corpora while respecting copyrights, to ensure accessibility for anyone without any restrictions. With ethical considerations in mind, we carefully design annotation protocols. Along with the benchmark tasks and data, we provide suitable evaluation metrics and fine-tuning recipes for pretrained language models for each task. We furthermore release the pretrained language models (PLM), KLUE-BERT and KLUE-RoBERTa, to help reproduce baseline models on KLUE and thereby facilitate future research. We make a few interesting observations from the preliminary experiments using the proposed KLUE benchmark suite, already demonstrating the usefulness of this new benchmark suite. First, we find KLUE-RoBERTa-large outperforms other baselines, including multilingual PLMs and existing open-source Korean PLMs. Second, we see minimal degradation in performance even when we replace personally identifiable information from the pretraining corpus, suggesting that privacy and NLU capability are not at odds with each other. Lastly, we find that using BPE tokenization in combination with morpheme-level pre-tokenization is effective in tasks involving morpheme-level tagging, detection and generation. In addition to accelerating Korean NLP research, our comprehensive documentation on creating KLUE will facilitate creating similar resources for other languages in the future. KLUE is available at this https URL (https://klue-benchmark.com/).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">do you want a language understanding evaluation suite for a new language, but you don’t know where to start? we didn’t either. check out KLUE: <a href="https://t.co/DN4Jr4n8LV">https://t.co/DN4Jr4n8LV</a></p>&mdash; Kyunghyun Cho (@kchonyc) <a href="https://twitter.com/kchonyc/status/1395549653240778753?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Happy to share the first <a href="https://twitter.com/hashtag/Korean?src=hash&amp;ref_src=twsrc%5Etfw">#Korean</a> <a href="https://twitter.com/hashtag/NLU?src=hash&amp;ref_src=twsrc%5Etfw">#NLU</a> benchmark including 8 tasks: <a href="https://twitter.com/hashtag/KLUE?src=hash&amp;ref_src=twsrc%5Etfw">#KLUE</a>. More than 30 Korean <a href="https://twitter.com/hashtag/NLP?src=hash&amp;ref_src=twsrc%5Etfw">#NLP</a> researchers from about 20 groups devoted this project. If you want to deep dive in Korean NLP or make your own language NLU benchmark, please check it out!<a href="https://t.co/nDYxqgMJvU">https://t.co/nDYxqgMJvU</a> <a href="https://t.co/DPQOMkwIOB">pic.twitter.com/DPQOMkwIOB</a></p>&mdash; Jung-Woo Ha (@JungWooHa2) <a href="https://twitter.com/JungWooHa2/status/1395655177839857664?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Birds of a Feather: Capturing Avian Shape Models from Images

Yufu Wang, Nikos Kolotouros, Kostas Daniilidis, Marc Badger

- retweets: 176, favorites: 82 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09396) | [pdf](https://arxiv.org/pdf/2105.09396)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Animals are diverse in shape, but building a deformable shape model for a new species is not always possible due to the lack of 3D data. We present a method to capture new species using an articulated template and images of that species. In this work, we focus mainly on birds. Although birds represent almost twice the number of species as mammals, no accurate shape model is available. To capture a novel species, we first fit the articulated template to each training sample. By disentangling pose and shape, we learn a shape space that captures variation both among species and within each species from image evidence. We learn models of multiple species from the CUB dataset, and contribute new species-specific and multi-species shape models that are useful for downstream reconstruction tasks. Using a low-dimensional embedding, we show that our learned 3D shape space better reflects the phylogenetic relationships among birds than learned perceptual features.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Birds of a Feather: Capturing Avian Shape Models from Images<br>pdf: <a href="https://t.co/C8cIfPLwbf">https://t.co/C8cIfPLwbf</a><br>abs: <a href="https://t.co/sGQKamOLlV">https://t.co/sGQKamOLlV</a><br>project page: <a href="https://t.co/i1uEzyRgvN">https://t.co/i1uEzyRgvN</a> <a href="https://t.co/n2fK88NFfT">pic.twitter.com/n2fK88NFfT</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395547815737282562?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Birds of a Feather, our CVPR&#39;21 paper; 3D vision in the service of biology (<a href="https://twitter.com/akanazawa?ref_src=twsrc%5Etfw">@akanazawa</a>, <a href="https://twitter.com/Michael_J_Black?ref_src=twsrc%5Etfw">@Michael_J_Black</a>, <a href="https://twitter.com/FuaPv?ref_src=twsrc%5Etfw">@FuaPv</a>, <a href="https://twitter.com/silvia_zuffi?ref_src=twsrc%5Etfw">@silvia_zuffi</a>, <a href="https://twitter.com/KordingLab?ref_src=twsrc%5Etfw">@KordingLab</a>); <a href="https://t.co/JB7FkmP9ZM">https://t.co/JB7FkmP9ZM</a><a href="https://t.co/mDdAtewQ60">https://t.co/mDdAtewQ60</a> <a href="https://t.co/0mMwYFnSik">pic.twitter.com/0mMwYFnSik</a></p>&mdash; Kostas Daniilidis (@KostasPenn) <a href="https://twitter.com/KostasPenn/status/1395764742480961536?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. More Than Just Attention: Learning Cross-Modal Attentions with  Contrastive Constraints

Yuxiao Chen, Jianbo Yuan, Long Zhao, Rui Luo, Larry Davis, Dimitris N. Metaxas

- retweets: 196, favorites: 54 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09597) | [pdf](https://arxiv.org/pdf/2105.09597)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Attention mechanisms have been widely applied to cross-modal tasks such as image captioning and information retrieval, and have achieved remarkable improvements due to its capability to learn fine-grained relevance across different modalities. However, existing attention models could be sub-optimal and lack preciseness because there is no direct supervision involved during training. In this work, we propose Contrastive Content Re-sourcing (CCR) and Contrastive Content Swapping (CCS) constraints to address such limitation. These constraints supervise the training of attention models in a contrastive learning manner without requiring explicit attention annotations. Additionally, we introduce three metrics, namely Attention Precision, Recall and F1-Score, to quantitatively evaluate the attention quality. We evaluate the proposed constraints with cross-modal retrieval (image-text matching) task. The experiments on both Flickr30k and MS-COCO datasets demonstrate that integrating these attention constraints into two state-of-the-art attention-based models improves the model performance in terms of both retrieval accuracy and attention metrics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">More Than Just Attention: Learning Cross-Modal Attentions with Contrastive Constraints<br>pdf: <a href="https://t.co/FoaQbmWa8N">https://t.co/FoaQbmWa8N</a><br>abs: <a href="https://t.co/Um8WpIiPuZ">https://t.co/Um8WpIiPuZ</a> <a href="https://t.co/GDx1BEqSF5">pic.twitter.com/GDx1BEqSF5</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395570796693700612?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient  Hyperparameter Optimization

Noor Awad, Neeratyoy Mallik, Frank Hutter

- retweets: 42, favorites: 48 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09821) | [pdf](https://arxiv.org/pdf/2105.09821)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.NE](https://arxiv.org/list/cs.NE/recent)

Modern machine learning algorithms crucially rely on several design decisions to achieve strong performance, making the problem of Hyperparameter Optimization (HPO) more important than ever. Here, we combine the advantages of the popular bandit-based HPO method Hyperband (HB) and the evolutionary search approach of Differential Evolution (DE) to yield a new HPO method which we call DEHB. Comprehensive results on a very broad range of HPO problems, as well as a wide range of tabular benchmarks from neural architecture search, demonstrate that DEHB achieves strong performance far more robustly than all previous HPO methods we are aware of, especially for high-dimensional problems with discrete input dimensions. For example, DEHB is up to 1000x faster than random search. It is also efficient in computational time, conceptually simple and easy to implement, positioning it well to become a new default HPO method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Surprising turn in <a href="https://twitter.com/hashtag/HPO?src=hash&amp;ref_src=twsrc%5Etfw">#HPO</a> and <a href="https://twitter.com/hashtag/NAS?src=hash&amp;ref_src=twsrc%5Etfw">#NAS</a>: we now have DEHB, an evolutionary successor of BOHB that&#39;s up to 1000x faster than random search for optimizing 6 NN hypers and up to 32x faster than BOHB on NAS-Bench-B201! <a href="https://twitter.com/hashtag/IJCAI21?src=hash&amp;ref_src=twsrc%5Etfw">#IJCAI21</a> paper: <a href="https://t.co/TTr7ypHhVA">https://t.co/TTr7ypHhVA</a> Code: <a href="https://t.co/vekLg4xiQO">https://t.co/vekLg4xiQO</a> <a href="https://t.co/yZQKlShOZu">pic.twitter.com/yZQKlShOZu</a></p>&mdash; Frank Hutter (@FrankRHutter) <a href="https://twitter.com/FrankRHutter/status/1395624548066906115?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Efficient and Robust LiDAR-Based End-to-End Navigation

Zhijian Liu, Alexander Amini, Sibo Zhu, Sertac Karaman, Song Han, Daniela Rus

- retweets: 36, favorites: 36 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09932) | [pdf](https://arxiv.org/pdf/2105.09932)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Deep learning has been used to demonstrate end-to-end neural network learning for autonomous vehicle control from raw sensory input. While LiDAR sensors provide reliably accurate information, existing end-to-end driving solutions are mainly based on cameras since processing 3D data requires a large memory footprint and computation cost. On the other hand, increasing the robustness of these systems is also critical; however, even estimating the model's uncertainty is very challenging due to the cost of sampling-based methods. In this paper, we present an efficient and robust LiDAR-based end-to-end navigation framework. We first introduce Fast-LiDARNet that is based on sparse convolution kernel optimization and hardware-aware model design. We then propose Hybrid Evidential Fusion that directly estimates the uncertainty of the prediction from only a single forward pass and then fuses the control predictions intelligently. We evaluate our system on a full-scale vehicle and demonstrate lane-stable as well as navigation capabilities. In the presence of out-of-distribution events (e.g., sensor failures), our system significantly improves robustness and reduces the number of takeovers in the real world.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient and Robust LiDAR-Based End-to-End Navigation<br>pdf: <a href="https://t.co/4w4oaki55v">https://t.co/4w4oaki55v</a><br>abs: <a href="https://t.co/HcioNC3RWS">https://t.co/HcioNC3RWS</a><br>project page: <a href="https://t.co/zvf6xZ03DC">https://t.co/zvf6xZ03DC</a> <a href="https://t.co/b0w3PeRO6D">pic.twitter.com/b0w3PeRO6D</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395556111328153601?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Contrastive Learning for Many-to-many Multilingual Neural Machine  Translation

Xiao Pan, Mingxuan Wang, Liwei Wu, Lei Li

- retweets: 40, favorites: 19 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09501) | [pdf](https://arxiv.org/pdf/2105.09501)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Existing multilingual machine translation approaches mainly focus on English-centric directions, while the non-English directions still lag behind. In this work, we aim to build a many-to-many translation system with an emphasis on the quality of non-English language directions. Our intuition is based on the hypothesis that a universal cross-language representation leads to better multilingual translation performance. To this end, we propose \method, a training method to obtain a single unified multilingual translation model. mCOLT is empowered by two techniques: (i) a contrastive learning scheme to close the gap among representations of different languages, and (ii) data augmentation on both multiple parallel and monolingual data to further align token representations. For English-centric directions, mCOLT achieves competitive or even better performance than a strong pre-trained model mBART on tens of WMT benchmarks. For non-English directions, mCOLT achieves an improvement of average 10+ BLEU compared with the multilingual baseline.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Contrastive Learning for Many-to-many Multilingual Neural Machine Translation<br>pdf: <a href="https://t.co/hO23mqH207">https://t.co/hO23mqH207</a><br>abs: <a href="https://t.co/Q55KIDjqjI">https://t.co/Q55KIDjqjI</a><br><br>contrastive learning can significantly improve zero-shot machine translation directions <a href="https://t.co/1qdsZFNRA4">pic.twitter.com/1qdsZFNRA4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395561460483465217?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. VTNet: Visual Transformer Network for Object Goal Navigation

Heming Du, Xin Yu, Liang Zheng

- retweets: 20, favorites: 37 (05/22/2021 08:44:23)

- links: [abs](https://arxiv.org/abs/2105.09447) | [pdf](https://arxiv.org/pdf/2105.09447)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Object goal navigation aims to steer an agent towards a target object based on observations of the agent. It is of pivotal importance to design effective visual representations of the observed scene in determining navigation actions. In this paper, we introduce a Visual Transformer Network (VTNet) for learning informative visual representation in navigation. VTNet is a highly effective structure that embodies two key properties for visual representations: First, the relationships among all the object instances in a scene are exploited; Second, the spatial locations of objects and image regions are emphasized so that directional navigation signals can be learned. Furthermore, we also develop a pre-training scheme to associate the visual representations with navigation signals, and thus facilitate navigation policy learning. In a nutshell, VTNet embeds object and region features with their location cues as spatial-aware descriptors and then incorporates all the encoded descriptors through attention operations to achieve informative representation for navigation. Given such visual representations, agents are able to explore the correlations between visual observations and navigation actions. For example, an agent would prioritize "turning right" over "turning left" when the visual representation emphasizes on the right side of activation map. Experiments in the artificial environment AI2-Thor demonstrate that VTNet significantly outperforms state-of-the-art methods in unseen testing environments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VTNet: Visual Transformer Network for Object Goal Navigation<br>pdf: <a href="https://t.co/jWu7ALSGdb">https://t.co/jWu7ALSGdb</a><br>abs: <a href="https://t.co/ABuqCPdVin">https://t.co/ABuqCPdVin</a><br><br>a spatial-enhanced local object descriptor and positional global descriptor, fused via multi-head attention to achieve final visual representation <a href="https://t.co/7uK4IQgr0C">pic.twitter.com/7uK4IQgr0C</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1395543192276242437?ref_src=twsrc%5Etfw">May 21, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



