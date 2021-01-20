---
title: Hot Papers 2021-01-19
date: 2021-01-20T22:35:30.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. CheXtransfer: Performance and Parameter Efficiency of ImageNet Models  for Chest X-Ray Interpretation

Alexander Ke, William Ellsworth, Oishi Banerjee, Andrew Y. Ng, Pranav Rajpurkar

- retweets: 4627, favorites: 369 (01/20/2021 22:35:30)

- links: [abs](https://arxiv.org/abs/2101.06871) | [pdf](https://arxiv.org/pdf/2101.06871)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep learning methods for chest X-ray interpretation typically rely on pretrained models developed for ImageNet. This paradigm assumes that better ImageNet architectures perform better on chest X-ray tasks and that ImageNet-pretrained weights provide a performance boost over random initialization. In this work, we compare the transfer performance and parameter efficiency of 16 popular convolutional architectures on a large chest X-ray dataset (CheXpert) to investigate these assumptions. First, we find no relationship between ImageNet performance and CheXpert performance for both models without pretraining and models with pretraining. Second, we find that, for models without pretraining, the choice of model family influences performance more than size within a family for medical imaging tasks. Third, we observe that ImageNet pretraining yields a statistically significant boost in performance across architectures, with a higher boost for smaller architectures. Fourth, we examine whether ImageNet architectures are unnecessarily large for CheXpert by truncating final blocks from pretrained models, and find that we can make models 3.25x more parameter-efficient on average without a statistically significant drop in performance. Our work contributes new experimental evidence about the relation of ImageNet to chest x-ray interpretation performance.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Does higher performance on ImageNet translate to higher performance on medical imaging tasks?<br><br>Surprisingly, the answer is no!<br><br>We investigate their relationship.<br>Paper: <a href="https://t.co/LDAbVkEmIf">https://t.co/LDAbVkEmIf</a><a href="https://twitter.com/_alexke?ref_src=twsrc%5Etfw">@_alexke</a>, William Ellsworth, Oishi Banerjee, <a href="https://twitter.com/AndrewYNg?ref_src=twsrc%5Etfw">@AndrewYNg</a> <a href="https://twitter.com/StanfordAILab?ref_src=twsrc%5Etfw">@StanfordAILab</a> <br><br>1/8 <a href="https://t.co/R0Gh0FaJsS">pic.twitter.com/R0Gh0FaJsS</a></p>&mdash; Pranav Rajpurkar (@pranavrajpurkar) <a href="https://twitter.com/pranavrajpurkar/status/1351462869284528129?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. A simple geometric proof for the benefit of depth in ReLU networks

Asaf Amrami, Yoav Goldberg

- retweets: 2110, favorites: 312 (01/20/2021 22:35:30)

- links: [abs](https://arxiv.org/abs/2101.07126) | [pdf](https://arxiv.org/pdf/2101.07126)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present a simple proof for the benefit of depth in multi-layer feedforward network with rectified activation ("depth separation"). Specifically we present a sequence of classification problems indexed by $m$ such that (a) for any fixed depth rectified network there exist an $m$ above which classifying problem $m$ correctly requires exponential number of parameters (in $m$); and (b) for any problem in the sequence, we present a concrete neural network with linear depth (in $m$) and small constant width ($\leq 4$) that classifies the problem with zero error.   The constructive proof is based on geometric arguments and a space folding construction.   While stronger bounds and results exist, our proof uses substantially simpler tools and techniques, and should be accessible to undergraduate students in computer science and people with similar backgrounds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Since <a href="https://twitter.com/asaf_amr?ref_src=twsrc%5Etfw">@asaf_amr</a> defended his masters yesterday, we decided its a good time to arxiv this ICLR 2020 reject.<br><br>It presents a *simple* constructive proof of the benefit of depth in neural nets, which, unlike other similar works, can be grasped by undergrads.<a href="https://t.co/CD7WNT79h6">https://t.co/CD7WNT79h6</a> <a href="https://t.co/P8WuEI2kG6">pic.twitter.com/P8WuEI2kG6</a></p>&mdash; (((ل()(ل() &#39;yoav)))) (@yoavgo) <a href="https://twitter.com/yoavgo/status/1351446335145340931?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Can a Fruit Fly Learn Word Embeddings?

Yuchen Liang, Chaitanya K. Ryali, Benjamin Hoover, Leopold Grinberg, Saket Navlakha, Mohammed J. Zaki, Dmitry Krotov

- retweets: 1300, favorites: 172 (01/20/2021 22:35:31)

- links: [abs](https://arxiv.org/abs/2101.06887) | [pdf](https://arxiv.org/pdf/2101.06887)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The mushroom body of the fruit fly brain is one of the best studied systems in neuroscience. At its core it consists of a population of Kenyon cells, which receive inputs from multiple sensory modalities. These cells are inhibited by the anterior paired lateral neuron, thus creating a sparse high dimensional representation of the inputs. In this work we study a mathematical formalization of this network motif and apply it to learning the correlational structure between words and their context in a corpus of unstructured text, a common natural language processing (NLP) task. We show that this network can learn semantic representations of words and can generate both static and context-dependent word embeddings. Unlike conventional methods (e.g., BERT, GloVe) that use dense representations for word embedding, our algorithm encodes semantic meaning of words and their context in the form of sparse binary hash codes. The quality of the learned representations is evaluated on word similarity analysis, word-sense disambiguation, and document classification. It is shown that not only can the fruit fly network motif achieve performance comparable to existing methods in NLP, but, additionally, it uses only a fraction of the computational resources (shorter training time and smaller memory footprint).

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can a Fruit Fly Learn Word Embeddings?<br>pdf: <a href="https://t.co/zhbnf3Vyvl">https://t.co/zhbnf3Vyvl</a><br>abs: <a href="https://t.co/9tep5IG80l">https://t.co/9tep5IG80l</a> <a href="https://t.co/3wsVAkdyDd">pic.twitter.com/3wsVAkdyDd</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351355495072886785?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our <a href="https://twitter.com/hashtag/ICLR2021?src=hash&amp;ref_src=twsrc%5Etfw">#ICLR2021</a> paper we study a well-established neurobiological network motif from the fruit fly brain and investigate the possibility of reusing its architecture for solving common natural language processing tasks.<br>Paper: <a href="https://t.co/vQYJePQG7Y">https://t.co/vQYJePQG7Y</a> <a href="https://t.co/Y7kWU5VDWH">pic.twitter.com/Y7kWU5VDWH</a></p>&mdash; Yuchen Liang (@YuchenLiangRPI) <a href="https://twitter.com/YuchenLiangRPI/status/1351642932223373312?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. The Role of Working Memory in Program Tracing

Will Crichton, Maneesh Agrawala, Pat Hanrahan

- retweets: 1190, favorites: 155 (01/20/2021 22:35:31)

- links: [abs](https://arxiv.org/abs/2101.06305) | [pdf](https://arxiv.org/pdf/2101.06305)
- [cs.HC](https://arxiv.org/list/cs.HC/recent)

Program tracing, or mentally simulating a program on concrete inputs, is an important part of general program comprehension. Programs involve many kinds of virtual state that must be held in memory, such as variable/value pairs and a call stack. In this work, we examine the influence of short-term working memory (WM) on a person's ability to remember program state during tracing. We first confirm that previous findings in cognitive psychology transfer to the programming domain: people can keep about 7 variable/value pairs in WM, and people will accidentally swap associations between variables due to WM load. We use a restricted focus viewing interface to further analyze the strategies people use to trace through programs, and the relationship of tracing strategy to WM. Given a straight-line program, we find half of our participants traced a program from the top-down line-by-line (linearly), and the other half start at the bottom and trace upward based on data dependencies (on-demand). Participants with an on-demand strategy made more WM errors while tracing straight-line code than with a linear strategy, but the two strategies contained an equal number of WM errors when tracing code with functions. We conclude with the implications of these findings for the design of programming tools: first, programs should be analyzed to identify and refactor human-memory-intensive sections of code. Second, programming environments should interactively visualize variable metadata to reduce WM load in accordance with a person's tracing strategy. Third, tools for program comprehension should enable externalizing program state while tracing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to announce my debut PL/HCI paper appearing at CHI&#39;21: &quot;The Role of Working Memory in Program Tracing&quot;.<br><br>Ever found it hard to remember stuff while you read a program? That&#39;s working memory! Check out our experiments exploring this phenomenon.<a href="https://t.co/nrVOMAVdya">https://t.co/nrVOMAVdya</a> <a href="https://t.co/CEZ7VBrSm6">pic.twitter.com/CEZ7VBrSm6</a></p>&mdash; Will Crichton (@wcrichton) <a href="https://twitter.com/wcrichton/status/1351644389475553288?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. GENIE: A Leaderboard for Human-in-the-Loop Evaluation of Text Generation

Daniel Khashabi, Gabriel Stanovsky, Jonathan Bragg, Nicholas Lourie, Jungo Kasai, Yejin Choi, Noah A. Smith, Daniel S. Weld

- retweets: 824, favorites: 135 (01/20/2021 22:35:31)

- links: [abs](https://arxiv.org/abs/2101.06561) | [pdf](https://arxiv.org/pdf/2101.06561)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Leaderboards have eased model development for many NLP datasets by standardizing their evaluation and delegating it to an independent external repository. Their adoption, however, is so far limited to tasks that can be reliably evaluated in an automatic manner. This work introduces GENIE, an extensible human evaluation leaderboard, which brings the ease of leaderboards to text generation tasks. GENIE automatically posts leaderboard submissions to crowdsourcing platforms asking human annotators to evaluate them on various axes (e.g., correctness, conciseness, fluency) and compares their answers to various automatic metrics. We introduce several datasets in English to GENIE, representing four core challenges in text generation: machine translation, summarization, commonsense reasoning, and machine comprehension. We provide formal granular evaluation metrics and identify areas for future research. We make GENIE publicly available and hope that it will spur progress in language generation models as well as their automatic and manual evaluation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Today we are releasing GENIE🧞, a human-in-loop leaderboard for the evaluation of text generation tasks!  We view this as a step forward towards streamlining human evaluation and making it more accessible. <a href="https://t.co/dPscoTFbkm">https://t.co/dPscoTFbkm</a><a href="https://t.co/l1onBnJWOe">https://t.co/l1onBnJWOe</a><a href="https://twitter.com/hashtag/NLP?src=hash&amp;ref_src=twsrc%5Etfw">#NLP</a></p>&mdash; Daniel Khashabi 🕊️ (@DanielKhashabi) <a href="https://twitter.com/DanielKhashabi/status/1351619145062703106?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Dissecting the Meme Magic: Understanding Indicators of Virality in Image  Memes

Chen Ling, Ihab AbuHilal, Jeremy Blackburn, Emiliano De Cristofaro, Savvas Zannettou, Gianluca Stringhini

- retweets: 528, favorites: 77 (01/20/2021 22:35:31)

- links: [abs](https://arxiv.org/abs/2101.06535) | [pdf](https://arxiv.org/pdf/2101.06535)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent)

Despite the increasingly important role played by image memes, we do not yet have a solid understanding of the elements that might make a meme go viral on social media. In this paper, we investigate what visual elements distinguish image memes that are highly viral on social media from those that do not get re-shared, across three dimensions: composition, subjects, and target audience. Drawing from research in art theory, psychology, marketing, and neuroscience, we develop a codebook to characterize image memes, and use it to annotate a set of 100 image memes collected from 4chan's Politically Incorrect Board (/pol/). On the one hand, we find that highly viral memes are more likely to use a close-up scale, contain characters, and include positive or negative emotions. On the other hand, image memes that do not present a clear subject the viewer can focus attention on, or that include long text are not likely to be re-shared by users.   We train machine learning models to distinguish between image memes that are likely to go viral and those that are unlikely to be re-shared, obtaining an AUC of 0.866 on our dataset. We also show that the indicators of virality identified by our model can help characterize the most viral memes posted on mainstream online social networks too, as our classifiers are able to predict 19 out of the 20 most popular image memes posted on Twitter and Reddit between 2016 and 2018. Overall, our analysis sheds light on what indicators characterize viral and non-viral visual content online, and set the basis for developing better techniques to create or moderate content that is more likely to catch the viewer's attention.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Ever wondered what visual elements contribute to make memes go viral? In our latest paper (accepted at <a href="https://twitter.com/ACM_CSCW?ref_src=twsrc%5Etfw">@ACM_CSCW</a>) we show that memes obey to the same rules of successful visual artworks. <br><br>Kudos to my student <a href="https://twitter.com/ciciling07?ref_src=twsrc%5Etfw">@ciciling07</a> for leading this project!<br><br>📜<a href="https://t.co/LMPYR9IifV">https://t.co/LMPYR9IifV</a> <a href="https://t.co/wfnWCXzvWm">pic.twitter.com/wfnWCXzvWm</a></p>&mdash; Gianluca Stringhini (@gianluca_string) <a href="https://twitter.com/gianluca_string/status/1351549795974873098?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Free Lunch for Few-shot Learning: Distribution Calibration

Shuo Yang, Lu Liu, Min Xu

- retweets: 441, favorites: 75 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.06395) | [pdf](https://arxiv.org/pdf/2101.06395)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Learning from a limited number of samples is challenging since the learned model can easily become overfitted based on the biased distribution formed by only a few training examples. In this paper, we calibrate the distribution of these few-sample classes by transferring statistics from the classes with sufficient examples, then an adequate number of examples can be sampled from the calibrated distribution to expand the inputs to the classifier. We assume every dimension in the feature representation follows a Gaussian distribution so that the mean and the variance of the distribution can borrow from that of similar classes whose statistics are better estimated with an adequate number of samples. Our method can be built on top of off-the-shelf pretrained feature extractors and classification models without extra parameters. We show that a simple logistic regression classifier trained using the features sampled from our calibrated distribution can outperform the state-of-the-art accuracy on two datasets (~5% improvement on miniImageNet compared to the next best). The visualization of these generated features demonstrates that our calibrated distribution is an accurate estimation.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Free Lunch for Few-shot Learning: Distribution Calibration<br>pdf: <a href="https://t.co/hN7jo5vGqz">https://t.co/hN7jo5vGqz</a><br>abs: <a href="https://t.co/S6sv6XnjWq">https://t.co/S6sv6XnjWq</a><br>github: <a href="https://t.co/KhBACOgHxU">https://t.co/KhBACOgHxU</a> <a href="https://t.co/XhNqmz2Euj">pic.twitter.com/XhNqmz2Euj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351382727967305728?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Trav-SHACL: Efficiently Validating Networks of SHACL Constraints

Mónica Figuera, Philipp D. Rohde, Maria-Esther Vidal

- retweets: 128, favorites: 81 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.07136) | [pdf](https://arxiv.org/pdf/2101.07136)
- [cs.DB](https://arxiv.org/list/cs.DB/recent)

Knowledge graphs have emerged as expressive data structures for Web data. Knowledge graph potential and the demand for ecosystems to facilitate their creation, curation, and understanding, is testified in diverse domains, e.g., biomedicine. The Shapes Constraint Language (SHACL) is the W3C recommendation language for integrity constraints over RDF knowledge graphs. Enabling quality assements of knowledge graphs, SHACL is rapidly gaining attention in real-world scenarios. SHACL models integrity constraints as a network of shapes, where a shape contains the constraints to be fullfiled by the same entities. The validation of a SHACL shape schema can face the issue of tractability during validation. To facilitate full adoption, efficient computational methods are required. We present Trav-SHACL, a SHACL engine capable of planning the traversal and execution of a shape schema in a way that invalid entities are detected early and needless validations are minimized. Trav-SHACL reorders the shapes in a shape schema for efficient validation and rewrites target and constraint queries for the fast detection of invalid entities. Trav-SHACL is empirically evaluated on 27 testbeds executed against knowledge graphs of up to 34M triples. Our experimental results suggest that Trav-SHACL exhibits high performance gradually and reduces validation time by a factor of up to 28.93 compared to the state of the art.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper entitled: &quot;Trav-SHACL: Efficiently Validating Networks of SHACL Constraints&quot; is accepted at WWW 2021.  <a href="https://t.co/KvPVxEgrwy">https://t.co/KvPVxEgrwy</a> Congratulations <a href="https://twitter.com/mofiguera?ref_src=twsrc%5Etfw">@mofiguera</a> <a href="https://twitter.com/philipp_rohde?ref_src=twsrc%5Etfw">@philipp_rohde</a> <a href="https://twitter.com/MEVidalSerodio?ref_src=twsrc%5Etfw">@MEVidalSerodio</a> <a href="https://twitter.com/TIB_SDM?ref_src=twsrc%5Etfw">@TIB_SDM</a> <a href="https://twitter.com/TIBHannover?ref_src=twsrc%5Etfw">@TIBHannover</a> <a href="https://twitter.com/l3s_luh?ref_src=twsrc%5Etfw">@l3s_luh</a></p>&mdash; Maria-Esther Vidal (@MEVidalSerodio) <a href="https://twitter.com/MEVidalSerodio/status/1351468677560999937?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The paper entitled: &quot;Trav-SHACL: Efficiently Validating Networks of SHACL Constraints&quot; by <a href="https://twitter.com/philipp_rohde?ref_src=twsrc%5Etfw">@philipp_rohde</a>, <a href="https://twitter.com/mofiguera?ref_src=twsrc%5Etfw">@mofiguera</a> and <a href="https://twitter.com/MEVidalSerodio?ref_src=twsrc%5Etfw">@MEVidalSerodio</a> is accepted at <a href="https://twitter.com/TheWebConf?ref_src=twsrc%5Etfw">@TheWebConf</a> 2021.  <br><br>Paper: <a href="https://t.co/QdjkZiOwiN">https://t.co/QdjkZiOwiN</a> <br>More papers: <a href="https://t.co/H6qaze6fUr">https://t.co/H6qaze6fUr</a><br><br>Congratulations!🎉<a href="https://twitter.com/TIBHannover?ref_src=twsrc%5Etfw">@TIBHannover</a> <a href="https://t.co/7cReyQNnTN">pic.twitter.com/7cReyQNnTN</a></p>&mdash; TIB SDM (@TIB_SDM) <a href="https://twitter.com/TIB_SDM/status/1351471191354187776?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. GeoSim: Photorealistic Image Simulation with Geometry-Aware Composition

Yun Chen, Frieda Rong, Shivam Duggal, Shenlong Wang, Xinchen Yan, Sivabalan Manivasagam, Shangjie Xue, Ersin Yumer, Raquel Urtasun

- retweets: 90, favorites: 51 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.06543) | [pdf](https://arxiv.org/pdf/2101.06543)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent)

Scalable sensor simulation is an important yet challenging open problem for safety-critical domains such as self-driving. Current work in image simulation either fail to be photorealistic or do not model the 3D environment and the dynamic objects within, losing high-level control and physical realism. In this paper, we present GeoSim, a geometry-aware image composition process that synthesizes novel urban driving scenes by augmenting existing images with dynamic objects extracted from other scenes and rendered at novel poses. Towards this goal, we first build a diverse bank of 3D objects with both realistic geometry and appearance from sensor data. During simulation, we perform a novel geometry-aware simulation-by-composition procedure which 1) proposes plausible and realistic object placements into a given scene, 2) renders novel views of dynamic objects from the asset bank, and 3) composes and blends the rendered image segments. The resulting synthetic images are photorealistic, traffic-aware, and geometrically consistent, allowing image simulation to scale to complex use cases. We demonstrate two such important applications: long-range realistic video simulation across multiple camera sensors, and synthetic data generation for data augmentation on downstream segmentation tasks.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GeoSim: Photorealistic Image Simulation with Geometry-Aware Composition for Self-Driving<br>pdf: <a href="https://t.co/G59c1PwskS">https://t.co/G59c1PwskS</a><br>abs: <a href="https://t.co/fGuQ6RNIJs">https://t.co/fGuQ6RNIJs</a> <a href="https://t.co/G0pjtao5wY">pic.twitter.com/G0pjtao5wY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351364087800885249?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Catching Out-of-Context Misinformation with Self-supervised Learning

Shivangi Aneja, Christoph Bregler, Matthias Nießner

- retweets: 90, favorites: 37 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.06278) | [pdf](https://arxiv.org/pdf/2101.06278)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Despite the recent attention to DeepFakes and other forms of image manipulations, one of the most prevalent ways to mislead audiences is the use of unaltered images in a new but false context. To address these challenges and support fact-checkers, we propose a new method that automatically detects out-of-context image and text pairs. Our core idea is a self-supervised training strategy where we only need images with matching (and non-matching) captions from different sources. At train time, our method learns to selectively align individual objects in an image with textual claims, without explicit supervision. At test time, we check for a given text pair if both texts correspond to same object(s) in the image but semantically convey different descriptions, which allows us to make fairly accurate out-of-context predictions. Our method achieves 82% out-of-context detection accuracy. To facilitate training our method, we created a large-scale dataset of 203,570 images which we match with 456,305 textual captions from a variety of news websites, blogs, and social media posts; i.e., for each image, we obtained several captions.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">(1/2)<br>One of the biggest issues on social media is the misuse of images in wrong/misleading contexts.<br><br>&quot;Catching Out-of-Context Misinformation with Self-supervised Learning&quot; detects these cases automatically!<br><br>Video: <a href="https://t.co/zLIYe63vbO">https://t.co/zLIYe63vbO</a><br>Paper: <a href="https://t.co/ihB2WUKjam">https://t.co/ihB2WUKjam</a> <a href="https://t.co/9ocjr6KxYY">pic.twitter.com/9ocjr6KxYY</a></p>&mdash; Matthias Niessner (@MattNiessner) <a href="https://twitter.com/MattNiessner/status/1351614540962795520?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. PLUME: Efficient 3D Object Detection from Stereo Images

Yan Wang, Bin Yang, Rui Hu, Ming Liang, Raquel Urtasun

- retweets: 92, favorites: 28 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.06594) | [pdf](https://arxiv.org/pdf/2101.06594)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

3D object detection plays a significant role in various robotic applications including self-driving. While many approaches rely on expensive 3D sensors like LiDAR to produce accurate 3D estimates, stereo-based methods have recently shown promising results at a lower cost. Existing methods tackle the problem in two steps: first depth estimation is performed, a pseudo LiDAR point cloud representation is computed from the depth estimates, and then object detection is performed in 3D space. However, because the two separate tasks are optimized in different metric spaces, the depth estimation is biased towards big objects and may cause sub-optimal performance of 3D detection. In this paper we propose a model that unifies these two tasks in the same metric space for the first time. Specifically, our model directly constructs a pseudo LiDAR feature volume (PLUME) in 3D space, which is used to solve both occupancy estimation and object detection tasks. PLUME achieves state-of-the-art performance on the challenging KITTI benchmark, with significantly reduced inference time compared with existing methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">PLUME: Efficient 3D Object Detection from Stereo Images<br>pdf: <a href="https://t.co/JorykV0aTm">https://t.co/JorykV0aTm</a><br>abs: <a href="https://t.co/7pUXGTkbxV">https://t.co/7pUXGTkbxV</a> <a href="https://t.co/XNBVdE6sNz">pic.twitter.com/XNBVdE6sNz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351388505646182402?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Hierarchical disentangled representation learning for singing voice  conversion

Naoya Takahashi, Mayank Kumar Singh, Yuki Mitsufuji

- retweets: 74, favorites: 41 (01/20/2021 22:35:32)

- links: [abs](https://arxiv.org/abs/2101.06842) | [pdf](https://arxiv.org/pdf/2101.06842)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

Conventional singing voice conversion (SVC) methods often suffer from operating in high-resolution audio owing to a high dimensionality of data. In this paper, we propose a hierarchical representation learning that enables the learning of disentangled representations with multiple resolutions independently. With the learned disentangled representations, the proposed method progressively performs SVC from low to high resolutions. Experimental results show that the proposed method outperforms baselines that operate with a single resolution in terms of mean opinion score (MOS), similarity score, and pitch accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical disentangled representation learning for singing voice conversion<br>pdf: <a href="https://t.co/f8wTOZ6mKY">https://t.co/f8wTOZ6mKY</a><br>abs: <a href="https://t.co/C199F88spA">https://t.co/C199F88spA</a> <a href="https://t.co/v22Qj6dXta">pic.twitter.com/v22Qj6dXta</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351364920055631874?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Deterministic Decremental SSSP and Approximate Min-Cost Flow in  Almost-Linear Time

Aaron Bernstein, Maximilian Probst Gutenberg, Thatchaphol Saranurak

- retweets: 49, favorites: 66 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.07149) | [pdf](https://arxiv.org/pdf/2101.07149)
- [cs.DS](https://arxiv.org/list/cs.DS/recent)

In the decremental single-source shortest paths problem, the goal is to maintain distances from a fixed source $s$ to every vertex $v$ in an $m$-edge graph undergoing edge deletions. In this paper, we conclude a long line of research on this problem by showing a near-optimal deterministic data structure that maintains $(1+\epsilon)$-approximate distance estimates and runs in $m^{1+o(1)}$ total update time.   Our result, in particular, removes the oblivious adversary assumption required by the previous breakthrough result by Henzinger et al. [FOCS'14], which leads to our second result: the first almost-linear time algorithm for $(1-\epsilon)$-approximate min-cost flow in undirected graphs where capacities and costs can be taken over edges and vertices. Previously, algorithms for max flow with vertex capacities, or min-cost flow with any capacities required super-linear time. Our result essentially completes the picture for approximate flow in undirected graphs.   The key technique of the first result is a novel framework that allows us to treat low-diameter graphs like expanders. This allows us to harness expander properties while bypassing shortcomings of expander decomposition, which almost all previous expander-based algorithms needed to deal with. For the second result, we break the notorious flow-decomposition barrier from the multiplicative-weight-update framework using randomization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is one of my strongest papers. We show...<br><br>1. A near-optimal deterministic single-source shortest paths algorithm on graphs undergoing edge deletions. <br><br>2. Almost-linear-time algorithms for approximate min-cost flow and balanced vertex separators.<a href="https://t.co/CBm3LoHsQw">https://t.co/CBm3LoHsQw</a></p>&mdash; Thatchaphol Saranurak (@eig) <a href="https://twitter.com/eig/status/1351547592530931712?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. What Makes Good In-Context Examples for GPT-$3$?

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, Weizhu Chen

- retweets: 56, favorites: 29 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.06804) | [pdf](https://arxiv.org/pdf/2101.06804)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

GPT-$3$ has attracted lots of attention due to its superior performance across a wide range of NLP tasks, especially with its powerful and versatile in-context few-shot learning ability. Despite its success, we found that the empirical results of GPT-$3$ depend heavily on the choice of in-context examples. In this work, we investigate whether there are more effective strategies for judiciously selecting in-context examples (relative to random sampling) that better leverage GPT-$3$'s few-shot capabilities. Inspired by the recent success of leveraging a retrieval module to augment large-scale neural network models, we propose to retrieve examples that are semantically-similar to a test sample to formulate its corresponding prompt. Intuitively, the in-context examples selected with such a strategy may serve as more informative inputs to unleash GPT-$3$'s extensive knowledge. We evaluate the proposed approach on several natural language understanding and generation benchmarks, where the retrieval-based prompt selection approach consistently outperforms the random baseline. Moreover, it is observed that the sentence encoders fine-tuned on task-related datasets yield even more helpful retrieval results. Notably, significant gains are observed on tasks such as table-to-text generation (41.9% on the ToTTo dataset) and open-domain question answering (45.5% on the NQ dataset). We hope our investigation could help understand the behaviors of GPT-$3$ and large-scale pre-trained LMs in general and enhance their few-shot capabilities.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What Makes Good In-Context Examples for GPT-3?<br>pdf: <a href="https://t.co/OoCMAzYaJO">https://t.co/OoCMAzYaJO</a><br>abs: <a href="https://t.co/9v9Ac4VDCK">https://t.co/9v9Ac4VDCK</a> <a href="https://t.co/jlyNgff29f">pic.twitter.com/jlyNgff29f</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351352980331753474?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling

Ze Yang, Shenlong Wang, Sivabalan Manivasagam, Zeng Huang, Wei-Chiu Ma, Xinchen Yan, Ersin Yumer, Raquel Urtasun

- retweets: 32, favorites: 44 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.06571) | [pdf](https://arxiv.org/pdf/2101.06571)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Constructing and animating humans is an important component for building virtual worlds in a wide variety of applications such as virtual reality or robotics testing in simulation. As there are exponentially many variations of humans with different shape, pose and clothing, it is critical to develop methods that can automatically reconstruct and animate humans at scale from real world data. Towards this goal, we represent the pedestrian's shape, pose and skinning weights as neural implicit functions that are directly learned from data. This representation enables us to handle a wide variety of different pedestrian shapes and poses without explicitly fitting a human parametric body model, allowing us to handle a wider range of human geometries and topologies. We demonstrate the effectiveness of our approach on various datasets and show that our reconstructions outperform existing state-of-the-art methods. Furthermore, our re-animation experiments show that we can generate 3D human animations at scale from a single RGB image (and/or an optional LiDAR sweep) as input.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">S3: Neural Shape, Skeleton, and Skinning Fields for 3D Human Modeling<br>pdf: <a href="https://t.co/W2LYBNjBBv">https://t.co/W2LYBNjBBv</a><br>abs: <a href="https://t.co/ZaVnVF7kiF">https://t.co/ZaVnVF7kiF</a> <a href="https://t.co/KrvT8cmZQW">pic.twitter.com/KrvT8cmZQW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351387827506974726?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Learning by Watching: Physical Imitation of Manipulation Skills from  Human Videos

Haoyu Xiong, Quanzhou Li, Yun-Chun Chen, Homanga Bharadhwaj, Samarth Sinha, Animesh Garg

- retweets: 22, favorites: 40 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.07241) | [pdf](https://arxiv.org/pdf/2101.07241)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We present an approach for physical imitation from human videos for robot manipulation tasks. The key idea of our method lies in explicitly exploiting the kinematics and motion information embedded in the video to learn structured representations that endow the robot with the ability to imagine how to perform manipulation tasks in its own context. To achieve this, we design a perception module that learns to translate human videos to the robot domain followed by unsupervised keypoint detection. The resulting keypoint-based representations provide semantically meaningful information that can be directly used for reward computing and policy learning. We evaluate the effectiveness of our approach on five robot manipulation tasks, including reaching, pushing, sliding, coffee making, and drawer closing. Detailed experimental evaluations demonstrate that our method performs favorably against previous approaches.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning by Watching: Physical Imitation of Manipulation Skills from Human Videos<br>pdf: <a href="https://t.co/ZvvY5YCaey">https://t.co/ZvvY5YCaey</a><br>abs: <a href="https://t.co/eumfNOPrny">https://t.co/eumfNOPrny</a><br>project page: <a href="https://t.co/PLfpuvJgeR">https://t.co/PLfpuvJgeR</a> <a href="https://t.co/cF2t8egfj6">pic.twitter.com/cF2t8egfj6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351398741828317184?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. Hierarchical Reinforcement Learning By Discovering Intrinsic Options

Jesse Zhang, Haonan Yu, Wei Xu

- retweets: 32, favorites: 27 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.06521) | [pdf](https://arxiv.org/pdf/2101.06521)
- [cs.LG](https://arxiv.org/list/cs.LG/recent)

We propose a hierarchical reinforcement learning method, HIDIO, that can learn task-agnostic options in a self-supervised manner while jointly learning to utilize them to solve sparse-reward tasks. Unlike current hierarchical RL approaches that tend to formulate goal-reaching low-level tasks or pre-define ad hoc lower-level policies, HIDIO encourages lower-level option learning that is independent of the task at hand, requiring few assumptions or little knowledge about the task structure. These options are learned through an intrinsic entropy minimization objective conditioned on the option sub-trajectories. The learned options are diverse and task-agnostic. In experiments on sparse-reward robotic manipulation and navigation tasks, HIDIO achieves higher success rates with greater sample efficiency than regular RL baselines and two state-of-the-art hierarchical RL methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Hierarchical Reinforcement Learning By Discovering Intrinsic Options<br>pdf: <a href="https://t.co/ztR3Nm3Yf0">https://t.co/ztR3Nm3Yf0</a><br>abs: <a href="https://t.co/uT0W3msqVr">https://t.co/uT0W3msqVr</a><br>github: <a href="https://t.co/xtzHecLpoK">https://t.co/xtzHecLpoK</a> <a href="https://t.co/Ir2mtGPqj9">pic.twitter.com/Ir2mtGPqj9</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351405844152414208?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. VideoClick: Video Object Segmentation with a Single Click

Namdar Homayounfar, Justin Liang, Wei-Chiu Ma, Raquel Urtasun

- retweets: 30, favorites: 27 (01/20/2021 22:35:33)

- links: [abs](https://arxiv.org/abs/2101.06545) | [pdf](https://arxiv.org/pdf/2101.06545)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Annotating videos with object segmentation masks typically involves a two stage procedure of drawing polygons per object instance for all the frames and then linking them through time. While simple, this is a very tedious, time consuming and expensive process, making the creation of accurate annotations at scale only possible for well-funded labs. What if we were able to segment an object in the full video with only a single click? This will enable video segmentation at scale with a very low budget opening the door to many applications. Towards this goal, in this paper we propose a bottom up approach where given a single click for each object in a video, we obtain the segmentation masks of these objects in the full video. In particular, we construct a correlation volume that assigns each pixel in a target frame to either one of the objects in the reference frame or the background. We then refine this correlation volume via a recurrent attention module and decode the final segmentation. To evaluate the performance, we label the popular and challenging Cityscapes dataset with video object segmentations. Results on this new CityscapesVideo dataset show that our approach outperforms all the baselines in this challenging setting.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VideoClick: Video Object Segmentation with a Single Click<br>pdf: <a href="https://t.co/ox6nJczlZR">https://t.co/ox6nJczlZR</a><br>abs: <a href="https://t.co/zi2dCjj5yx">https://t.co/zi2dCjj5yx</a> <a href="https://t.co/6gOEIaKSMQ">pic.twitter.com/6gOEIaKSMQ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1351386544104464385?ref_src=twsrc%5Etfw">January 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



