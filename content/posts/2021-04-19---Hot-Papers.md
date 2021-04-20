---
title: Hot Papers 2021-04-19
date: 2021-04-20T09:43:48.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-19"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-19"
socialImage: "/media/flying-marine.jpg"

---

# 1. MeshTalk: 3D Face Animation from Speech using Cross-Modality  Disentanglement

Alexander Richard, Michael Zollhoefer, Yandong Wen, Fernando de la Torre, Yaser Sheikh

- retweets: 1150, favorites: 241 (04/20/2021 09:43:48)

- links: [abs](https://arxiv.org/abs/2104.08223) | [pdf](https://arxiv.org/pdf/2104.08223)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

This paper presents a generic method for generating full facial 3D animation from speech. Existing approaches to audio-driven facial animation exhibit uncanny or static upper face animation, fail to produce accurate and plausible co-articulation or rely on person-specific models that limit their scalability. To improve upon existing models, we propose a generic audio-driven facial animation approach that achieves highly realistic motion synthesis results for the entire face. At the core of our approach is a categorical latent space for facial animation that disentangles audio-correlated and audio-uncorrelated information based on a novel cross-modality loss. Our approach ensures highly accurate lip motion, while also synthesizing plausible animation of the parts of the face that are uncorrelated to the audio signal, such as eye blinks and eye brow motion. We demonstrate that our approach outperforms several baselines and obtains state-of-the-art quality both qualitatively and quantitatively. A perceptual user study demonstrates that our approach is deemed more realistic than the current state-of-the-art in over 75% of cases. We recommend watching the supplemental video before reading the paper: https://research.fb.com/wp-content/uploads/2021/04/mesh_talk.mp4

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MeshTalk: 3D Face Animation from Speech using Cross-Modality Disentanglement<br>pdf: <a href="https://t.co/xgG6E6Qxhr">https://t.co/xgG6E6Qxhr</a><br>abs: <a href="https://t.co/mveay6UObE">https://t.co/mveay6UObE</a> <a href="https://t.co/CHMID54TgU">pic.twitter.com/CHMID54TgU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383990658680037381?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Check out MeshTalk, our latest work on audio-driven face animation!<br><br>Given a 3D neutral face mesh and speech as input, our approach can generate realistic lip motion and realistic upper face motion.<br><br>Paper: <a href="https://t.co/Cw1NbjKBOf">https://t.co/Cw1NbjKBOf</a><br>Video: <a href="https://t.co/P2xrKXYlRv">https://t.co/P2xrKXYlRv</a> <a href="https://t.co/H1jB9aWrKy">pic.twitter.com/H1jB9aWrKy</a></p>&mdash; Alexander Richard (@AlexRichardCS) <a href="https://twitter.com/AlexRichardCS/status/1384158612654723079?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Generating Bug-Fixes Using Pretrained Transformers

Dawn Drain, Chen Wu, Alexey Svyatkovskiy, Neel Sundaresan

- retweets: 323, favorites: 112 (04/20/2021 09:43:49)

- links: [abs](https://arxiv.org/abs/2104.07896) | [pdf](https://arxiv.org/pdf/2104.07896)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.PL](https://arxiv.org/list/cs.PL/recent)

Detecting and fixing bugs are two of the most important yet frustrating parts of the software development cycle. Existing bug detection tools are based mainly on static analyzers, which rely on mathematical logic and symbolic reasoning about the program execution to detect common types of bugs. Fixing bugs is typically left out to the developer. In this work we introduce DeepDebug: a data-driven program repair approach which learns to detect and fix bugs in Java methods mined from real-world GitHub repositories. We frame bug-patching as a sequence-to-sequence learning task consisting of two steps: (i) denoising pretraining, and (ii) supervised finetuning on the target translation task. We show that pretraining on source code programs improves the number of patches found by 33% as compared to supervised training from scratch, while domain-adaptive pretraining from natural language to code further improves the accuracy by another 32%. We refine the standard accuracy evaluation metric into non-deletion and deletion-only fixes, and show that our best model generates 75% more non-deletion fixes than the previous state of the art. In contrast to prior work, we attain our best results when generating raw code, as opposed to working with abstracted code that tends to only benefit smaller capacity models. Finally, we observe a subtle improvement from adding syntax embeddings along with the standard positional embeddings, as well as with adding an auxiliary task to predict each token's syntactic class. Despite focusing on Java, our approach is language agnostic, requiring only a general-purpose parser such as tree-sitter.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Generating Bug-Fixes Using Pretrained Transformers<br>pdf: <a href="https://t.co/dQO1kklYrD">https://t.co/dQO1kklYrD</a><br>abs: <a href="https://t.co/3nXK7XYnzm">https://t.co/3nXK7XYnzm</a> <a href="https://t.co/uDksKLeTBA">pic.twitter.com/uDksKLeTBA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383947501993021441?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Dual Contrastive Learning for Unsupervised Image-to-Image Translation

Junlin Han, Mehrdad Shoeiby, Lars Petersson, Mohammad Ali Armin

- retweets: 244, favorites: 109 (04/20/2021 09:43:49)

- links: [abs](https://arxiv.org/abs/2104.07689) | [pdf](https://arxiv.org/pdf/2104.07689)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.IV](https://arxiv.org/list/eess.IV/recent)

Unsupervised image-to-image translation tasks aim to find a mapping between a source domain X and a target domain Y from unpaired training data. Contrastive learning for Unpaired image-to-image Translation (CUT) yields state-of-the-art results in modeling unsupervised image-to-image translation by maximizing mutual information between input and output patches using only one encoder for both domains. In this paper, we propose a novel method based on contrastive learning and a dual learning setting (exploiting two encoders) to infer an efficient mapping between unpaired data. Additionally, while CUT suffers from mode collapse, a variant of our method efficiently addresses this issue. We further demonstrate the advantage of our approach through extensive ablation studies demonstrating superior performance comparing to recent approaches in multiple challenging image translation tasks. Lastly, we demonstrate that the gap between unsupervised methods and supervised methods can be efficiently closed.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Dual Contrastive Learning for Unsupervised Image-to-Image Translation<br>pdf: <a href="https://t.co/RpRRWmyvDS">https://t.co/RpRRWmyvDS</a><br>abs: <a href="https://t.co/1Fsnnopj3M">https://t.co/1Fsnnopj3M</a> <a href="https://t.co/MQlVpj2cia">pic.twitter.com/MQlVpj2cia</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383953696745168897?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. How to Train BERT with an Academic Budget

Peter Izsak, Moshe Berchansky, Omer Levy

- retweets: 174, favorites: 112 (04/20/2021 09:43:49)

- links: [abs](https://arxiv.org/abs/2104.07705) | [pdf](https://arxiv.org/pdf/2104.07705)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

While large language models \`a la BERT are used ubiquitously in NLP, pretraining them is considered a luxury that only a few well-funded industry labs can afford. How can one train such models with a more modest budget? We present a recipe for pretraining a masked language model in 24 hours, using only 8 low-range 12GB GPUs. We demonstrate that through a combination of software optimizations, design choices, and hyperparameter tuning, it is possible to produce models that are competitive with BERT-base on GLUE tasks at a fraction of the original pretraining cost.

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">大学の研究室規模の計算資源でも、24時間あればBERTを訓練できることを示した研究。12GBx8GPU設定を想定。最大系列長は短く・巨大ミニバッチサイズ・高学習率・巨大モデルなど、既存の知見の合わせ技のもと、BERT-baseと同等の性能を達成。 <a href="https://t.co/waWNMlxf3Z">https://t.co/waWNMlxf3Z</a> <a href="https://t.co/iTudEObbUB">pic.twitter.com/iTudEObbUB</a></p>&mdash; Shun Kiyono (@shunkiyono) <a href="https://twitter.com/shunkiyono/status/1384011120235618304?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How to Train BERT with an Academic Budget<br>While large language models à la BERT are used ubiquitously in NLP, pretraining them is considered a luxury that only a few well-funded industry labs can afford. <br><br>Paper <a href="https://t.co/b8YnncsNDW">https://t.co/b8YnncsNDW</a><br>GitHub <a href="https://t.co/ob8pDwZVQN">https://t.co/ob8pDwZVQN</a><br><br>↓ 1/3 <a href="https://t.co/wQMdqLFntu">pic.twitter.com/wQMdqLFntu</a></p>&mdash; Philip Vollet (@philipvollet) <a href="https://twitter.com/philipvollet/status/1384016685426872334?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. A Review of the State-of-the-Art on Tours for Dynamic Visualization of  High-dimensional Data

Stuart Lee, Dianne Cook, Natalia Da Silva, Ursula Laa, Earo Wang, Nick Spyrison, H. Sherry Zhang

- retweets: 196, favorites: 32 (04/20/2021 09:43:49)

- links: [abs](https://arxiv.org/abs/2104.08016) | [pdf](https://arxiv.org/pdf/2104.08016)
- [cs.GR](https://arxiv.org/list/cs.GR/recent) | [stat.OT](https://arxiv.org/list/stat.OT/recent)

This article discusses a high-dimensional visualization technique called the tour, which can be used to view data in more than three dimensions. We review the theory and history behind the technique, as well as modern software developments and applications of the tour that are being found across the sciences and machine learning.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">📜New preprint up that reviews the history and developments of using linear projections (the tour) for visualizing data in more than 3 dimensions. Team effort with <a href="https://twitter.com/visnut?ref_src=twsrc%5Etfw">@visnut</a>, <a href="https://twitter.com/UschiLaa?ref_src=twsrc%5Etfw">@UschiLaa</a>, <a href="https://twitter.com/nspyrison?ref_src=twsrc%5Etfw">@nspyrison</a>, <a href="https://twitter.com/huizezhangsh?ref_src=twsrc%5Etfw">@huizezhangsh</a>, <a href="https://twitter.com/pacocuak?ref_src=twsrc%5Etfw">@pacocuak</a> <a href="https://twitter.com/earowang?ref_src=twsrc%5Etfw">@earowang</a> <a href="https://t.co/tOfPi5Ddob">https://t.co/tOfPi5Ddob</a></p>&mdash; Stuart Lee (@_StuartLee) <a href="https://twitter.com/_StuartLee/status/1384012740549697536?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. ProphetNet-X: Large-Scale Pre-training Models for English, Chinese,  Multi-lingual, Dialog, and Code Generation

Weizhen Qi, Yeyun Gong, Yu Yan, Can Xu, Bolun Yao, Bartuer Zhou, Biao Cheng, Daxin Jiang, Jiusheng Chen, Ruofei Zhang, Houqiang Li, Nan Duan

- retweets: 156, favorites: 47 (04/20/2021 09:43:49)

- links: [abs](https://arxiv.org/abs/2104.08006) | [pdf](https://arxiv.org/pdf/2104.08006)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

Now, the pre-training technique is ubiquitous in natural language processing field. ProphetNet is a pre-training based natural language generation method which shows powerful performance on English text summarization and question generation tasks. In this paper, we extend ProphetNet into other domains and languages, and present the ProphetNet family pre-training models, named ProphetNet-X, where X can be English, Chinese, Multi-lingual, and so on. We pre-train a cross-lingual generation model ProphetNet-Multi, a Chinese generation model ProphetNet-Zh, two open-domain dialog generation models ProphetNet-Dialog-En and ProphetNet-Dialog-Zh. And also, we provide a PLG (Programming Language Generation) model ProphetNet-Code to show the generation performance besides NLG (Natural Language Generation) tasks. In our experiments, ProphetNet-X models achieve new state-of-the-art performance on 10 benchmarks. All the models of ProphetNet-X share the same model structure, which allows users to easily switch between different models. We make the code and models publicly available, and we will keep updating more pre-training models and finetuning scripts. A video to introduce ProphetNet-X usage is also released.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ProphetNet-X: Large-Scale Pre-training Models for English, Chinese, Multi-lingual, Dialog, and Code Generation<br>pdf: <a href="https://t.co/YZ0U6H6ob1">https://t.co/YZ0U6H6ob1</a><br>abs: <a href="https://t.co/M97osIY1xo">https://t.co/M97osIY1xo</a> <a href="https://t.co/Y1uUtE5H3z">pic.twitter.com/Y1uUtE5H3z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383981910959935496?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Actionable Models: Unsupervised Offline Reinforcement Learning of  Robotic Skills

Yevgen Chebotar, Karol Hausman, Yao Lu, Ted Xiao, Dmitry Kalashnikov, Jake Varley, Alex Irpan, Benjamin Eysenbach, Ryan Julian, Chelsea Finn, Sergey Levine

- retweets: 118, favorites: 79 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.07749) | [pdf](https://arxiv.org/pdf/2104.07749)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

We consider the problem of learning useful robotic skills from previously collected offline data without access to manually specified rewards or additional online exploration, a setting that is becoming increasingly important for scaling robot learning by reusing past robotic data. In particular, we propose the objective of learning a functional understanding of the environment by learning to reach any goal state in a given dataset. We employ goal-conditioned Q-learning with hindsight relabeling and develop several techniques that enable training in a particularly challenging offline setting. We find that our method can operate on high-dimensional camera images and learn a variety of skills on real robots that generalize to previously unseen scenes and objects. We also show that our method can learn to reach long-horizon goals across multiple episodes, and learn rich representations that can help with downstream tasks through pre-training or auxiliary objectives. The videos of our experiments can be found at https://actionable-models.github.io

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Excited to present our new work on Actionable Models, an approach for learning functional understanding of the world via goal-conditioned Q-functions in a fully-offline setting!<br><br>paper: <a href="https://t.co/qQCrOItwFj">https://t.co/qQCrOItwFj</a><br>website: <a href="https://t.co/cshpBx1YSD">https://t.co/cshpBx1YSD</a><a href="https://t.co/lYBJzyzWwU">https://t.co/lYBJzyzWwU</a></p>&mdash; Yevgen Chebotar (@YevgenChebotar) <a href="https://twitter.com/YevgenChebotar/status/1384246978834419717?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills<br>pdf: <a href="https://t.co/jOlIVw523B">https://t.co/jOlIVw523B</a><br>abs: <a href="https://t.co/hF2uiXz2L5">https://t.co/hF2uiXz2L5</a><br>project page: <a href="https://t.co/ZjXbw5Od1l">https://t.co/ZjXbw5Od1l</a> <a href="https://t.co/dYFTPuYsxz">pic.twitter.com/dYFTPuYsxz</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383954874723835905?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Enabling Electronic Structure-Based Ab-Initio Molecular Dynamics  Simulations with Hundreds of Millions of Atoms

Robert Schade, Tobias Kenter, Hossam Elgabarty, Michael Lass, Ole Schütt, Alfio Lazzaro, Hans Pabst, Stephan Mohr, Jürg Hutter, Thomas D. Kühne, Christian Plessl

- retweets: 127, favorites: 48 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.08245) | [pdf](https://arxiv.org/pdf/2104.08245)
- [physics.comp-ph](https://arxiv.org/list/physics.comp-ph/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent)

We push the boundaries of electronic structure-based \textit{ab-initio} molecular dynamics (AIMD) beyond 100 million atoms. This scale is otherwise barely reachable with classical force-field methods or novel neural network and machine learning potentials. We achieve this breakthrough by combining innovations in linear-scaling AIMD, efficient and approximate sparse linear algebra, low and mixed-precision floating-point computation on GPUs, and a compensation scheme for the errors introduced by numerical approximations.   The core of our work is the non-orthogonalized local submatrix (NOLSM) method, which scales very favorably to massively parallel computing systems and translates large sparse matrix operations into highly parallel, dense matrix operations that are ideally suited to hardware accelerators. We demonstrate that the NOLSM method, which is at the center point of each AIMD step, is able to achieve a sustained performance of 324 PFLOP/s in mixed FP16/FP32 precision corresponding to an efficiency of 67.7\% when running on 1536 NVIDIA A100 GPUs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Submatrix method reloaded:<br><br>We simulated &gt;100 million atoms w/ <a href="https://twitter.com/CP2Kproject?ref_src=twsrc%5Etfw">@CP2Kproject</a>  achieving 324 PLFLOPs (FP32/16) on JUWELS Booster at  <a href="https://twitter.com/fzj_jsc?ref_src=twsrc%5Etfw">@fzj_jsc</a><br><br>Enabling Electronic Structure-Based Ab-Initio Molecular Dynamics Simulations with Hundreds of Millions of Atoms<a href="https://t.co/6yAYnzIutL">https://t.co/6yAYnzIutL</a> <a href="https://t.co/3Mv27Xn2oI">pic.twitter.com/3Mv27Xn2oI</a></p>&mdash; Christian Plessl (@plessl) <a href="https://twitter.com/plessl/status/1384216075416440844?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale

Dmitry Kalashnikov, Jacob Varley, Yevgen Chebotar, Benjamin Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, Karol Hausman

- retweets: 123, favorites: 51 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.08212) | [pdf](https://arxiv.org/pdf/2104.08212)
- [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

General-purpose robotic systems must master a large repertoire of diverse skills to be useful in a range of daily tasks. While reinforcement learning provides a powerful framework for acquiring individual behaviors, the time needed to acquire each skill makes the prospect of a generalist robot trained with RL daunting. In this paper, we study how a large-scale collective robotic learning system can acquire a repertoire of behaviors simultaneously, sharing exploration, experience, and representations across tasks. In this framework new tasks can be continuously instantiated from previously learned tasks improving overall performance and capabilities of the system. To instantiate this system, we develop a scalable and intuitive framework for specifying new tasks through user-provided examples of desired outcomes, devise a multi-robot collective learning system for data collection that simultaneously collects experience for multiple tasks, and develop a scalable and generalizable multi-task deep reinforcement learning method, which we call MT-Opt. We demonstrate how MT-Opt can learn a wide range of skills, including semantic picking (i.e., picking an object from a particular category), placing into various fixtures (e.g., placing a food item onto a plate), covering, aligning, and rearranging. We train and evaluate our system on a set of 12 real-world tasks with data collected from 7 robots, and demonstrate the performance of our system both in terms of its ability to generalize to structurally similar new tasks, and acquire distinct new tasks more quickly by leveraging past experience. We recommend viewing the videos at https://karolhausman.github.io/mt-opt/

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale<br>pdf: <a href="https://t.co/6hDmIUgTdz">https://t.co/6hDmIUgTdz</a><br>abs: <a href="https://t.co/hk8xjgSSnj">https://t.co/hk8xjgSSnj</a><br>project page: <a href="https://t.co/hxgxDPQTTQ">https://t.co/hxgxDPQTTQ</a> <a href="https://t.co/Sf9bVBHUvs">pic.twitter.com/Sf9bVBHUvs</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383988280375476238?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Does BERT Pretrained on Clinical Notes Reveal Sensitive Data?

Eric Lehman, Sarthak Jain, Karl Pichotta, Yoav Goldberg, Byron C. Wallace

- retweets: 84, favorites: 81 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.07762) | [pdf](https://arxiv.org/pdf/2104.07762)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Large Transformers pretrained over clinical notes from Electronic Health Records (EHR) have afforded substantial gains in performance on predictive clinical tasks. The cost of training such models (and the necessity of data access to do so) coupled with their utility motivates parameter sharing, i.e., the release of pretrained models such as ClinicalBERT. While most efforts have used deidentified EHR, many researchers have access to large sets of sensitive, non-deidentified EHR with which they might train a BERT model (or similar). Would it be safe to release the weights of such a model if they did? In this work, we design a battery of approaches intended to recover Personal Health Information (PHI) from a trained BERT. Specifically, we attempt to recover patient names and conditions with which they are associated. We find that simple probing methods are not able to meaningfully extract sensitive information from BERT trained over the MIMIC-III corpus of EHR. However, more sophisticated "attacks" may succeed in doing so: To facilitate such research, we make our experimental setup and baseline probing models available at https://github.com/elehman16/exposing_patient_data_release

<blockquote class="twitter-tweet"><p lang="en" dir="ltr"><a href="https://twitter.com/hashtag/nlproc?src=hash&amp;ref_src=twsrc%5Etfw">#nlproc</a> Hi Folks, our NAACL paper &quot;Does BERT Pretrained on Clinical Notes Reveal Sensitive Data?&quot; is now available on arXiv <a href="https://t.co/f7F74ZVcK0">https://t.co/f7F74ZVcK0</a> (1/6) <a href="https://t.co/ZjMBlLaXAj">pic.twitter.com/ZjMBlLaXAj</a></p>&mdash; Sarthak Jain (@successar_nlp) <a href="https://twitter.com/successar_nlp/status/1384179401059835906?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep  Learning

Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, Yuxiong He

- retweets: 66, favorites: 51 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.07857) | [pdf](https://arxiv.org/pdf/2104.07857)
- [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.PF](https://arxiv.org/list/cs.PF/recent)

In the last three years, the largest dense deep learning models have grown over 1000x to reach hundreds of billions of parameters, while the GPU memory has only grown by 5x (16 GB to 80 GB). Therefore, the growth in model scale has been supported primarily though system innovations that allow large models to fit in the aggregate GPU memory of multiple GPUs. However, we are getting close to the GPU memory wall. It requires 800 NVIDIA V100 GPUs just to fit a trillion parameter model for training, and such clusters are simply out of reach for most data scientists. In addition, training models at that scale requires complex combinations of parallelism techniques that puts a big burden on the data scientists to refactor their model.   In this paper we present ZeRO-Infinity, a novel heterogeneous system technology that leverages GPU, CPU, and NVMe memory to allow for unprecedented model scale on limited resources without requiring model code refactoring. At the same time it achieves excellent training throughput and scalability, unencumbered by the limited CPU or NVMe bandwidth. ZeRO-Infinity can fit models with tens and even hundreds of trillions of parameters for training on current generation GPU clusters. It can be used to fine-tune trillion parameter models on a single NVIDIA DGX-2 node, making large models more accessible. In terms of training throughput and scalability, it sustains over 25 petaflops on 512 NVIDIA V100 GPUs(40% of peak), while also demonstrating super linear scalability. An open source implementation of ZeRO-Infinity is available through DeepSpeed, a deep learning optimization library that makes distributed training easy, efficient, and effective.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ZeRO-infinity paper has just been published <a href="https://t.co/4xaQLoKz7A">https://t.co/4xaQLoKz7A</a> - it uses CPU and NVMe offload, making the massive models even more accessible to everybody. The code should be available shortly!<br><br>Congrats to the <a href="https://twitter.com/hashtag/DeepSpeed?src=hash&amp;ref_src=twsrc%5Etfw">#DeepSpeed</a> team for this amazing accomplishment.</p>&mdash; Stas Bekman (@StasBekman) <a href="https://twitter.com/StasBekman/status/1384035928004464643?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Exploring Visual Engagement Signals for Representation Learning

Menglin Jia, Zuxuan Wu, Austin Reiter, Claire Cardie, Serge Belongie, Ser-Nam Lim

- retweets: 42, favorites: 40 (04/20/2021 09:43:50)

- links: [abs](https://arxiv.org/abs/2104.07767) | [pdf](https://arxiv.org/pdf/2104.07767)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Visual engagement in social media platforms comprises interactions with photo posts including comments, shares, and likes. In this paper, we leverage such visual engagement clues as supervisory signals for representation learning. However, learning from engagement signals is non-trivial as it is not clear how to bridge the gap between low-level visual information and high-level social interactions. We present VisE, a weakly supervised learning approach, which maps social images to pseudo labels derived by clustered engagement signals. We then study how models trained in this way benefit subjective downstream computer vision tasks such as emotion recognition or political bias detection. Through extensive studies, we empirically demonstrate the effectiveness of VisE across a diverse set of classification tasks beyond the scope of conventional recognition.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Exploring Visual Engagement Signals for Representation Learning<br>pdf: <a href="https://t.co/JoyItUXRRl">https://t.co/JoyItUXRRl</a><br>abs: <a href="https://t.co/QCxaiIi1ex">https://t.co/QCxaiIi1ex</a> <a href="https://t.co/o7lEskJJdA">pic.twitter.com/o7lEskJJdA</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1384004002400980997?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model  Stanislav Beliaev, Boris Ginsburgfor Speech Synthesis with Explicit Pitch and  Duration Prediction

Stanislav Beliaev, Boris Ginsburg

- retweets: 42, favorites: 39 (04/20/2021 09:43:51)

- links: [abs](https://arxiv.org/abs/2104.08189) | [pdf](https://arxiv.org/pdf/2104.08189)
- [eess.AS](https://arxiv.org/list/eess.AS/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

We propose TalkNet, a non-autoregressive convolutional neural model for speech synthesis with explicit pitch and duration prediction. The model consists of three feed-forward convolutional networks. The first network predicts grapheme durations. An input text is expanded by repeating each symbol according to the predicted duration. The second network predicts pitch value for every mel frame. The third network generates a mel-spectrogram from the expanded text conditioned on predicted pitch. All networks are based on 1D depth-wise separable convolutional architecture. The explicit duration prediction eliminates word skipping and repeating. The quality of the generated speech nearly matches the best auto-regressive models - TalkNet trained on the LJSpeech dataset got MOS4.08. The model has only 13.2M parameters, almost 2x less than the present state-of-the-art text-to-speech models. The non-autoregressive architecture allows for fast training and inference - 422x times faster than real-time. The small model size and fast inference make the TalkNet an attractive candidate for embedded speech synthesis.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">TalkNet 2: Non-Autoregressive Depth-Wise Separable Convolutional Model for Speech Synthesis with Explicit Pitch and Duration Prediction<br>pdf: <a href="https://t.co/Ul4y7DRJ2r">https://t.co/Ul4y7DRJ2r</a><br>abs: <a href="https://t.co/lz1s7Zc0DI">https://t.co/lz1s7Zc0DI</a> <a href="https://t.co/ZRT6oyeUU4">pic.twitter.com/ZRT6oyeUU4</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383952305536462852?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. An Adversarially-Learned Turing Test for Dialog Generation Models

Xiang Gao, Yizhe Zhang, Michel Galley, Bill Dolan

- retweets: 36, favorites: 25 (04/20/2021 09:43:51)

- links: [abs](https://arxiv.org/abs/2104.08231) | [pdf](https://arxiv.org/pdf/2104.08231)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

The design of better automated dialogue evaluation metrics offers the potential of accelerate evaluation research on conversational AI. However, existing trainable dialogue evaluation models are generally restricted to classifiers trained in a purely supervised manner, which suffer a significant risk from adversarial attacking (e.g., a nonsensical response that enjoys a high classification score). To alleviate this risk, we propose an adversarial training approach to learn a robust model, ATT (Adversarial Turing Test), that discriminates machine-generated responses from human-written replies. In contrast to previous perturbation-based methods, our discriminator is trained by iteratively generating unrestricted and diverse adversarial examples using reinforcement learning. The key benefit of this unrestricted adversarial training approach is allowing the discriminator to improve robustness in an iterative attack-defense game. Our discriminator shows high accuracy on strong attackers including DialoGPT and GPT-3.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">An Adversarially-Learned Turing Test for Dialog Generation Models<br>pdf: <a href="https://t.co/2H9bDKnSQq">https://t.co/2H9bDKnSQq</a><br>abs: <a href="https://t.co/U6x3FppIkC">https://t.co/U6x3FppIkC</a> <a href="https://t.co/RNXHuasdOo">pic.twitter.com/RNXHuasdOo</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383945473623433217?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. Rethinking Text Line Recognition Models

Daniel Hernandez Diaz, Siyang Qin, Reeve Ingle, Yasuhisa Fujii, Alessandro Bissacco

- retweets: 30, favorites: 29 (04/20/2021 09:43:51)

- links: [abs](https://arxiv.org/abs/2104.07787) | [pdf](https://arxiv.org/pdf/2104.07787)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

In this paper, we study the problem of text line recognition. Unlike most approaches targeting specific domains such as scene-text or handwritten documents, we investigate the general problem of developing a universal architecture that can extract text from any image, regardless of source or input modality. We consider two decoder families (Connectionist Temporal Classification and Transformer) and three encoder modules (Bidirectional LSTMs, Self-Attention, and GRCLs), and conduct extensive experiments to compare their accuracy and performance on widely used public datasets of scene and handwritten text. We find that a combination that so far has received little attention in the literature, namely a Self-Attention encoder coupled with the CTC decoder, when compounded with an external language model and trained on both public and internal data, outperforms all the others in accuracy and computational complexity. Unlike the more common Transformer-based models, this architecture can handle inputs of arbitrary length, a requirement for universal line recognition. Using an internal dataset collected from multiple sources, we also expose the limitations of current public datasets in evaluating the accuracy of line recognizers, as the relatively narrow image width and sequence length distributions do not allow to observe the quality degradation of the Transformer approach when applied to the transcription of long lines.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking Text Line Recognition Models<br>pdf: <a href="https://t.co/pzmVKhChHh">https://t.co/pzmVKhChHh</a><br>abs: <a href="https://t.co/tWGwjuaB6t">https://t.co/tWGwjuaB6t</a> <a href="https://t.co/bx4beLLkRp">pic.twitter.com/bx4beLLkRp</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1383946841851523075?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Memory Order Decomposition of Symbolic Sequences

Unai Alvarez-Rodriguez, Vito Latora

- retweets: 30, favorites: 20 (04/20/2021 09:43:51)

- links: [abs](https://arxiv.org/abs/2104.07798) | [pdf](https://arxiv.org/pdf/2104.07798)
- [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [cs.IT](https://arxiv.org/list/cs.IT/recent) | [physics.data-an](https://arxiv.org/list/physics.data-an/recent)

We introduce a general method for the study of memory in symbolic sequences based on higher-order Markov analysis. The Markov process that best represents a sequence is expressed as a mixture of matrices of minimal orders, enabling the definition of the so-called memory profile, which unambiguously reflects the true order of correlations. The method is validated by recovering the memory profiles of tunable synthetic sequences. Finally, we scan real data and showcase with practical examples how our protocol can be used to extract relevant stochastic properties of symbolic sequences.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Today V. Latora and myself release our⛓️higher-order Markov chain decomposition algorithm⛓️ <a href="https://t.co/kXKcCRwwpn">https://t.co/kXKcCRwwpn</a></p>&mdash; Unai Alvarez-Rodriguez (@unaialro) <a href="https://twitter.com/unaialro/status/1384017220062179334?ref_src=twsrc%5Etfw">April 19, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



