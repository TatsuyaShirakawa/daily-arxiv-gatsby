---
title: Hot Papers 2021-01-26
date: 2021-01-27T10:46:54.Z
template: "post"
draft: false
slug: "hot-papers-2021-01-26"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-01-26"
socialImage: "/media/flying-marine.jpg"

---

# 1. GUIGAN: Learning to Generate GUI Designs Using Generative Adversarial  Networks

Tianming Zhao, Chunyang Chen, Yuanning Liu, Xiaodong Zhu

- retweets: 420, favorites: 69 (01/27/2021 10:46:54)

- links: [abs](https://arxiv.org/abs/2101.09978) | [pdf](https://arxiv.org/pdf/2101.09978)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Graphical User Interface (GUI) is ubiquitous in almost all modern desktop software, mobile applications, and online websites. A good GUI design is crucial to the success of the software in the market, but designing a good GUI which requires much innovation and creativity is difficult even to well-trained designers. Besides, the requirement of the rapid development of GUI design also aggravates designers' working load. So, the availability of various automated generated GUIs can help enhance the design personalization and specialization as they can cater to the taste of different designers. To assist designers, we develop a model GUIGAN to automatically generate GUI designs. Different from conventional image generation models based on image pixels, our GUIGAN is to reuse GUI components collected from existing mobile app GUIs for composing a new design that is similar to natural-language generation. Our GUIGAN is based on SeqGAN by modeling the GUI component style compatibility and GUI structure. The evaluation demonstrates that our model significantly outperforms the best of the baseline methods by 30.77% in Frechet Inception distance (FID) and 12.35% in 1-Nearest Neighbor Accuracy (1-NNA). Through a pilot user study, we provide initial evidence of the usefulness of our approach for generating acceptable brand new GUI designs.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GUIGAN: Learning to Generate GUI Designs Using Generative Adversarial Networks<br>pdf: <a href="https://t.co/deJPAI0lJj">https://t.co/deJPAI0lJj</a><br>abs: <a href="https://t.co/jMIb7UYrC3">https://t.co/jMIb7UYrC3</a><br>github: <a href="https://t.co/rA6O3No5ml">https://t.co/rA6O3No5ml</a> <a href="https://t.co/kahz66VUys">pic.twitter.com/kahz66VUys</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1353925000336728064?ref_src=twsrc%5Etfw">January 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Memory-Efficient Semi-Supervised Continual Learning: The World is its  Own Replay Buffer

James Smith, Jonathan Balloch, Yen-Chang Hsu, Zsolt Kira

- retweets: 230, favorites: 99 (01/27/2021 10:46:54)

- links: [abs](https://arxiv.org/abs/2101.09536) | [pdf](https://arxiv.org/pdf/2101.09536)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Rehearsal is a critical component for class-incremental continual learning, yet it requires a substantial memory budget. Our work investigates whether we can significantly reduce this memory budget by leveraging unlabeled data from an agent's environment in a realistic and challenging continual learning paradigm. Specifically, we explore and formalize a novel semi-supervised continual learning (SSCL) setting, where labeled data is scarce yet non-i.i.d. unlabeled data from the agent's environment is plentiful. Importantly, data distributions in the SSCL setting are realistic and therefore reflect object class correlations between, and among, the labeled and unlabeled data distributions. We show that a strategy built on pseudo-labeling, consistency regularization, Out-of-Distribution (OoD) detection, and knowledge distillation reduces forgetting in this setting. Our approach, DistillMatch, increases performance over the state-of-the-art by no less than 8.7% average task accuracy and up to a 54.5% increase in average task accuracy in SSCL CIFAR-100 experiments. Moreover, we demonstrate that DistillMatch can save up to 0.23 stored images per processed unlabeled image compared to the next best method which only saves 0.08. Our results suggest that focusing on realistic correlated distributions is a significantly new perspective, which accentuates the importance of leveraging the world's structure as a continual learning strategy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Memory-Efficient Semi-Supervised Continual Learning: The World is its Own Replay Buffer<br>pdf: <a href="https://t.co/1sdMip6kuw">https://t.co/1sdMip6kuw</a><br>abs: <a href="https://t.co/a3gE2Wheli">https://t.co/a3gE2Wheli</a> <a href="https://t.co/q3G2PkTLxa">pic.twitter.com/q3G2PkTLxa</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1353910987750051841?ref_src=twsrc%5Etfw">January 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Hypergraph clustering: from blockmodels to modularity

Philip S. Chodrow, Nate Veldt, Austin R. Benson

- retweets: 188, favorites: 60 (01/27/2021 10:46:54)

- links: [abs](https://arxiv.org/abs/2101.09611) | [pdf](https://arxiv.org/pdf/2101.09611)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.DM](https://arxiv.org/list/cs.DM/recent) | [physics.data-an](https://arxiv.org/list/physics.data-an/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Hypergraphs are a natural modeling paradigm for a wide range of complex relational systems with multibody interactions. A standard analysis task is to identify clusters of closely related or densely interconnected nodes. While many probabilistic generative models for graph clustering have been proposed, there are relatively few such models for hypergraphs. We propose a Poisson degree-corrected hypergraph stochastic blockmodel (DCHSBM), an expressive generative model of clustered hypergraphs with heterogeneous node degrees and edge sizes. Maximum-likelihood inference in the DCHSBM naturally leads to a clustering objective that generalizes the popular modularity objective for graphs. We derive a general Louvain-type algorithm for this objective, as well as a a faster, specialized "All-Or-Nothing" (AON) variant in which edges are expected to lie fully within clusters. This special case encompasses a recent proposal for modularity in hypergraphs, while also incorporating flexible resolution and edge-size parameters. We show that hypergraph Louvain is highly scalable, including as an example an experiment on a synthetic hypergraph of one million nodes. We also demonstrate through synthetic experiments that the detectability regimes for hypergraph community detection differ from methods based on dyadic graph projections. In particular, there are regimes in which hypergraph methods can recover planted partitions even though graph based methods necessarily fail due to information-theoretic limits. We use our model to analyze different patterns of higher-order structure in school contact networks, U.S. congressional bill cosponsorship, U.S. congressional committees, product categories in co-purchasing behavior, and hotel locations from web browsing sessions, that it is able to recover ground truth clusters in empirical data sets exhibiting the corresponding higher-order structure.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint y&#39;all! 🎉&quot;Generative hypergraph clustering: from blockmodels to modularity&quot; now out on arXiv (<a href="https://t.co/z5uLqiXhLX">https://t.co/z5uLqiXhLX</a>). Joint work with dream team Nate Veldt (<a href="https://twitter.com/n_veldt?ref_src=twsrc%5Etfw">@n_veldt</a>) and Austin Benson (<a href="https://twitter.com/austinbenson?ref_src=twsrc%5Etfw">@austinbenson</a>). Nate is on the job market---HIRE HIM, he&#39;s awesome! 🧵🧵🧵</p>&mdash; Dr. Phil Chodrow (@PhilChodrow) <a href="https://twitter.com/PhilChodrow/status/1354154458989555712?ref_src=twsrc%5Etfw">January 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Explainable Artificial Intelligence Approaches: A Survey

Sheikh Rabiul Islam, William Eberle, Sheikh Khaled Ghafoor, Mohiuddin Ahmed

- retweets: 139, favorites: 18 (01/27/2021 10:46:55)

- links: [abs](https://arxiv.org/abs/2101.09429) | [pdf](https://arxiv.org/pdf/2101.09429)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The lack of explainability of a decision from an Artificial Intelligence (AI) based "black box" system/model, despite its superiority in many real-world applications, is a key stumbling block for adopting AI in many high stakes applications of different domain or industry. While many popular Explainable Artificial Intelligence (XAI) methods or approaches are available to facilitate a human-friendly explanation of the decision, each has its own merits and demerits, with a plethora of open challenges. We demonstrate popular XAI methods with a mutual case study/task (i.e., credit default prediction), analyze for competitive advantages from multiple perspectives (e.g., local, global), provide meaningful insight on quantifying explainability, and recommend paths towards responsible or human-centered AI using XAI as a medium. Practitioners can use this work as a catalog to understand, compare, and correlate competitive advantages of popular XAI methods. In addition, this survey elicits future research directions towards responsible or human-centric AI systems, which is crucial to adopt AI in high stakes applications.




# 5. Variational Neural Annealing

Mohamed Hibat-Allah, Estelle M. Inack, Roeland Wiersema, Roger G. Melko, Juan Carrasquilla

- retweets: 64, favorites: 34 (01/27/2021 10:46:55)

- links: [abs](https://arxiv.org/abs/2101.10154) | [pdf](https://arxiv.org/pdf/2101.10154)
- [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [quant-ph](https://arxiv.org/list/quant-ph/recent)

Many important challenges in science and technology can be cast as optimization problems. When viewed in a statistical physics framework, these can be tackled by simulated annealing, where a gradual cooling procedure helps search for groundstate solutions of a target Hamiltonian. While powerful, simulated annealing is known to have prohibitively slow sampling dynamics when the optimization landscape is rough or glassy. Here we show that by generalizing the target distribution with a parameterized model, an analogous annealing framework based on the variational principle can be used to search for groundstate solutions. Modern autoregressive models such as recurrent neural networks provide ideal parameterizations since they can be exactly sampled without slow dynamics even when the model encodes a rough landscape. We implement this procedure in the classical and quantum settings on several prototypical spin glass Hamiltonians, and find that it significantly outperforms traditional simulated annealing in the asymptotic limit, illustrating the potential power of this yet unexplored route to optimization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Variational Neural Annealing <a href="https://t.co/9claeWpmnG">https://t.co/9claeWpmnG</a>  <a href="https://twitter.com/MedHibatAllah?ref_src=twsrc%5Etfw">@MedHibatAllah</a>  <a href="https://twitter.com/InackEstelle?ref_src=twsrc%5Etfw">@InackEstelle</a> Roeland Wiersema <a href="https://twitter.com/rgmelko?ref_src=twsrc%5Etfw">@rgmelko</a>  <a href="https://twitter.com/VectorInst?ref_src=twsrc%5Etfw">@VectorInst</a> <a href="https://twitter.com/UWaterloo?ref_src=twsrc%5Etfw">@UWaterloo</a> <a href="https://twitter.com/Perimeter?ref_src=twsrc%5Etfw">@Perimeter</a></p>&mdash; Juan Felipe Carrasquilla Álvarez (@carrasqu) <a href="https://twitter.com/carrasqu/status/1354058187423539201?ref_src=twsrc%5Etfw">January 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. SGD-Net: Efficient Model-Based Deep Learning with Theoretical Guarantees

Jiaming Liu, Yu Sun, Weijie Gan, Xiaojian Xu, Brendt Wohlberg, Ulugbek S. Kamilov

- retweets: 64, favorites: 16 (01/27/2021 10:46:55)

- links: [abs](https://arxiv.org/abs/2101.09379) | [pdf](https://arxiv.org/pdf/2101.09379)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Deep unfolding networks have recently gained popularity in the context of solving imaging inverse problems. However, the computational and memory complexity of data-consistency layers within traditional deep unfolding networks scales with the number of measurements, limiting their applicability to large-scale imaging inverse problems. We propose SGD-Net as a new methodology for improving the efficiency of deep unfolding through stochastic approximations of the data-consistency layers. Our theoretical analysis shows that SGD-Net can be trained to approximate batch deep unfolding networks to an arbitrary precision. Our numerical results on intensity diffraction tomography and sparse-view computed tomography show that SGD-Net can match the performance of the batch network at a fraction of training and testing complexity.




# 7. The Shifting Sands of Motivation: Revisiting What Drives Contributors in  Open Source

Marco Gerosa, Igor Wiese, Bianca Trinkenreich, Georg Link, Gregorio Robles, Christoph Treude, Igor Steinmacher, Anita Sarma

- retweets: 56, favorites: 22 (01/27/2021 10:46:55)

- links: [abs](https://arxiv.org/abs/2101.10291) | [pdf](https://arxiv.org/pdf/2101.10291)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Open Source Software (OSS) has changed drastically over the last decade, with OSS projects now producing a large ecosystem of popular products, involving industry participation, and providing professional career opportunities. But our field's understanding of what motivates people to contribute to OSS is still fundamentally grounded in studies from the early 2000s. With the changed landscape of OSS, it is very likely that motivations to join OSS have also evolved. Through a survey of 242 OSS contributors, we investigate shifts in motivation from three perspectives: (1) the impact of the new OSS landscape, (2) the impact of individuals' personal growth as they become part of OSS communities, and (3) the impact of differences in individuals' demographics. Our results show that some motivations related to social aspects and reputation increased in frequency and that some intrinsic and internalized motivations, such as learning and intellectual stimulation, are still highly relevant. We also found that contributing to OSS often transforms extrinsic motivations to intrinsic, and that while experienced contributors often shift toward altruism, novices often shift toward career, fun, kinship, and learning. OSS projects can leverage our results to revisit current strategies to attract and retain contributors, and researchers and tool builders can better support the design of new studies and tools to engage and support OSS development.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are the OSS motivations changing? YEAH! Check out i) the impact of the new OSS landscape, (ii) the impact of individuals&#39; personal growth as they become part of OSS communities, and (ii) the impact of differences in individuals&#39; demographics.  <a href="https://twitter.com/hashtag/icsepromo?src=hash&amp;ref_src=twsrc%5Etfw">#icsepromo</a> <a href="https://t.co/TGLPJlVcRP">https://t.co/TGLPJlVcRP</a></p>&mdash; Igor Scaliante Wiese (@IgorWiese) <a href="https://twitter.com/IgorWiese/status/1353963930071166976?ref_src=twsrc%5Etfw">January 26, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. JuTrack: A Digital Biomarker Platform for Remote Monitoring in  Neuropsychiatric and Psychiatric Diseases

Mehran Sahandi Far, Michael Stolz, Jona M. Fischer, Simon B. Eickhoff, Juergen Dukart

- retweets: 56, favorites: 17 (01/27/2021 10:46:55)

- links: [abs](https://arxiv.org/abs/2101.10091) | [pdf](https://arxiv.org/pdf/2101.10091)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

Objective: Health-related data being collected by smartphones offer a promising complementary approach to in-clinic assessments. Here we introduce the JuTrack platform as a secure, reliable and extendable open-source solution for remote monitoring in daily-life and digital phenotyping. Method: JuTrack consists of an Android-based smartphone application and a web-based project management dashboard. A wide range of anonymized measurements from motion-sensors, social and physical activities and geolocation information can be collected in either active or passive modes. The dashboard also provides management tools to monitor and manage data collection across studies. To facilitate scaling, reproducibility, data management and sharing we integrated DataLad as a data management infrastructure. JuTrack was developed to comply with security, privacy and the General Data Protection Regulation (GDPR) requirements. Results: JuTrack is an open-source (released under open-source Apache 2.0 licenses) platform for remote assessment of digital biomarkers (DB) in neurological, psychiatric and other indications. The main components of the JuTrack platform and example of data being collected using JuTrack are presented here. Conclusion: Smartphone-based Digital Biomarker data may provide valuable insight into daily life behaviour in health and disease. JuTrack provides an easy and reliable open-source solution for collection of such data.



