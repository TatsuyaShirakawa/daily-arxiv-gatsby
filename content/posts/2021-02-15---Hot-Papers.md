---
title: Hot Papers 2021-02-15
date: 2021-02-16T09:39:16.Z
template: "post"
draft: false
slug: "hot-papers-2021-02-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-02-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. Explaining Neural Scaling Laws

Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma

- retweets: 3332, favorites: 351 (02/16/2021 09:39:16)

- links: [abs](https://arxiv.org/abs/2102.06701) | [pdf](https://arxiv.org/pdf/2102.06701)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cond-mat.dis-nn](https://arxiv.org/list/cond-mat.dis-nn/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The test loss of well-trained neural networks often follows precise power-law scaling relations with either the size of the training dataset or the number of parameters in the network. We propose a theory that explains and connects these scaling laws. We identify variance-limited and resolution-limited scaling behavior for both dataset and model size, for a total of four scaling regimes. The variance-limited scaling follows simply from the existence of a well-behaved infinite data or infinite width limit, while the resolution-limited regime can be explained by positing that models are effectively resolving a smooth data manifold. In the large width limit, this can be equivalently obtained from the spectrum of certain kernels, and we present evidence that large width and large dataset resolution-limited scaling exponents are related by a duality. We exhibit all four scaling regimes in the controlled setting of large random feature and pretrained models and test the predictions empirically on a range of standard architectures and datasets. We also observe several empirical relationships between datasets and scaling exponents: super-classing image tasks does not change exponents, while changing input distribution (via changing datasets or adding noise) has a strong effect. We further explore the effect of architecture aspect ratio on scaling exponents.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Explaining Neural Scaling Laws<br><br>Proposes a theory that explains and connects various scaling laws concerning the size of the dataset, the number of parameters, resolution and variance. <a href="https://t.co/rq96u1mkyL">https://t.co/rq96u1mkyL</a> <a href="https://t.co/GWNzWOidBz">pic.twitter.com/GWNzWOidBz</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1361128053565349894?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. A Too-Good-to-be-True Prior to Reduce Shortcut Reliance

Nikolay Dagaev, Brett D. Roads, Xiaoliang Luo, Daniel N. Barry, Kaustubh R. Patil, Bradley C. Love

- retweets: 1768, favorites: 209 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06406) | [pdf](https://arxiv.org/pdf/2102.06406)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Despite their impressive performance in object recognition and other tasks under standard testing conditions, deep convolutional neural networks (DCNNs) often fail to generalize to out-of-distribution (o.o.d.) samples. One cause for this shortcoming is that modern architectures tend to rely on "shortcuts" - superficial features that correlate with categories without capturing deeper invariants that hold across contexts. Real-world concepts often possess a complex structure that can vary superficially across contexts, which can make the most intuitive and promising solutions in one context not generalize to others. One potential way to improve o.o.d. generalization is to assume simple solutions are unlikely to be valid across contexts and downweight them, which we refer to as the too-good-to-be-true prior. We implement this inductive bias in a two-stage approach that uses predictions from a low-capacity network (LCN) to inform the training of a high-capacity network (HCN). Since the shallow architecture of the LCN can only learn surface relationships, which includes shortcuts, we downweight training items for the HCN that the LCN can master, thereby encouraging the HCN to rely on deeper invariant features that should generalize broadly. Using a modified version of the CIFAR-10 dataset in which we introduced shortcuts, we found that the two-stage LCN-HCN approach reduced reliance on shortcuts and facilitated o.o.d. generalization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New preprint, &quot;A Too-Good-to-be-True Prior to Reduce Shortcut Reliance&quot;. If it&#39;s too good to be true, it probably is and that holds for deep learning as well. To generalize broadly, models need to learn invariants but instead are fooled by shortcuts. <a href="https://t.co/ylrZcVxSGS">https://t.co/ylrZcVxSGS</a> (1/4) <a href="https://t.co/gxPilisojd">pic.twitter.com/gxPilisojd</a></p>&mdash; Bradley Love (@ProfData) <a href="https://twitter.com/ProfData/status/1361278046486093830?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers  Suffice Across Batch Sizes

Zachary Nado, Justin M. Gilmer, Christopher J. Shallue, Rohan Anil, George E. Dahl

- retweets: 304, favorites: 91 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06356) | [pdf](https://arxiv.org/pdf/2102.06356)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Recently the LARS and LAMB optimizers have been proposed for training neural networks faster using large batch sizes. LARS and LAMB add layer-wise normalization to the update rules of Heavy-ball momentum and Adam, respectively, and have become popular in prominent benchmarks and deep learning libraries. However, without fair comparisons to standard optimizers, it remains an open question whether LARS and LAMB have any benefit over traditional, generic algorithms. In this work we demonstrate that standard optimization algorithms such as Nesterov momentum and Adam can match or exceed the results of LARS and LAMB at large batch sizes. Our results establish new, stronger baselines for future comparisons at these batch sizes and shed light on the difficulties of comparing optimizers for neural network training more generally.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Large Batch Optimizer Reality Check: Traditional, Generic Optimizers Suffice Across Batch Sizes<br><br>In fact, Nesterov momentum and Adam matches or exceeds the results of LARS and LAMB at large batch sizes.<a href="https://t.co/yI3L7CiuKX">https://t.co/yI3L7CiuKX</a> <a href="https://t.co/J8qJuiQA0D">pic.twitter.com/J8qJuiQA0D</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1361130705657294848?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Banana for scale: Gauging trends in academic interest by normalising  publication rates to common and innocuous keywords

Edwin S. Dalmaijer, Joram Van Rheede, Edwin V. Sperr, Juliane Tkotz

- retweets: 306, favorites: 67 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06418) | [pdf](https://arxiv.org/pdf/2102.06418)
- [cs.DL](https://arxiv.org/list/cs.DL/recent)

Many academics use yearly publication numbers to quantify academic interest for their research topic. While such visualisations are ubiquitous in grant applications, manuscript introductions, and review articles, they fail to account for the rapid growth in scientific publications. As a result, any search term will likely show an increase in supposed "academic interest". One proposed solution is to normalise yearly publication rates by field size, but this is arduous and difficult. Here, we propose an simpler index that normalises keywords of interest by a ubiquitous and innocuous keyword, such as "banana". Alternatively, one could opt for field-specific keywords or hierarchical structures (e.g. PubMed's Medical Subject Headings, MeSH) to compute "interest market share". Using this approach, we uncovered plausible trends in academic interest in examples from the medical literature. In neuroimaging, we found that not the supplementary motor area (as was previously claimed), but the prefrontal cortex is the most interesting part of the brain. In cancer research, we found a contemporary preference for cancers with high prevalence and clinical severity, and notable declines in interest for more treatable or likely benign neoplasms. Finally, we found that interest in respiratory viral infections spiked when strains showed potential for pandemic involvement, with SARS-CoV-2 and the COVID-19 pandemic being the most extreme example. In sum, the time is ripe for a quick and easy method to quantify trends in academic interest for anecdotal purposes. We provide such a method, along with software for researchers looking to implement it in their own writing.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Writing a grant or review? Highlighting changes in interest for the topic? That&#39;s hard when publication rates skyrocket everywhere! Here&#39;s a simple solution: just use a banana for scale! 🍌<br>Manuscript with <a href="https://twitter.com/Joranium?ref_src=twsrc%5Etfw">@Joranium</a> <a href="https://twitter.com/edsperr?ref_src=twsrc%5Etfw">@edsperr</a> <a href="https://twitter.com/juli_tkotz?ref_src=twsrc%5Etfw">@juli_tkotz</a> here: <a href="https://t.co/uS0mAOk1re">https://t.co/uS0mAOk1re</a><br>Thread below! <a href="https://t.co/3KN7wedEGm">pic.twitter.com/3KN7wedEGm</a></p>&mdash; Edwin Dalmaijer (@esdalmaijer) <a href="https://twitter.com/esdalmaijer/status/1361277638602612736?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Why Don't Developers Detect Improper Input Validation?'; DROP TABLE  Papers; --

Larissa Braz, Enrico Fregnan, Gül Çalikli, Alberto Bacchelli

- retweets: 210, favorites: 41 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06251) | [pdf](https://arxiv.org/pdf/2102.06251)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Improper Input Validation (IIV) is a software vulnerability that occurs when a system does not safely handle input data. Even though IIV is easy to detect and fix, it still commonly happens in practice. In this paper, we study to what extent developers can detect IIV and investigate underlying reasons. This knowledge is essential to better understand how to support developers in creating secure software systems. We conduct an online experiment with 146 participants, of which 105 report at least three years of professional software development experience. Our results show that the existence of a visible attack scenario facilitates the detection of IIV vulnerabilities and that a significant portion of developers who did not find the vulnerability initially could identify it when warned about its existence. Yet, a total of 60 participants could not detect the vulnerability even after the warning. Other factors, such as the frequency with which the participants perform code reviews, influence the detection of IIV. Data and materials: https://doi.org/10.5281/zenodo.3996696

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our &quot;Why Don’t Developers Detect Improper Input Validation? &#39;; DROP TABLE Papers; --&quot; <a href="https://twitter.com/ICSEconf?ref_src=twsrc%5Etfw">@ICSEconf</a> 2021 paper pre-print is now available! <br>check it out: <a href="https://t.co/fn0oBuZLRH">https://t.co/fn0oBuZLRH</a><a href="https://twitter.com/EnFregnan?ref_src=twsrc%5Etfw">@EnFregnan</a> <a href="https://twitter.com/GulCalikli?ref_src=twsrc%5Etfw">@GulCalikli</a> <a href="https://twitter.com/sback_?ref_src=twsrc%5Etfw">@sback_</a></p>&mdash; Larissa Braz (@larissabrazb) <a href="https://twitter.com/larissabrazb/status/1361318781839114243?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Optimizing Inference Performance of Transformers on CPUs

Dave Dice, Alex Kogan

- retweets: 143, favorites: 48 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06621) | [pdf](https://arxiv.org/pdf/2102.06621)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.DC](https://arxiv.org/list/cs.DC/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.MS](https://arxiv.org/list/cs.MS/recent)

The Transformer architecture revolutionized the field of natural language processing (NLP). Transformers-based models (e.g., BERT) power many important Web services, such as search, translation, question-answering, etc. While enormous research attention is paid to the training of those models, relatively little efforts are made to improve their inference performance. This paper comes to address this gap by presenting an empirical analysis of scalability and performance of inferencing a Transformer-based model on CPUs. Focusing on the highly popular BERT model, we identify key components of the Transformer architecture where the bulk of the computation happens, and propose three optimizations to speed them up. The optimizations are evaluated using the inference benchmark from HuggingFace, and are shown to achieve the speedup of up to x2.36. The considered optimizations do not require any changes to the implementation of the models nor affect their accuracy.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Optimizing Inference Performance of Transformers on CPUs<br>pdf: <a href="https://t.co/PViwYSdCVy">https://t.co/PViwYSdCVy</a><br>abs: <a href="https://t.co/L4DiVypnOH">https://t.co/L4DiVypnOH</a> <a href="https://t.co/Ave1s4XTXj">pic.twitter.com/Ave1s4XTXj</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361133005151076363?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Improving Object Detection in Art Images Using Only Style Transfer

David Kadish, Sebastian Risi, Anders Sundnes Løvlie

- retweets: 90, favorites: 72 (02/16/2021 09:39:17)

- links: [abs](https://arxiv.org/abs/2102.06529) | [pdf](https://arxiv.org/pdf/2102.06529)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

Despite recent advances in object detection using deep learning neural networks, these neural networks still struggle to identify objects in art images such as paintings and drawings. This challenge is known as the cross depiction problem and it stems in part from the tendency of neural networks to prioritize identification of an object's texture over its shape. In this paper we propose and evaluate a process for training neural networks to localize objects - specifically people - in art images. We generate a large dataset for training and validation by modifying the images in the COCO dataset using AdaIn style transfer. This dataset is used to fine-tune a Faster R-CNN object detection network, which is then tested on the existing People-Art testing dataset. The result is a significant improvement on the state of the art and a new way forward for creating datasets to train neural networks to process art images.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Improving Object Detection in Art Images Using Only Style Transfer<br>pdf: <a href="https://t.co/kxMdMmurys">https://t.co/kxMdMmurys</a><br>abs: <a href="https://t.co/IBkfF1l2Gz">https://t.co/IBkfF1l2Gz</a> <a href="https://t.co/frEYjvuEYU">pic.twitter.com/frEYjvuEYU</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361142081884987396?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. How do climate change skeptics engage with opposing views? Understanding  mechanisms of social identity and cognitive dissonance in an online forum

Lisa Oswald, Jonathan Bright

- retweets: 56, favorites: 64 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06516) | [pdf](https://arxiv.org/pdf/2102.06516)
- [cs.SI](https://arxiv.org/list/cs.SI/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Does engagement with opposing views help break down ideological `echo chambers'; or does it backfire and reinforce them? This question remains critical as academics, policymakers and activists grapple with the question of how to regulate political discussion on social media. In this study, we contribute to the debate by examining the impact of opposing views within a major climate change skeptic online community on Reddit. A large sample of posts (N = 3000) was manually coded as either dissonant or consonant which allowed the automated classification of the full dataset of more than 50,000 posts, with codes inferred from linked websites. We find that ideologically dissonant submissions act as a stimulant to activity in the community: they received more attention (comments) than consonant submissions, even though they received lower scores through up-voting and down-voting. Users who engaged with dissonant submissions were also more likely to return to the forum. Consistent with identity theory, confrontation with opposing views triggered activity in the forum, particularly among users that are highly engaged with the community. In light of the findings, theory of social identity and echo chambers is discussed and enhanced.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How do climate change skeptics engage with opposing views? Understanding mechanisms of social identity and cognitive dissonance in an online forum - Fresh draft with <a href="https://twitter.com/jonmbright?ref_src=twsrc%5Etfw">@jonmbright</a> out on <a href="https://t.co/RRcHNGXnNd">https://t.co/RRcHNGXnNd</a> ! ✨ <a href="https://t.co/R7V3zYbknB">pic.twitter.com/R7V3zYbknB</a></p>&mdash; Lisa Oswald (@LisaFOswaldo) <a href="https://twitter.com/LisaFOswaldo/status/1361265629328736256?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. VARA-TTS: Non-Autoregressive Text-to-Speech Synthesis based on Very Deep  VAE with Residual Attention

Peng Liu, Yuewen Cao, Songxiang Liu, Na Hu, Guangzhi Li, Chao Weng, Dan Su

- retweets: 37, favorites: 66 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06431) | [pdf](https://arxiv.org/pdf/2102.06431)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper proposes VARA-TTS, a non-autoregressive (non-AR) text-to-speech (TTS) model using a very deep Variational Autoencoder (VDVAE) with Residual Attention mechanism, which refines the textual-to-acoustic alignment layer-wisely. Hierarchical latent variables with different temporal resolutions from the VDVAE are used as queries for residual attention module. By leveraging the coarse global alignment from previous attention layer as an extra input, the following attention layer can produce a refined version of alignment. This amortizes the burden of learning the textual-to-acoustic alignment among multiple attention layers and outperforms the use of only a single attention layer in robustness. An utterance-level speaking speed factor is computed by a jointly-trained speaking speed predictor, which takes the mean-pooled latent variables of the coarsest layer as input, to determine number of acoustic frames at inference. Experimental results show that VARA-TTS achieves slightly inferior speech quality to an AR counterpart Tacotron 2 but an order-of-magnitude speed-up at inference; and outperforms an analogous non-AR model, BVAE-TTS, in terms of speech quality.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VARA-TTS: Non-Autoregressive Text-to-Speech Synthesis based on Very Deep VAE with Residual Attention<br><br>Proposes a nonautoregressive end-to-end text-tospeech model that performs close to Tacotoron 2 with substantially faster inference speed. <a href="https://t.co/wEh5yDXas7">https://t.co/wEh5yDXas7</a> <a href="https://t.co/jlXlHMP1Mw">pic.twitter.com/jlXlHMP1Mw</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1361131941366439939?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">VARA-TTS: Non-Autoregressive Text-to-Speech Synthesis based on Very Deep VAE with Residual Attention<br>pdf: <a href="https://t.co/yiprp0IeDK">https://t.co/yiprp0IeDK</a><br>abs: <a href="https://t.co/mSg7OUyF1w">https://t.co/mSg7OUyF1w</a><br>project page: <a href="https://t.co/KpQg9S9TOT">https://t.co/KpQg9S9TOT</a> <a href="https://t.co/60DyWjs6NR">pic.twitter.com/60DyWjs6NR</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361139755736522752?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Efficient Conditional GAN Transfer with Knowledge Propagation across  Classes

Mohamad Shahbazi, Zhiwu Huang, Danda Pani Paudel, Ajad Chhatkuli, Luc Van Gool

- retweets: 72, favorites: 26 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06696) | [pdf](https://arxiv.org/pdf/2102.06696)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Generative adversarial networks (GANs) have shown impressive results in both unconditional and conditional image generation. In recent literature, it is shown that pre-trained GANs, on a different dataset, can be transferred to improve the image generation from a small target data. The same, however, has not been well-studied in the case of conditional GANs (cGANs), which provides new opportunities for knowledge transfer compared to unconditional setup. In particular, the new classes may borrow knowledge from the related old classes, or share knowledge among themselves to improve the training. This motivates us to study the problem of efficient conditional GAN transfer with knowledge propagation across classes. To address this problem, we introduce a new GAN transfer method to explicitly propagate the knowledge from the old classes to the new classes. The key idea is to enforce the popularly used conditional batch normalization (BN) to learn the class-specific information of the new classes from that of the old classes, with implicit knowledge sharing among the new ones. This allows for an efficient knowledge propagation from the old classes to the new classes, with the BN parameters increasing linearly with the number of new classes. The extensive evaluation demonstrates the clear superiority of the proposed method over state-of-the-art competitors for efficient conditional GAN transfer tasks. The code will be available at: https://github.com/mshahbazi72/cGANTransfer

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Efficient Conditional GAN Transfer with Knowledge Propagation across Classes<br>pdf: <a href="https://t.co/qKojLkD9AU">https://t.co/qKojLkD9AU</a><br>abs: <a href="https://t.co/qg5XUVxYBm">https://t.co/qg5XUVxYBm</a> <a href="https://t.co/x0iadNTNRK">pic.twitter.com/x0iadNTNRK</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361137218174214151?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Same File, Different Changes: The Potential of Meta-Maintenance on  GitHub

Hideaki Hata, Raula Gaikovina Kula, Takashi Ishio, Christoph Treude

- retweets: 42, favorites: 19 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06355) | [pdf](https://arxiv.org/pdf/2102.06355)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

Online collaboration platforms such as GitHub have provided software developers with the ability to easily reuse and share code between repositories. With clone-and-own and forking becoming prevalent, maintaining these shared files is important, especially for keeping the most up-to-date version of reused code. Different to related work, we propose the concept of meta-maintenance -- i.e., tracking how the same files evolve in different repositories with the aim to provide useful maintenance opportunities to those files. We conduct an exploratory study by analyzing repositories from seven different programming languages to explore the potential of meta-maintenance. Our results indicate that a majority of active repositories on GitHub contains at least one file which is also present in another repository, and that a significant minority of these files are maintained differently in the different repositories which contain them. We manually analyzed a representative sample of shared files and their variants to understand which changes might be useful for meta-maintenance. Our findings support the potential of meta-maintenance and open up avenues for future work to capitalize on this potential.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our <a href="https://twitter.com/ICSEconf?ref_src=twsrc%5Etfw">@ICSEconf</a> 2021 paper we (<a href="https://twitter.com/Augaiko?ref_src=twsrc%5Etfw">@Augaiko</a>, Takashi Ishio, <a href="https://twitter.com/ctreude?ref_src=twsrc%5Etfw">@ctreude</a>) propose &#39;meta-maintenance&#39;, a concept for maintaining the entire software ecosystem.<br> <a href="https://twitter.com/hashtag/icsePromo?src=hash&amp;ref_src=twsrc%5Etfw">#icsePromo</a><br><br>Preprint: <a href="https://t.co/Mo8Gn7mEyw">https://t.co/Mo8Gn7mEyw</a><br>Data: <a href="https://t.co/tyq7YYF5MR">https://t.co/tyq7YYF5MR</a></p>&mdash; Hideaki Hata (@hideakihata) <a href="https://twitter.com/hideakihata/status/1361198098588962817?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Multiversal views on language models

Laria Reynolds, Kyle McDonell

- retweets: 20, favorites: 37 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06391) | [pdf](https://arxiv.org/pdf/2102.06391)
- [cs.HC](https://arxiv.org/list/cs.HC/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

The virtuosity of language models like GPT-3 opens a new world of possibility for human-AI collaboration in writing. In this paper, we present a framework in which generative language models are conceptualized as multiverse generators. This framework also applies to human imagination and is core to how we read and write fiction. We call for exploration into this commonality through new forms of interfaces which allow humans to couple their imagination to AI to write, explore, and understand non-linear fiction. We discuss the early insights we have gained from actively pursuing this approach by developing and testing a novel multiversal GPT-3-assisted writing interface.

<blockquote class="twitter-tweet"><p lang="fr" dir="ltr">Multiversal views on language models<br>pdf: <a href="https://t.co/jSptVyWkP7">https://t.co/jSptVyWkP7</a><br>abs: <a href="https://t.co/M5pn2i9NUw">https://t.co/M5pn2i9NUw</a> <a href="https://t.co/5VU7iDhSM6">pic.twitter.com/5VU7iDhSM6</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361130831331401729?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. End-to-end Audio-visual Speech Recognition with Conformers

Pingchuan Ma, Stavros Petridis, Maja Pantic

- retweets: 20, favorites: 31 (02/16/2021 09:39:18)

- links: [abs](https://arxiv.org/abs/2102.06657) | [pdf](https://arxiv.org/pdf/2102.06657)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

In this work, we present a hybrid CTC/Attention model based on a ResNet-18 and Convolution-augmented transformer (Conformer), that can be trained in an end-to-end manner. In particular, the audio and visual encoders learn to extract features directly from raw pixels and audio waveforms, respectively, which are then fed to conformers and then fusion takes place via a Multi-Layer Perceptron (MLP). The model learns to recognise characters using a combination of CTC and an attention mechanism. We show that end-to-end training, instead of using pre-computed visual features which is common in the literature, the use of a conformer, instead of a recurrent network, and the use of a transformer-based language model, significantly improve the performance of our model. We present results on the largest publicly available datasets for sentence-level speech recognition, Lip Reading Sentences 2 (LRS2) and Lip Reading Sentences 3 (LRS3), respectively. The results show that our proposed models raise the state-of-the-art performance by a large margin in audio-only, visual-only, and audio-visual experiments.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">End-to-end Audio-visual Speech Recognition with Conformers<br>pdf: <a href="https://t.co/f3HDIZpWnK">https://t.co/f3HDIZpWnK</a><br>abs: <a href="https://t.co/EYBm5zeiGX">https://t.co/EYBm5zeiGX</a> <a href="https://t.co/YAOMTo167b">pic.twitter.com/YAOMTo167b</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1361131943102984193?ref_src=twsrc%5Etfw">February 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. The Software Heritage Filesystem (SwhFS): Integrating Source Code  Archival with Development

Thibault Allançon, Antoine Pietri, Stefano Zacchiroli

- retweets: 42, favorites: 8 (02/16/2021 09:39:19)

- links: [abs](https://arxiv.org/abs/2102.06390) | [pdf](https://arxiv.org/pdf/2102.06390)
- [cs.SE](https://arxiv.org/list/cs.SE/recent)

We introduce the Software Heritage filesystem (SwhFS), a user-space filesystem that integrates large-scale open source software archival with development workflows. SwhFS provides a POSIX filesystem view of Software Heritage, the largest public archive of software source code and version control system (VCS) development history.Using SwhFS, developers can quickly "checkout" any of the 2 billion commits archived by Software Heritage, even after they disappear from their previous known location and without incurring the performance cost of repository cloning. SwhFS works across unrelated repositories and different VCS technologies. Other source code artifacts archived by Software Heritage-individual source code files and trees, releases, and branches-can also be accessed using common programming tools and custom scripts, as if they were locally available.A screencast of SwhFS is available online at dx.doi.org/10.5281/zenodo.4531411.



