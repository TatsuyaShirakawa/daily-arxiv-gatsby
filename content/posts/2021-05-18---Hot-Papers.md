---
title: Hot Papers 2021-05-18
date: 2021-05-19T09:44:42.Z
template: "post"
draft: false
slug: "hot-papers-2021-05-18"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-05-18"
socialImage: "/media/flying-marine.jpg"

---

# 1. Rethinking "Batch" in BatchNorm

Yuxin Wu, Justin Johnson

- retweets: 7553, favorites: 534 (05/19/2021 09:44:42)

- links: [abs](https://arxiv.org/abs/2105.07576) | [pdf](https://arxiv.org/pdf/2105.07576)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

BatchNorm is a critical building block in modern convolutional neural networks. Its unique property of operating on "batches" instead of individual samples introduces significantly different behaviors from most other operations in deep learning. As a result, it leads to many hidden caveats that can negatively impact model's performance in subtle ways. This paper thoroughly reviews such problems in visual recognition tasks, and shows that a key to address them is to rethink different choices in the concept of "batch" in BatchNorm. By presenting these caveats and their mitigations, we hope this review can help researchers use BatchNorm more effectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking “Batch” in BatchNorm<br>pdf: <a href="https://t.co/ZfLGqlGxPv">https://t.co/ZfLGqlGxPv</a><br>abs: <a href="https://t.co/oJArBeNN90">https://t.co/oJArBeNN90</a> <a href="https://t.co/TgqI1HkQv7">pic.twitter.com/TgqI1HkQv7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394520385941643264?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. Pay Attention to MLPs

Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le

- retweets: 6170, favorites: 349 (05/19/2021 09:44:43)

- links: [abs](https://arxiv.org/abs/2105.08050) | [pdf](https://arxiv.org/pdf/2105.08050)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Transformers have become one of the most important architectural innovations in deep learning and have enabled many breakthroughs over the past few years. Here we propose a simple attention-free network architecture, gMLP, based solely on MLPs with gating, and show that it can perform as well as Transformers in key language and vision applications. Our comparisons show that self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy. For BERT, our model achieves parity with Transformers on pretraining perplexity and is better on some downstream tasks. On finetuning tasks where gMLP performs worse, making the gMLP model substantially larger can close the gap with Transformers. In general, our experiments show that gMLP can scale as well as Transformers over increased data and compute.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">2017: Attention is all you need<br><br>2021: You don’t need attention<a href="https://t.co/PYWMcQdccE">https://t.co/PYWMcQdccE</a> <a href="https://t.co/KtAyQ3ICvK">pic.twitter.com/KtAyQ3ICvK</a></p>&mdash; Mark O. Riedl (@mark_riedl) <a href="https://twitter.com/mark_riedl/status/1394745653180477441?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="ja" dir="ltr">我々はひょっとしたら「トランスフォーマー時代」の終わりを目の当たりにしているかも。<br><br>Google Brain から新たに発表された多層パーセプトロン (MLP) にゲート機構を組み合わせた「gMLP」、画像認識とBERT的言語モデルにおいてトランスフォーマーに匹敵する性能を叩き出す <a href="https://t.co/bADNOTRV6c">https://t.co/bADNOTRV6c</a> <a href="https://t.co/iclb5nafNE">pic.twitter.com/iclb5nafNE</a></p>&mdash; ステート・オブ・AI ガイド (@stateofai_ja) <a href="https://twitter.com/stateofai_ja/status/1394509528780054530?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Are Convolutional Neural Networks or Transformers more like human  vision?

Shikhar Tuli, Ishita Dasgupta, Erin Grant, Thomas L. Griffiths

- retweets: 5284, favorites: 55 (05/19/2021 09:44:43)

- links: [abs](https://arxiv.org/abs/2105.07197) | [pdf](https://arxiv.org/pdf/2105.07197)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Modern machine learning models for computer vision exceed humans in accuracy on specific visual recognition tasks, notably on datasets like ImageNet. However, high accuracy can be achieved in many ways. The particular decision function found by a machine learning system is determined not only by the data to which the system is exposed, but also the inductive biases of the model, which are typically harder to characterize. In this work, we follow a recent trend of in-depth behavioral analyses of neural network models that go beyond accuracy as an evaluation metric by looking at patterns of errors. Our focus is on comparing a suite of standard Convolutional Neural Networks (CNNs) and a recently-proposed attention-based network, the Vision Transformer (ViT), which relaxes the translation-invariance constraint of CNNs and therefore represents a model with a weaker set of inductive biases. Attention-based networks have previously been shown to achieve higher accuracy than CNNs on vision tasks, and we demonstrate, using new metrics for examining error consistency with more granularity, that their errors are also more consistent with those of humans. These results have implications both for building more human-like vision models, as well as for understanding visual object recognition in humans.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Vision transformers are more biased towards shapes (as humans are) than Convolutional Networks:<a href="https://t.co/iyuyMr1BbL">https://t.co/iyuyMr1BbL</a> <a href="https://t.co/pR0X9TEh3Z">pic.twitter.com/pR0X9TEh3Z</a></p>&mdash; Arthur Douillard (@Ar_Douillard) <a href="https://twitter.com/Ar_Douillard/status/1394570593241079808?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Texture Generation with Neural Cellular Automata

Alexander Mordvintsev, Eyvind Niklasson, Ettore Randazzo

- retweets: 2538, favorites: 244 (05/19/2021 09:44:43)

- links: [abs](https://arxiv.org/abs/2105.07299) | [pdf](https://arxiv.org/pdf/2105.07299)
- [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Neural Cellular Automata (NCA) have shown a remarkable ability to learn the required rules to "grow" images, classify morphologies, segment images, as well as to do general computation such as path-finding. We believe the inductive prior they introduce lends itself to the generation of textures. Textures in the natural world are often generated by variants of locally interacting reaction-diffusion systems. Human-made textures are likewise often generated in a local manner (textile weaving, for instance) or using rules with local dependencies (regular grids or geometric patterns). We demonstrate learning a texture generator from a single template image, with the generation method being embarrassingly parallel, exhibiting quick convergence and high fidelity of output, and requiring only some minimal assumptions around the underlying state manifold. Furthermore, we investigate properties of the learned models that are both useful and interesting, such as non-stationary dynamics and an inherent robustness to damage. Finally, we make qualitative claims that the behaviour exhibited by the NCA model is a learned, distributed, local algorithm to generate a texture, setting our method apart from existing work on texture generation. We discuss the advantages of such a paradigm.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Texture Generation with Neural Cellular Automata<br>pdf: <a href="https://t.co/3YmeRwvyTa">https://t.co/3YmeRwvyTa</a><br>abs: <a href="https://t.co/ISLLwCdY2Q">https://t.co/ISLLwCdY2Q</a><br>project page: <a href="https://t.co/BEeNstngWS">https://t.co/BEeNstngWS</a> <a href="https://t.co/GbHsS7yr2N">pic.twitter.com/GbHsS7yr2N</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394479451841433600?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Divide and Contrast: Self-supervised Learning from Uncurated Data

Yonglong Tian, Olivier J. Henaff, Aaron van den Oord

- retweets: 1437, favorites: 277 (05/19/2021 09:44:43)

- links: [abs](https://arxiv.org/abs/2105.08054) | [pdf](https://arxiv.org/pdf/2105.08054)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Self-supervised learning holds promise in leveraging large amounts of unlabeled data, however much of its progress has thus far been limited to highly curated pre-training data such as ImageNet. We explore the effects of contrastive learning from larger, less-curated image datasets such as YFCC, and find there is indeed a large difference in the resulting representation quality. We hypothesize that this curation gap is due to a shift in the distribution of image classes -- which is more diverse and heavy-tailed -- resulting in less relevant negative samples to learn from. We test this hypothesis with a new approach, Divide and Contrast (DnC), which alternates between contrastive learning and clustering-based hard negative mining. When pretrained on less curated datasets, DnC greatly improves the performance of self-supervised learning on downstream tasks, while remaining competitive with the current state-of-the-art on curated datasets.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">What makes self-supervised learning from uncurated data so challenging? We identified the heavy-tailed distribution of image content as a limiting factor, and address it with Divide and Contrast. Leads to big gains in data-efficient and transfer learning. <a href="https://t.co/Pf6gClADME">https://t.co/Pf6gClADME</a> <a href="https://t.co/ULiRS41r25">pic.twitter.com/ULiRS41r25</a></p>&mdash; olivierhenaff (@olivierhenaff) <a href="https://twitter.com/olivierhenaff/status/1394681222501216259?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Divide and Contrast: Self-supervised Learning from Uncurated Data<br>pdf: <a href="https://t.co/Qf5gBiCUP6">https://t.co/Qf5gBiCUP6</a><br>abs: <a href="https://t.co/Be6bvqiIWM">https://t.co/Be6bvqiIWM</a><br><br>pretrained on less curated datasets, DnC greatly improves<br>the performance of self-supervised learning on downstream tasks <a href="https://t.co/FhXOy65oMP">pic.twitter.com/FhXOy65oMP</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394471875175231497?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How contrastive learning works on large-scale uncurated data?<br><br>Want to significantly improve it in such a scenario?<br><br>Check the new &quot;Divide and Contrast (DnC)&quot; work jointly with <a href="https://twitter.com/avdnoord?ref_src=twsrc%5Etfw">@avdnoord</a> and <a href="https://twitter.com/olivierhenaff?ref_src=twsrc%5Etfw">@olivierhenaff</a><br><br>ArXiv: <a href="https://t.co/wrq1MjvgWN">https://t.co/wrq1MjvgWN</a> <a href="https://t.co/Yz7lYlv5nA">pic.twitter.com/Yz7lYlv5nA</a></p>&mdash; Yonglong Tian (@YonglongT) <a href="https://twitter.com/YonglongT/status/1394678078039740417?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. A Light Stage on Every Desk

Soumyadip Sengupta, Brian Curless, Ira Kemelmacher-Shlizerman, Steve Seitz

- retweets: 756, favorites: 156 (05/19/2021 09:44:44)

- links: [abs](https://arxiv.org/abs/2105.08051) | [pdf](https://arxiv.org/pdf/2105.08051)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

Every time you sit in front of a TV or monitor, your face is actively illuminated by time-varying patterns of light. This paper proposes to use this time-varying illumination for synthetic relighting of your face with any new illumination condition. In doing so, we take inspiration from the light stage work of Debevec et al., who first demonstrated the ability to relight people captured in a controlled lighting environment. Whereas existing light stages require expensive, room-scale spherical capture gantries and exist in only a few labs in the world, we demonstrate how to acquire useful data from a normal TV or desktop monitor. Instead of subjecting the user to uncomfortable rapidly flashing light patterns, we operate on images of the user watching a YouTube video or other standard content. We train a deep network on images plus monitor patterns of a given user and learn to predict images of that user under any target illumination (monitor pattern). Experimental evaluation shows that our method produces realistic relighting results. Video results are available at http://grail.cs.washington.edu/projects/Light_Stage_on_Every_Desk/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">A Light Stage on Every Desk<br>pdf: <a href="https://t.co/bxb8Olr9W3">https://t.co/bxb8Olr9W3</a><br>abs: <a href="https://t.co/aTAMQP1Xgb">https://t.co/aTAMQP1Xgb</a><br>project page: <a href="https://t.co/nt7QVBZSpo">https://t.co/nt7QVBZSpo</a><br><br>train a deep network on images plus monitor patterns of a given user and learn to predict images of that user under any target illumination <a href="https://t.co/Cf8gqJgt5Y">pic.twitter.com/Cf8gqJgt5Y</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394491427242000385?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Rethinking the Design Principles of Robust Vision Transformer

Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Shaokai Ye, Yuan He, Hui Xue

- retweets: 480, favorites: 91 (05/19/2021 09:44:44)

- links: [abs](https://arxiv.org/abs/2105.07926) | [pdf](https://arxiv.org/pdf/2105.07926)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Recent advances on Vision Transformers (ViT) have shown that self-attention-based networks, which take advantage of long-range dependencies modeling ability, surpassed traditional convolution neural networks (CNNs) in most vision tasks. To further expand the applicability for computer vision, many improved variants are proposed to re-design the Transformer architecture by considering the superiority of CNNs, i.e., locality, translation invariance, for better performance. However, these methods only consider the standard accuracy or computation cost of the model. In this paper, we rethink the design principles of ViTs based on the robustness. We found some design components greatly harm the robustness and generalization ability of ViTs while some others are beneficial. By combining the robust design components, we propose Robust Vision Transformer (RVT). RVT is a new vision transformer, which has superior performance and strong robustness. We further propose two new plug-and-play techniques called position-aware attention rescaling and patch-wise augmentation to train our RVT. The experimental results on ImageNet and six robustness benchmarks show the advanced robustness and generalization ability of RVT compared with previous Transformers and state-of-the-art CNNs. Our RVT-S* also achieves Top-1 rank on multiple robustness leaderboards including ImageNet-C and ImageNet-Sketch. The code will be available at https://github.com/vtddggg/Robust-Vision-Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Rethinking the Design Principles of Robust Vision Transformer<br>pdf: <a href="https://t.co/kfsyLqgyX2">https://t.co/kfsyLqgyX2</a><br>abs: <a href="https://t.co/LVQc3OWCIU">https://t.co/LVQc3OWCIU</a><br>github: <a href="https://t.co/N1vkO7z7xy">https://t.co/N1vkO7z7xy</a><br><br>experimental results on ImageNet and six robustness benchmarks show the robustness and generalization ability of RVT <a href="https://t.co/DNDb15oDRx">pic.twitter.com/DNDb15oDRx</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394462608057634817?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. NeuroGen: activation optimized image synthesis for discovery  neuroscience

Zijin Gu, Keith W. Jamison, Meenakshi Khosla, Emily J. Allen, Yihan Wu, Thomas Naselaris, Kendrick Kay, Mert R. Sabuncu, Amy Kuceyeski

- retweets: 214, favorites: 78 (05/19/2021 09:44:44)

- links: [abs](https://arxiv.org/abs/2105.07140) | [pdf](https://arxiv.org/pdf/2105.07140)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent) | [q-bio.QM](https://arxiv.org/list/q-bio.QM/recent)

Functional MRI (fMRI) is a powerful technique that has allowed us to characterize visual cortex responses to stimuli, yet such experiments are by nature constructed based on a priori hypotheses, limited to the set of images presented to the individual while they are in the scanner, are subject to noise in the observed brain responses, and may vary widely across individuals. In this work, we propose a novel computational strategy, which we call NeuroGen, to overcome these limitations and develop a powerful tool for human vision neuroscience discovery. NeuroGen combines an fMRI-trained neural encoding model of human vision with a deep generative network to synthesize images predicted to achieve a target pattern of macro-scale brain activation. We demonstrate that the reduction of noise that the encoding model provides, coupled with the generative network's ability to produce images of high fidelity, results in a robust discovery architecture for visual neuroscience. By using only a small number of synthetic images created by NeuroGen, we demonstrate that we can detect and amplify differences in regional and individual human brain response patterns to visual stimuli. We then verify that these discoveries are reflected in the several thousand observed image responses measured with fMRI. We further demonstrate that NeuroGen can create synthetic images predicted to achieve regional response patterns not achievable by the best-matching natural images. The NeuroGen framework extends the utility of brain encoding models and opens up a new avenue for exploring, and possibly precisely controlling, the human visual system.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">I am proud to announce NeuroGen, a novel framework for generating synthetic images predicted to achieve preselected brain activation patterns. <a href="https://t.co/MClIURQ3l0">https://t.co/MClIURQ3l0</a> <br>from an amazing collaboration: <a href="https://twitter.com/zijin_gu?ref_src=twsrc%5Etfw">@zijin_gu</a> <a href="https://twitter.com/mertrory?ref_src=twsrc%5Etfw">@mertrory</a> <a href="https://twitter.com/meenakshik93?ref_src=twsrc%5Etfw">@meenakshik93</a> 1/5 <a href="https://t.co/AsItmGO0fc">pic.twitter.com/AsItmGO0fc</a></p>&mdash; Amy Kuceyeski (@amykooz) <a href="https://twitter.com/amykooz/status/1394446577461248003?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. NeLF: Practical Novel View Synthesis with Neural Light Field

Celong Liu, Zhong Li, Junsong Yuan, Yi Xu

- retweets: 110, favorites: 63 (05/19/2021 09:44:44)

- links: [abs](https://arxiv.org/abs/2105.07112) | [pdf](https://arxiv.org/pdf/2105.07112)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent)

In this paper, we present a practical and robust deep learning solution for the novel view synthesis of complex scenes. In our approach, a continuous scene is represented as a light field, i.e., a set of rays, each of which has a corresponding color. We adopt a 4D parameterization of the light field. We then formulate the light field as a 4D function that maps 4D coordinates to corresponding color values. We train a deep fully connected network to optimize this function. Then, the scene-specific model is used to synthesize novel views. Previous light field approaches usually require dense view sampling to reliably render high-quality novel views. Our method can render novel views by sampling rays and querying the color for each ray from the network directly; thus enabling fast light field rendering with a very sparse set of input images. Our method achieves state-of-the-art novel view synthesis results while maintaining an interactive frame rate.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">NeLF: Practical Novel View Synthesis with Neural Light Field<br>pdf: <a href="https://t.co/5smg0KnT2k">https://t.co/5smg0KnT2k</a><br>abs: <a href="https://t.co/3qG5hd8GKz">https://t.co/3qG5hd8GKz</a><br><br>represent a continuous scene using a 4D light field and train an MLP network to learn this mapping from input posed images <a href="https://t.co/BQHQpkU0Wi">pic.twitter.com/BQHQpkU0Wi</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394477383726604288?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Deep learning for detecting pulmonary tuberculosis via chest  radiography: an international study across 10 countries

Sahar Kazemzadeh, Jin Yu, Shahar Jamshy, Rory Pilgrim, Zaid Nabulsi, Christina Chen, Neeral Beladia, Charles Lau, Scott Mayer McKinney, Thad Hughes, Atilla Kiraly, Sreenivasa Raju Kalidindi, Monde Muyoyeta, Jameson Malemela, Ting Shih, Greg S. Corrado, Lily Peng, Katherine Chou, Po-Hsuan Cameron Chen, Yun Liu, Krish Eswaran, Daniel Tse, Shravya Shetty, Shruthi Prabhakara

- retweets: 113, favorites: 30 (05/19/2021 09:44:44)

- links: [abs](https://arxiv.org/abs/2105.07540) | [pdf](https://arxiv.org/pdf/2105.07540)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Tuberculosis (TB) is a top-10 cause of death worldwide. Though the WHO recommends chest radiographs (CXRs) for TB screening, the limited availability of CXR interpretation is a barrier. We trained a deep learning system (DLS) to detect active pulmonary TB using CXRs from 9 countries across Africa, Asia, and Europe, and utilized large-scale CXR pretraining, attention pooling, and noisy student semi-supervised learning. Evaluation was on (1) a combined test set spanning China, India, US, and Zambia, and (2) an independent mining population in South Africa. Given WHO targets of 90% sensitivity and 70% specificity, the DLS's operating point was prespecified to favor sensitivity over specificity. On the combined test set, the DLS's ROC curve was above all 9 India-based radiologists, with an AUC of 0.90 (95%CI 0.87-0.92). The DLS's sensitivity (88%) was higher than the India-based radiologists (75% mean sensitivity), p<0.001 for superiority; and its specificity (79%) was non-inferior to the radiologists (84% mean specificity), p=0.004. Similar trends were observed within HIV positive and sputum smear positive sub-groups, and in the South Africa test set. We found that 5 US-based radiologists (where TB isn't endemic) were more sensitive and less specific than the India-based radiologists (where TB is endemic). The DLS also remained non-inferior to the US-based radiologists. In simulations, using the DLS as a prioritization tool for confirmatory testing reduced the cost per positive case detected by 40-80% compared to using confirmatory testing alone. To conclude, our DLS generalized to 5 countries, and merits prospective evaluation to assist cost-effective screening efforts in radiologist-limited settings. Operating point flexibility may permit customization of the DLS to account for site-specific factors such as TB prevalence, demographics, clinical resources, and customary practice patterns.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Deep learning for detecting pulmonary tuberculosis via chest radiography: an international study across 10 countries<br>pdf: <a href="https://t.co/lspC4Xg61L">https://t.co/lspC4Xg61L</a><br>abs: <a href="https://t.co/V4P3X9G7Ed">https://t.co/V4P3X9G7Ed</a> <a href="https://t.co/cCHjkuOAE7">pic.twitter.com/cCHjkuOAE7</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394527087093063680?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. ExSinGAN: Learning an Explainable Generative Model from a Single Image

ZiCheng Zhang, CongYing Han, TianDe Guo

- retweets: 64, favorites: 54 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07350) | [pdf](https://arxiv.org/pdf/2105.07350)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Generating images from a single sample, as a newly developing branch of image synthesis, has attracted extensive attention. In this paper, we formulate this problem as sampling from the conditional distribution of a single image, and propose a hierarchical framework that simplifies the learning of the intricate conditional distributions through the successive learning of the distributions about structure, semantics and texture, making the process of learning and generation comprehensible. On this basis, we design ExSinGAN composed of three cascaded GANs for learning an explainable generative model from a given image, where the cascaded GANs model the distributions about structure, semantics and texture successively. ExSinGAN is learned not only from the internal patches of the given image as the previous works did, but also from the external prior obtained by the GAN inversion technique. Benefiting from the appropriate combination of internal and external information, ExSinGAN has a more powerful capability of generation and competitive generalization ability for the image manipulation tasks compared with prior works.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ExSinGAN: Learning an Explainable Generative Model from a Single Image<br>pdf: <a href="https://t.co/R4D6hGlZyT">https://t.co/R4D6hGlZyT</a><br>abs: <a href="https://t.co/RCXgaD0Jw2">https://t.co/RCXgaD0Jw2</a><br><br>composed of 3 cascaded GANs for learning an explainable generative model from a given image <a href="https://t.co/yqsInLUKtw">pic.twitter.com/yqsInLUKtw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394513983718178819?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Vision Transformers are Robust Learners

Sayak Paul, Pin-Yu Chen

- retweets: 73, favorites: 43 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07581) | [pdf](https://arxiv.org/pdf/2105.07581)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Transformers, composed of multiple self-attention layers, hold strong promises toward a generic learning primitive applicable to different data modalities, including the recent breakthroughs in computer vision achieving state-of-the-art (SOTA) standard accuracy with better parameter efficiency. Since self-attention helps a model systematically align different components present inside the input data, it leaves grounds to investigate its performance under model robustness benchmarks. In this work, we study the robustness of the Vision Transformer (ViT) against common corruptions and perturbations, distribution shifts, and natural adversarial examples. We use six different diverse ImageNet datasets concerning robust classification to conduct a comprehensive performance comparison of ViT models and SOTA convolutional neural networks (CNNs), Big-Transfer. Through a series of six systematically designed experiments, we then present analyses that provide both quantitative and qualitative indications to explain why ViTs are indeed more robust learners. For example, with fewer parameters and similar dataset and pre-training combinations, ViT gives a top-1 accuracy of 28.10% on ImageNet-A which is 4.3x higher than a comparable variant of BiT. Our analyses on image masking, Fourier spectrum sensitivity, and spread on discrete cosine energy spectrum reveal intriguing properties of ViT attributing to improved robustness. Code for reproducing our experiments is available here: https://git.io/J3VO0.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work w/ <a href="https://twitter.com/pinyuchenTW?ref_src=twsrc%5Etfw">@pinyuchenTW</a> - &quot;Vision Transformers are Robust Learners&quot;<br><br>With self-attention and other design choices, can vision transformers provide improved robustness to common corruptions, perturbations, etc.?<br><br>Paper: <a href="https://t.co/jn4KvPFBqC">https://t.co/jn4KvPFBqC</a><br>Code: <a href="https://t.co/4qxPZvl24L">https://t.co/4qxPZvl24L</a><br><br>1/n</p>&mdash; Sayak Paul (@RisingSayak) <a href="https://twitter.com/RisingSayak/status/1394470774086328324?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Protein sequence-to-structure learning: Is this the end(-to-end  revolution)?

Elodie Laine, Stephan Eismann, Arne Elofsson, Sergei Grudinin

- retweets: 81, favorites: 27 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07407) | [pdf](https://arxiv.org/pdf/2105.07407)
- [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

The potential of deep learning has been recognized in the protein structure prediction community for some time, and became indisputable after CASP13. In CASP14, deep learning has boosted the field to unanticipated levels reaching near-experimental accuracy. This success comes from advances transferred from other machine learning areas, as well as methods specifically designed to deal with protein sequences and structures, and their abstractions. Novel emerging approaches include (i) geometric learning, i.e. learning on representations such as graphs, 3D Voronoi tessellations, and point clouds; (ii) pre-trained protein language models leveraging attention; (iii) equivariant architectures preserving the symmetry of 3D space; (iv) use of large meta-genome databases; (v) combinations of protein representations; (vi) and finally truly end-to-end architectures, i.e. differentiable models starting from a sequence and returning a 3D structure. Here, we provide an overview and our opinion of the novel deep learning approaches developed in the last two years and widely used in CASP14.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[2105.07407] Protein sequence-to-structure learning: Is this the end(-to-end revolution)? <a href="https://t.co/hNnHCxw5Ap">https://t.co/hNnHCxw5Ap</a></p>&mdash; Arne Elofsson (@arneelof) <a href="https://twitter.com/arneelof/status/1394517235444633600?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. Learning a Universal Template for Few-shot Dataset Generalization

Eleni Triantafillou, Hugo Larochelle, Richard Zemel, Vincent Dumoulin

- retweets: 57, favorites: 36 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07029) | [pdf](https://arxiv.org/pdf/2105.07029)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

Few-shot dataset generalization is a challenging variant of the well-studied few-shot classification problem where a diverse training set of several datasets is given, for the purpose of training an adaptable model that can then learn classes from new datasets using only a few examples. To this end, we propose to utilize the diverse training set to construct a universal template: a partial model that can define a wide array of dataset-specialized models, by plugging in appropriate components. For each new few-shot classification problem, our approach therefore only requires inferring a small number of parameters to insert into the universal template. We design a separate network that produces an initialization of those parameters for each given task, and we then fine-tune its proposed initialization via a few steps of gradient descent. Our approach is more parameter-efficient, scalable and adaptable compared to previous methods, and achieves the state-of-the-art on the challenging Meta-Dataset benchmark.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Learning a Universal Template for Few-shot Dataset Generalization<br>pdf: <a href="https://t.co/8QlVdsWJcV">https://t.co/8QlVdsWJcV</a><br>abs: <a href="https://t.co/U4W50fMTPL">https://t.co/U4W50fMTPL</a><br><br>SOTA on Meta-Dataset, significantly outperforming previous methods on few-shot dataset generalization <a href="https://t.co/ctlmLQm6z1">pic.twitter.com/ctlmLQm6z1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394475938927325187?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 15. It$\hat{\text{o}}$TTS and It$\hat{\text{o}}$Wave: Linear Stochastic  Differential Equation Is All You Need For Audio Generation

Shoule Wu, Ziqiang Shi

- retweets: 44, favorites: 45 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07583) | [pdf](https://arxiv.org/pdf/2105.07583)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

In this paper, we propose to unify the two aspects of voice synthesis, namely text-to-speech (TTS) and vocoder, into one framework based on a pair of forward and reverse-time linear stochastic differential equations (SDE). The solutions of this SDE pair are two stochastic processes, one of which turns the distribution of mel spectrogram (or wave), that we want to generate, into a simple and tractable distribution. The other is the generation procedure that turns this tractable simple signal into the target mel spectrogram (or wave). The model that generates mel spectrogram is called It$\hat{\text{o}}$TTS, and the model that generates wave is called It$\hat{\text{o}}$Wave. It$\hat{\text{o}}$TTS and It$\hat{\text{o}}$Wave use the Wiener process as a driver to gradually subtract the excess signal from the noise signal to generate realistic corresponding meaningful mel spectrogram and audio respectively, under the conditional inputs of original text or mel spectrogram. The results of the experiment show that the mean opinion scores (MOS) of It$\hat{\text{o}}$TTS and It$\hat{\text{o}}$Wave can exceed the current state-of-the-art methods, reached 3.925$\pm$ 0.160 and 4.35$\pm$ 0.115 respectively.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ItôTTS and ItôWave: Linear Stochastic Differential Equation Is All You Need For Audio Generation<br>pdf: <a href="https://t.co/ErDgTruTsT">https://t.co/ErDgTruTsT</a><br>abs: <a href="https://t.co/dQzIueu0QN">https://t.co/dQzIueu0QN</a><br><br>general methods based on linear SDE that can accomplish TTS and vocoder tasks at the same time <a href="https://t.co/75GlRsLfax">pic.twitter.com/75GlRsLfax</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394494483564310528?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Follow the Money: Analyzing @slpng_giants_pt's Strategy to Combat  Misinformation

Bárbara Gomes Ribeiro, Manoel Horta Ribeiro, Virgílio Almeida, Wagner Meira Jr

- retweets: 49, favorites: 34 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07523) | [pdf](https://arxiv.org/pdf/2105.07523)
- [cs.CY](https://arxiv.org/list/cs.CY/recent)

In 2020, the activist movement @sleeping_giants_pt (SGB) made a splash in Brazil. Similar to its international counterparts, the movement carried "campaigns" against media outlets spreading misinformation. In those, SGB targeted companies whose ads were shown in these outlets, publicly asking them to remove the ads. In this work, we present a careful characterization of SGB's activism model, analyzing the three campaigns carried by the movement up to September 2020. We study how successful its complaints were and what factors are associated with their success, how attention towards the targeted media outlets progressed, and how online interactions with the companies were impacted after they were targeted. Leveraging an annotated corpus of SGB's tweets as well as other data from Twitter and Google Search, we show that SGB's "campaigns" were largely successful: over 86\% of companies (n=161) responded positively to SGB's requests, and, for those that responded, we find user pressure to be negatively correlated with the time companies take to answer ($r$=-0.67; $p$<0.001). Finally, we find that, although changes in the interactions with companies were transient, the impact in targeted media outlets endured: all three outlets experienced a significant decrease in engagement on Twitter and search volume on Google following the start of SGB's campaigns. Overall, our work suggests that internet-based activism can leverage the transient attention it captures towards concrete goals to have a long-lasting impact.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">In our (w <a href="https://twitter.com/BarbaraGRibeiro?ref_src=twsrc%5Etfw">@BarbaraGRibeiro</a>, <a href="https://twitter.com/virgilioalmeida?ref_src=twsrc%5Etfw">@virgilioalmeida</a>, and <a href="https://twitter.com/wagnermeirajr?ref_src=twsrc%5Etfw">@wagnermeirajr</a>) newest pre-print, we analyze the <a href="https://twitter.com/slpng_giants_pt?ref_src=twsrc%5Etfw">@slpng_giants_pt</a> activist movement. We study how the movement impacted the media outlets it targeted, and the companies it interacted with!<br><br>📜<a href="https://t.co/EoeoGJYUu5">https://t.co/EoeoGJYUu5</a> <a href="https://t.co/OASQuvmfGW">pic.twitter.com/OASQuvmfGW</a></p>&mdash; Manoel (@manoelribeiro) <a href="https://twitter.com/manoelribeiro/status/1394626004719706112?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 17. The Online Pivot: Lessons Learned from Teaching a Text and Data Mining  Course in Lockdown, Enhancing online Teaching with Pair Programming and  Digital Badges

Beatrice Alex, Clare Llewellyn, Pawel Michal Orzechowski, Maria Boutchkova

- retweets: 42, favorites: 27 (05/19/2021 09:44:45)

- links: [abs](https://arxiv.org/abs/2105.07847) | [pdf](https://arxiv.org/pdf/2105.07847)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

In this paper we provide an account of how we ported a text and data mining course online in summer 2020 as a result of the COVID-19 pandemic and how we improved it in a second pilot run. We describe the course, how we adapted it over the two pilot runs and what teaching techniques we used to improve students' learning and community building online. We also provide information on the relentless feedback collected during the course which helped us to adapt our teaching from one session to the next and one pilot to the next. We discuss the lessons learned and promote the use of innovative teaching techniques applied to the digital such as digital badges and pair programming in break-out rooms for teaching Natural Language Processing courses to beginners and students with different backgrounds.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Our paper The Online Pivot: Lessons Learned from Teaching a Text and Data Mining<br> Course in Lockdown now up <a href="https://t.co/npEuH6vz23">https://t.co/npEuH6vz23</a>.  Presented at the Teaching NLP workshop at <a href="https://twitter.com/naacl?ref_src=twsrc%5Etfw">@naacl</a> in June. <a href="https://twitter.com/clarellewellyn?ref_src=twsrc%5Etfw">@clarellewellyn</a> <a href="https://twitter.com/DrPawel?ref_src=twsrc%5Etfw">@DrPawel</a> <a href="https://twitter.com/Boutchkova?ref_src=twsrc%5Etfw">@Boutchkova</a></p>&mdash; Bea Alex (@bea_alex) <a href="https://twitter.com/bea_alex/status/1394572228235255809?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 18. SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image  Warping

Austin Stone, Daniel Maurer, Alper Ayvaci, Anelia Angelova, Rico Jonschkowski

- retweets: 30, favorites: 37 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07014) | [pdf](https://arxiv.org/pdf/2105.07014)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We present SMURF, a method for unsupervised learning of optical flow that improves state of the art on all benchmarks by $36\%$ to $40\%$ (over the prior best method UFlow) and even outperforms several supervised approaches such as PWC-Net and FlowNet2. Our method integrates architecture improvements from supervised optical flow, i.e. the RAFT model, with new ideas for unsupervised learning that include a sequence-aware self-supervision loss, a technique for handling out-of-frame motion, and an approach for learning effectively from multi-frame video data while still only requiring two frames for inference.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping<br>pdf: <a href="https://t.co/xNc86d8nM9">https://t.co/xNc86d8nM9</a><br>abs: <a href="https://t.co/zmh4DDnxZN">https://t.co/zmh4DDnxZN</a><br>github: <a href="https://t.co/rtsIQssokF">https://t.co/rtsIQssokF</a><br><br>unsupervised learning of optical flow that improves sota on all benchmarks by 36% to 40% <a href="https://t.co/U4cioNmeEL">pic.twitter.com/U4cioNmeEL</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394473465772707840?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 19. Differentiable SLAM-net: Learning Particle SLAM for Visual Navigation

Peter Karkus, Shaojun Cai, David Hsu

- retweets: 41, favorites: 23 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07593) | [pdf](https://arxiv.org/pdf/2105.07593)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

Simultaneous localization and mapping (SLAM) remains challenging for a number of downstream applications, such as visual robot navigation, because of rapid turns, featureless walls, and poor camera quality. We introduce the Differentiable SLAM Network (SLAM-net) along with a navigation architecture to enable planar robot navigation in previously unseen indoor environments. SLAM-net encodes a particle filter based SLAM algorithm in a differentiable computation graph, and learns task-oriented neural network components by backpropagating through the SLAM algorithm. Because it can optimize all model components jointly for the end-objective, SLAM-net learns to be robust in challenging conditions. We run experiments in the Habitat platform with different real-world RGB and RGB-D datasets. SLAM-net significantly outperforms the widely adapted ORB-SLAM in noisy conditions. Our navigation architecture with SLAM-net improves the state-of-the-art for the Habitat Challenge 2020 PointNav task by a large margin (37% to 64% success). Project website: http://sites.google.com/view/slamnet

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Differentiable SLAM-net: Learning Particle SLAM for Visual Navigation <a href="https://t.co/2tRPJjjeel">https://t.co/2tRPJjjeel</a></p>&mdash; arXiv CS-CV (@arxiv_cscv) <a href="https://twitter.com/arxiv_cscv/status/1394543437232234499?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 20. How Deep is your Learning: the DL-HARD Annotated Deep Learning Dataset

Iain Mackie, Jeffery Dalton, Andrew Yates

- retweets: 42, favorites: 21 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07975) | [pdf](https://arxiv.org/pdf/2105.07975)
- [cs.IR](https://arxiv.org/list/cs.IR/recent)

Deep Learning Hard (DL-HARD) is a new annotated dataset designed to more effectively evaluate neural ranking models on complex topics. It builds on TREC Deep Learning (DL) topics by extensively annotating them with question intent categories, answer types, wikified entities, topic categories, and result type metadata from a commercial web search engine. Based on this data, we introduce a framework for identifying challenging queries. DL-HARD contains fifty topics from the official DL 2019/2020 evaluation benchmark, half of which are newly and independently assessed. We perform experiments using the official submitted runs to DL on DL-HARD and find substantial differences in metrics and the ranking of participating systems. Overall, DL-HARD is a new resource that promotes research on neural ranking methods by focusing on challenging and complex topics.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DL-HARD <a href="https://twitter.com/SIGIRConf?ref_src=twsrc%5Etfw">@SIGIRConf</a> preprint: <a href="https://t.co/uSY9xzkpQ1">https://t.co/uSY9xzkpQ1</a> <br><br>If you are looking for a challenging IR dataset? Why not try: <a href="https://t.co/83n7hTYPD5">https://t.co/83n7hTYPD5</a>. Includes annotations, entity links, and framework for selecting complex queries. <a href="https://twitter.com/JeffD?ref_src=twsrc%5Etfw">@JeffD</a> <a href="https://twitter.com/andrewyates?ref_src=twsrc%5Etfw">@andrewyates</a> <a href="https://twitter.com/ir_glasgow?ref_src=twsrc%5Etfw">@ir_glasgow</a> <a href="https://twitter.com/UofGlasgow?ref_src=twsrc%5Etfw">@UofGlasgow</a> <a href="https://twitter.com/MSMarcoAI?ref_src=twsrc%5Etfw">@MSMarcoAI</a></p>&mdash; Iain Mackie (@iain_with_2is) <a href="https://twitter.com/iain_with_2is/status/1394603900779565056?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 21. Move2Hear: Active Audio-Visual Source Separation

Sagnik Majumder, Ziad Al-Halah, Kristen Grauman

- retweets: 30, favorites: 31 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07142) | [pdf](https://arxiv.org/pdf/2105.07142)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.RO](https://arxiv.org/list/cs.RO/recent) | [cs.SD](https://arxiv.org/list/cs.SD/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

We introduce the active audio-visual source separation problem, where an agent must move intelligently in order to better isolate the sounds coming from an object of interest in its environment. The agent hears multiple audio sources simultaneously (e.g., a person speaking down the hall in a noisy household) and must use its eyes and ears to automatically separate out the sounds originating from the target object within a limited time budget. Towards this goal, we introduce a reinforcement learning approach that trains movement policies controlling the agent's camera and microphone placement over time, guided by the improvement in predicted audio separation quality. We demonstrate our approach in scenarios motivated by both augmented reality (system is already co-located with the target object) and mobile robotics (agent begins arbitrarily far from the target object). Using state-of-the-art realistic audio-visual simulations in 3D environments, we demonstrate our model's ability to find minimal movement sequences with maximal payoff for audio source separation. Project: http://vision.cs.utexas.edu/projects/move2hear.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Move2Hear: Active Audio-Visual Source Separation<br>pdf: <a href="https://t.co/EIiwoku2Y0">https://t.co/EIiwoku2Y0</a><br>abs: <a href="https://t.co/GQnn7CBlNG">https://t.co/GQnn7CBlNG</a><br>project page: <a href="https://t.co/mR9hMVC5LJ">https://t.co/mR9hMVC5LJ</a> <a href="https://t.co/zZ0Pd2CxTY">pic.twitter.com/zZ0Pd2CxTY</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1394509369803526146?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 22. An End-to-End Framework for Molecular Conformation Generation via  Bilevel Programming

Minkai Xu, Wujie Wang, Shitong Luo, Chence Shi, Yoshua Bengio, Rafael Gomez-Bombarelli, Jian Tang

- retweets: 42, favorites: 18 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07246) | [pdf](https://arxiv.org/pdf/2105.07246)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [q-bio.BM](https://arxiv.org/list/q-bio.BM/recent)

Predicting molecular conformations (or 3D structures) from molecular graphs is a fundamental problem in many applications. Most existing approaches are usually divided into two steps by first predicting the distances between atoms and then generating a 3D structure through optimizing a distance geometry problem. However, the distances predicted with such two-stage approaches may not be able to consistently preserve the geometry of local atomic neighborhoods, making the generated structures unsatisfying. In this paper, we propose an end-to-end solution for molecular conformation prediction called ConfVAE based on the conditional variational autoencoder framework. Specifically, the molecular graph is first encoded in a latent space, and then the 3D structures are generated by solving a principled bilevel optimization program. Extensive experiments on several benchmark data sets prove the effectiveness of our proposed approach over existing state-of-the-art approaches.




# 23. Choice Set Confounding in Discrete Choice

Kiran Tomlinson, Johan Ugander, Austin R. Benson

- retweets: 27, favorites: 33 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07959) | [pdf](https://arxiv.org/pdf/2105.07959)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.SI](https://arxiv.org/list/cs.SI/recent) | [econ.EM](https://arxiv.org/list/econ.EM/recent)

Standard methods in preference learning involve estimating the parameters of discrete choice models from data of selections (choices) made by individuals from a discrete set of alternatives (the choice set). While there are many models for individual preferences, existing learning methods overlook how choice set assignment affects the data. Often, the choice set itself is influenced by an individual's preferences; for instance, a consumer choosing a product from an online retailer is often presented with options from a recommender system that depend on information about the consumer's preferences. Ignoring these assignment mechanisms can mislead choice models into making biased estimates of preferences, a phenomenon that we call choice set confounding; we demonstrate the presence of such confounding in widely-used choice datasets.   To address this issue, we adapt methods from causal inference to the discrete choice setting. We use covariates of the chooser for inverse probability weighting and/or regression controls, accurately recovering individual preferences in the presence of choice set confounding under certain assumptions. When such covariates are unavailable or inadequate, we develop methods that take advantage of structured choice set assignment to improve prediction. We demonstrate the effectiveness of our methods on real-world choice data, showing, for example, that accounting for choice set confounding makes choices observed in hotel booking and commute transportation more consistent with rational utility-maximization.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Choices under recommender systems are highly confounded: different people see different sets. Here&#39;s how <a href="https://twitter.com/kiran_tomlinson?ref_src=twsrc%5Etfw">@kiran_tomlinson</a> <a href="https://twitter.com/austinbenson?ref_src=twsrc%5Etfw">@austinbenson</a> and I have come to think about that and related settings, now on arxiv and to appear at KDD! <a href="https://t.co/L2f1K0QP4o">https://t.co/L2f1K0QP4o</a> <a href="https://t.co/FrvbD6djLb">https://t.co/FrvbD6djLb</a></p>&mdash; Johan Ugander (@jugander) <a href="https://twitter.com/jugander/status/1394668783709474820?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 24. Urban Analytics: History, Trajectory, and Critique

Geoff Boeing, Michael Batty, Shan Jiang, Lisa Schweitzer

- retweets: 44, favorites: 13 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07020) | [pdf](https://arxiv.org/pdf/2105.07020)
- [cs.CY](https://arxiv.org/list/cs.CY/recent) | [physics.soc-ph](https://arxiv.org/list/physics.soc-ph/recent)

Urban analytics combines spatial analysis, statistics, computer science, and urban planning to understand and shape city futures. While it promises better policymaking insights, concerns exist around its epistemological scope and impacts on privacy, ethics, and social control. This chapter reflects on the history and trajectory of urban analytics as a scholarly and professional discipline. In particular, it considers the direction in which this field is going and whether it improves our collective and individual welfare. It first introduces early theories, models, and deductive methods from which the field originated before shifting toward induction. It then explores urban network analytics that enrich traditional representations of spatial interaction and structure. Next it discusses urban applications of spatiotemporal big data and machine learning. Finally, it argues that privacy and ethical concerns are too often ignored as ubiquitous monitoring and analytics can empower social repression. It concludes with a call for a more critical urban analytics that recognizes its epistemological limits, emphasizes human dignity, and learns from and supports marginalized communities.




# 25. Bayesian reconstruction of memories stored in neural networks from their  connectivity

Sebastian Goldt, Florent Krzakala, Lenka Zdeborová, Nicolas Brunel

- retweets: 12, favorites: 38 (05/19/2021 09:44:46)

- links: [abs](https://arxiv.org/abs/2105.07416) | [pdf](https://arxiv.org/pdf/2105.07416)
- [q-bio.NC](https://arxiv.org/list/q-bio.NC/recent) | [cond-mat.stat-mech](https://arxiv.org/list/cond-mat.stat-mech/recent) | [stat.ML](https://arxiv.org/list/stat.ML/recent)

The advent of comprehensive synaptic wiring diagrams of large neural circuits has created the field of connectomics and given rise to a number of open research questions. One such question is whether it is possible to reconstruct the information stored in a recurrent network of neurons, given its synaptic connectivity matrix. Here, we address this question by determining when solving such an inference problem is theoretically possible in specific attractor network models and by providing a practical algorithm to do so. The algorithm builds on ideas from statistical physics to perform approximate Bayesian inference and is amenable to exact analysis. We study its performance on three different models and explore the limitations of reconstructing stored patterns from synaptic connectivity.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Did you wonder how many patterns one can retrieve from a Hopfield network if you do not give it a hot start towards any of the patterns? Well not so many .... <a href="https://t.co/l9Hnn6tVYQ">https://t.co/l9Hnn6tVYQ</a> including some wild perspectives for retrieval of what a brain learned from its wiring.</p>&mdash; Lenka Zdeborova (@zdeborova) <a href="https://twitter.com/zdeborova/status/1394662573845491712?ref_src=twsrc%5Etfw">May 18, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>



