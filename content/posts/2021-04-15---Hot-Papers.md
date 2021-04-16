---
title: Hot Papers 2021-04-15
date: 2021-04-16T09:28:02.Z
template: "post"
draft: false
slug: "hot-papers-2021-04-15"
category: "arXiv"
tags:
  - "arXiv"
  - "Twitter"
  - "Machine Learning"
  - "Computer Science"
description: "Hot papers 2021-04-15"
socialImage: "/media/flying-marine.jpg"

---

# 1. Masked Language Modeling and the Distributional Hypothesis: Order Word  Matters Pre-training for Little

Koustuv Sinha, Robin Jia, Dieuwke Hupkes, Joelle Pineau, Adina Williams, Douwe Kiela

- retweets: 6082, favorites: 321 (04/16/2021 09:28:02)

- links: [abs](https://arxiv.org/abs/2104.06644) | [pdf](https://arxiv.org/pdf/2104.06644)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

A possible explanation for the impressive performance of masked language model (MLM) pre-training is that such models have learned to represent the syntactic structures prevalent in classical NLP pipelines. In this paper, we propose a different explanation: MLMs succeed on downstream tasks almost entirely due to their ability to model higher-order word co-occurrence statistics. To demonstrate this, we pre-train MLMs on sentences with randomly shuffled word order, and show that these models still achieve high accuracy after fine-tuning on many downstream tasks -- including on tasks specifically designed to be challenging for models that ignore word order. Our models perform surprisingly well according to some parametric syntactic probes, indicating possible deficiencies in how we test representations for syntactic information. Overall, our results show that purely distributional information largely explains the success of pre-training, and underscore the importance of curating challenging evaluation datasets that require deeper linguistic knowledge.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">[1/7] Excited to announce “Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little”.  BERT gets high task scores due to its distributional prior rather than its ability to “discover the NLP pipeline”.  <a href="https://t.co/2s0gq8vGFz">https://t.co/2s0gq8vGFz</a> <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> <a href="https://t.co/yF3rCFgCc9">pic.twitter.com/yF3rCFgCc9</a></p>&mdash; Koustuv Sinha (@koustuvsinha) <a href="https://twitter.com/koustuvsinha/status/1382671525623558145?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 2. DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort

Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler

- retweets: 4160, favorites: 256 (04/16/2021 09:28:03)

- links: [abs](https://arxiv.org/abs/2104.06490) | [pdf](https://arxiv.org/pdf/2104.06490)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

We introduce DatasetGAN: an automatic procedure to generate massive datasets of high-quality semantically segmented images requiring minimal human effort. Current deep networks are extremely data-hungry, benefiting from training on large-scale datasets, which are time consuming to annotate. Our method relies on the power of recent GANs to generate realistic images. We show how the GAN latent code can be decoded to produce a semantic segmentation of the image. Training the decoder only needs a few labeled examples to generalize to the rest of the latent space, resulting in an infinite annotated dataset generator! These generated datasets can then be used for training any computer vision architecture just as real datasets are. As only a few images need to be manually segmented, it becomes possible to annotate images in extreme detail and generate datasets with rich object and part segmentations. To showcase the power of our approach, we generated datasets for 7 image segmentation tasks which include pixel-level labels for 34 human face parts, and 32 car parts. Our approach outperforms all semi-supervised baselines significantly and is on par with fully supervised methods, which in some cases require as much as 100x more annotated data as our method.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort<br>pdf: <a href="https://t.co/uDMbrDU7V8">https://t.co/uDMbrDU7V8</a><br>abs: <a href="https://t.co/obT7dz6GqO">https://t.co/obT7dz6GqO</a> <a href="https://t.co/gQmmMimM6Z">pic.twitter.com/gQmmMimM6Z</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382501497188061188?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 3. Aligning Latent and Image Spaces to Connect the Unconnectable

Ivan Skorokhodov, Grigorii Sotnikov, Mohamed Elhoseiny

- retweets: 1024, favorites: 168 (04/16/2021 09:28:03)

- links: [abs](https://arxiv.org/abs/2104.06954) | [pdf](https://arxiv.org/pdf/2104.06954)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent)

In this work, we develop a method to generate infinite high-resolution images with diverse and complex content. It is based on a perfectly equivariant generator with synchronous interpolations in the image and latent spaces. Latent codes, when sampled, are positioned on the coordinate grid, and each pixel is computed from an interpolation of the nearby style codes. We modify the AdaIN mechanism to work in such a setup and train the generator in an adversarial setting to produce images positioned between any two latent vectors. At test time, this allows for generating complex and diverse infinite images and connecting any two unrelated scenes into a single arbitrarily large panorama. Apart from that, we introduce LHQ: a new dataset of \lhqsize high-resolution nature landscapes. We test the approach on LHQ, LSUN Tower and LSUN Bridge and outperform the baselines by at least 4 times in terms of quality and diversity of the produced infinite images. The project page is located at https://universome.github.io/alis.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Aligning Latent and Image Spaces to Connect the Unconnectable<br>pdf: <a href="https://t.co/2KB14lzCFq">https://t.co/2KB14lzCFq</a><br>abs: <a href="https://t.co/QP9g0kg0fw">https://t.co/QP9g0kg0fw</a><br>project page: <a href="https://t.co/7HyH8Ew8pS">https://t.co/7HyH8Ew8pS</a> <a href="https://t.co/sOPnN73ImW">pic.twitter.com/sOPnN73ImW</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382546630407241732?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 4. Modeling Framing in Immigration Discourse on Social Media

Julia Mendelsohn, Ceren Budak, David Jurgens

- retweets: 272, favorites: 56 (04/16/2021 09:28:03)

- links: [abs](https://arxiv.org/abs/2104.06443) | [pdf](https://arxiv.org/pdf/2104.06443)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

The framing of political issues can influence policy and public opinion. Even though the public plays a key role in creating and spreading frames, little is known about how ordinary people on social media frame political issues. By creating a new dataset of immigration-related tweets labeled for multiple framing typologies from political communication theory, we develop supervised models to detect frames. We demonstrate how users' ideology and region impact framing choices, and how a message's framing influences audience responses. We find that the more commonly-used issue-generic frames obscure important ideological and regional patterns that are only revealed by immigration-specific frames. Furthermore, frames oriented towards human interests, culture, and politics are associated with higher user engagement. This large-scale analysis of a complex social and linguistic phenomenon contributes to both NLP and social science research.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We know how politicians talk about political issues. But how about ordinary people, and why do they matter? My <a href="https://twitter.com/hashtag/NAACL2021?src=hash&amp;ref_src=twsrc%5Etfw">#NAACL2021</a> paper w/ <a href="https://twitter.com/david__jurgens?ref_src=twsrc%5Etfw">@david__jurgens</a> &amp; <a href="https://twitter.com/cerenbudak?ref_src=twsrc%5Etfw">@cerenbudak</a> address these questions by analyzing the framing of immigration on Twitter.<br>Link: <a href="https://t.co/tVDdg27THk">https://t.co/tVDdg27THk</a><br><br>🧵 (1/11) <a href="https://t.co/W2TQUAhrg3">pic.twitter.com/W2TQUAhrg3</a></p>&mdash; Julia Mendelsohn (@jmendelsohn2) <a href="https://twitter.com/jmendelsohn2/status/1382771772639080448?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 5. Sparse Attention with Linear Units

Biao Zhang, Ivan Titov, Rico Sennrich

- retweets: 158, favorites: 129 (04/16/2021 09:28:03)

- links: [abs](https://arxiv.org/abs/2104.07012) | [pdf](https://arxiv.org/pdf/2104.07012)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recently, it has been argued that encoder-decoder models can be made more interpretable by replacing the softmax function in the attention with its sparse variants. In this work, we introduce a novel, simple method for achieving sparsity in attention: we replace the softmax activation with a ReLU, and show that sparsity naturally emerges from such a formulation. Training stability is achieved with layer normalization with either a specialized initialization or an additional gating function. Our model, which we call Rectified Linear Attention (ReLA), is easy to implement and more efficient than previously proposed sparse attention mechanisms. We apply ReLA to the Transformer and conduct experiments on five machine translation tasks. ReLA achieves translation performance comparable to several strong baselines, with training and decoding speed similar to that of the vanilla attention. Our analysis shows that ReLA delivers high sparsity rate and head diversity, and the induced cross attention achieves better accuracy with respect to source-target word alignment than recent sparsified softmax-based models. Intriguingly, ReLA heads also learn to attend to nothing (i.e. 'switch off') for some queries, which is not possible with sparsified softmax alternatives.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Sparse Attention with Linear Units<br>pdf: <a href="https://t.co/g0Qsx4Clmr">https://t.co/g0Qsx4Clmr</a><br>abs: <a href="https://t.co/HP7UkNJW2W">https://t.co/HP7UkNJW2W</a> <a href="https://t.co/8rPmVB4LH3">pic.twitter.com/8rPmVB4LH3</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382497371012149258?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 6. Large-Scale Self- and Semi-Supervised Learning for Speech Translation

Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau

- retweets: 169, favorites: 51 (04/16/2021 09:28:03)

- links: [abs](https://arxiv.org/abs/2104.06678) | [pdf](https://arxiv.org/pdf/2104.06678)
- [cs.CL](https://arxiv.org/list/cs.CL/recent)

In this paper, we improve speech translation (ST) through effectively leveraging large quantities of unlabeled speech and text data in different and complementary ways. We explore both pretraining and self-training by using the large Libri-Light speech audio corpus and language modeling with CommonCrawl. Our experiments improve over the previous state of the art by 2.6 BLEU on average on all four considered CoVoST 2 language pairs via a simple recipe of combining wav2vec 2.0 pretraining, a single iteration of self-training and decoding with a language model. Different to existing work, our approach does not leverage any other supervision than ST data. Code and models will be publicly released.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Large-Scale Self- and Semi-Supervised Learning for Speech Translation<br><br>Improves over the previous SotA by 2.6 BLEU on average with wav2vec 2.0, pretraining, a single iteration of self-training and decoding with a LM. <a href="https://t.co/okuZJDlwsA">https://t.co/okuZJDlwsA</a> <a href="https://t.co/RL6d9FSk5O">pic.twitter.com/RL6d9FSk5O</a></p>&mdash; Aran Komatsuzaki (@arankomatsuzaki) <a href="https://twitter.com/arankomatsuzaki/status/1382494749773287424?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 7. Towards a framework for evaluating the safety, acceptability and  efficacy of AI systems for health: an initial synthesis

Jessica Morley, Caroline Morton, Kassandra Karpathakis, Mariarosaria Taddeo, Luciano Floridi

- retweets: 169, favorites: 23 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06910) | [pdf](https://arxiv.org/pdf/2104.06910)
- [cs.AI](https://arxiv.org/list/cs.AI/recent)

The potential presented by Artificial Intelligence (AI) for healthcare has long been recognised by the technical community. More recently, this potential has been recognised by policymakers, resulting in considerable public and private investment in the development of AI for healthcare across the globe. Despite this, excepting limited success stories, real-world implementation of AI systems into front-line healthcare has been limited. There are numerous reasons for this, but a main contributory factor is the lack of internationally accepted, or formalised, regulatory standards to assess AI safety and impact and effectiveness. This is a well-recognised problem with numerous ongoing research and policy projects to overcome it. Our intention here is to contribute to this problem-solving effort by seeking to set out a minimally viable framework for evaluating the safety, acceptability and efficacy of AI systems for healthcare. We do this by conducting a systematic search across Scopus, PubMed and Google Scholar to identify all the relevant literature published between January 1970 and November 2020 related to the evaluation of: output performance; efficacy; and real-world use of AI systems, and synthesising the key themes according to the stages of evaluation: pre-clinical (theoretical phase); exploratory phase; definitive phase; and post-market surveillance phase (monitoring). The result is a framework to guide AI system developers, policymakers, and regulators through a sufficient evaluation of an AI system designed for use in healthcare.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s <a href="https://twitter.com/hashtag/preprint?src=hash&amp;ref_src=twsrc%5Etfw">#preprint</a> time! Here is my new paper with <a href="https://twitter.com/dr_c_morton?ref_src=twsrc%5Etfw">@dr_c_morton</a>, Kassandra Karpathakis, <a href="https://twitter.com/RosariaTaddeo?ref_src=twsrc%5Etfw">@RosariaTaddeo</a> &amp; <a href="https://twitter.com/Floridi?ref_src=twsrc%5Etfw">@Floridi</a> - an initial synthesis of requirements set out in lit 4 evaluating AI CDS - designed to provide the theory 4 the many policy convos in this space: <a href="https://t.co/VgkC2fEP7o">https://t.co/VgkC2fEP7o</a></p>&mdash; Jess Morley (@jessRmorley) <a href="https://twitter.com/jessRmorley/status/1382619276830466048?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 8. Few-shot Image Generation via Cross-domain Correspondence

Utkarsh Ojha, Yijun Li, Jingwan Lu, Alexei A. Efros, Yong Jae Lee, Eli Shechtman, Richard Zhang

- retweets: 90, favorites: 73 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06820) | [pdf](https://arxiv.org/pdf/2104.06820)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.GR](https://arxiv.org/list/cs.GR/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Training generative models, such as GANs, on a target domain containing limited examples (e.g., 10) can easily result in overfitting. In this work, we seek to utilize a large source domain for pretraining and transfer the diversity information from source to target. We propose to preserve the relative similarities and differences between instances in the source via a novel cross-domain distance consistency loss. To further reduce overfitting, we present an anchor-based strategy to encourage different levels of realism over different regions in the latent space. With extensive results in both photorealistic and non-photorealistic domains, we demonstrate qualitatively and quantitatively that our few-shot model automatically discovers correspondences between source and target domains and generates more diverse and realistic images than previous methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Few-shot Image Generation via Cross-domain Correspondence<br>pdf: <a href="https://t.co/dxbGIIjIu1">https://t.co/dxbGIIjIu1</a><br>abs: <a href="https://t.co/uyDRTtTKED">https://t.co/uyDRTtTKED</a> <a href="https://t.co/IjM2vuGyT1">pic.twitter.com/IjM2vuGyT1</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382502680598351873?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 9. ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image  Registration

Junyu Chen, Yufan He, Eric C. Frey, Ye Li, Yong Du

- retweets: 72, favorites: 41 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06468) | [pdf](https://arxiv.org/pdf/2104.06468)
- [eess.IV](https://arxiv.org/list/eess.IV/recent) | [cs.CV](https://arxiv.org/list/cs.CV/recent)

In the last decade, convolutional neural networks (ConvNets) have dominated and achieved state-of-the-art performances in a variety of medical imaging applications. However, the performances of ConvNets are still limited by lacking the understanding of long-range spatial relations in an image. The recently proposed Vision Transformer (ViT) for image classification uses a purely self-attention-based model that learns long-range spatial relations to focus on the relevant parts of an image. Nevertheless, ViT emphasizes the low-resolution features because of the consecutive downsamplings, result in a lack of detailed localization information, making it unsuitable for image registration. Recently, several ViT-based image segmentation methods have been combined with ConvNets to improve the recovery of detailed localization information. Inspired by them, we present ViT-V-Net, which bridges ViT and ConvNet to provide volumetric medical image registration. The experimental results presented here demonstrate that the proposed architecture achieves superior performance to several top-performing registration methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration<br>pdf: <a href="https://t.co/0KZZtzXOID">https://t.co/0KZZtzXOID</a><br>abs: <a href="https://t.co/lWJt1YBtyN">https://t.co/lWJt1YBtyN</a><br>github: <a href="https://t.co/GdcCoDDGFH">https://t.co/GdcCoDDGFH</a> <a href="https://t.co/OvpvBDNq5I">pic.twitter.com/OvpvBDNq5I</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382497879235829761?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 10. Non-autoregressive sequence-to-sequence voice conversion

Tomoki Hayashi, Wen-Chin Huang, Kazuhiro Kobayashi, Tomoki Toda

- retweets: 42, favorites: 51 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06793) | [pdf](https://arxiv.org/pdf/2104.06793)
- [cs.SD](https://arxiv.org/list/cs.SD/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent) | [eess.AS](https://arxiv.org/list/eess.AS/recent)

This paper proposes a novel voice conversion (VC) method based on non-autoregressive sequence-to-sequence (NAR-S2S) models. Inspired by the great success of NAR-S2S models such as FastSpeech in text-to-speech (TTS), we extend the FastSpeech2 model for the VC problem. We introduce the convolution-augmented Transformer (Conformer) instead of the Transformer, making it possible to capture both local and global context information from the input sequence. Furthermore, we extend variance predictors to variance converters to explicitly convert the source speaker's prosody components such as pitch and energy into the target speaker. The experimental evaluation with the Japanese speaker dataset, which consists of male and female speakers of 1,000 utterances, demonstrates that the proposed model enables us to perform more stable, faster, and better conversion than autoregressive S2S (AR-S2S) models such as Tacotron2 and Transformer.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Non-autoregressive sequence-to-sequence voice conversion<br>pdf: <a href="https://t.co/0vHhEcjpge">https://t.co/0vHhEcjpge</a><br>abs: <a href="https://t.co/H0r3u60dIU">https://t.co/H0r3u60dIU</a><br>project page: <a href="https://t.co/GEx57qswG5">https://t.co/GEx57qswG5</a> <a href="https://t.co/RehOOzgxJF">pic.twitter.com/RehOOzgxJF</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382495348376080398?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 11. Stereo Radiance Fields (SRF): Learning View Synthesis for Sparse Views  of Novel Scenes

Julian Chibane, Aayush Bansal, Verica Lazova, Gerard Pons-Moll

- retweets: 42, favorites: 44 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06935) | [pdf](https://arxiv.org/pdf/2104.06935)
- [cs.CV](https://arxiv.org/list/cs.CV/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

Recent neural view synthesis methods have achieved impressive quality and realism, surpassing classical pipelines which rely on multi-view reconstruction. State-of-the-Art methods, such as NeRF, are designed to learn a single scene with a neural network and require dense multi-view inputs. Testing on a new scene requires re-training from scratch, which takes 2-3 days. In this work, we introduce Stereo Radiance Fields (SRF), a neural view synthesis approach that is trained end-to-end, generalizes to new scenes, and requires only sparse views at test time. The core idea is a neural architecture inspired by classical multi-view stereo methods, which estimates surface points by finding similar image regions in stereo images. In SRF, we predict color and density for each 3D point given an encoding of its stereo correspondence in the input images. The encoding is implicitly learned by an ensemble of pair-wise similarities -- emulating classical stereo. Experiments show that SRF learns structure instead of overfitting on a scene. We train on multiple scenes of the DTU dataset and generalize to new ones without re-training, requiring only 10 sparse and spread-out views as input. We show that 10-15 minutes of fine-tuning further improve the results, achieving significantly sharper, more detailed results than scene-specific models. The code, model, and videos are available at https://virtualhumans.mpi-inf.mpg.de/srf/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stereo Radiance Fields (SRF): Learning View Synthesis for Sparse Views of Novel Scenes<br>pdf: <a href="https://t.co/X7OKXw3aOx">https://t.co/X7OKXw3aOx</a><br>abs: <a href="https://t.co/uTPcFm0SqU">https://t.co/uTPcFm0SqU</a> <a href="https://t.co/7tpSKc3MI0">pic.twitter.com/7tpSKc3MI0</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382498703743848450?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 12. Revisiting Hierarchical Approach for Persistent Long-Term Video  Prediction

Wonkwang Lee, Whie Jung, Han Zhang, Ting Chen, Jing Yu Koh, Thomas Huang, Hyungsuk Yoon, Honglak Lee, Seunghoon Hong

- retweets: 42, favorites: 31 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06697) | [pdf](https://arxiv.org/pdf/2104.06697)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

Learning to predict the long-term future of video frames is notoriously challenging due to inherent ambiguities in the distant future and dramatic amplifications of prediction error through time. Despite the recent advances in the literature, existing approaches are limited to moderately short-term prediction (less than a few seconds), while extrapolating it to a longer future quickly leads to destruction in structure and content. In this work, we revisit hierarchical models in video prediction. Our method predicts future frames by first estimating a sequence of semantic structures and subsequently translating the structures to pixels by video-to-video translation. Despite the simplicity, we show that modeling structures and their dynamics in the discrete semantic structure space with a stochastic recurrent estimator leads to surprisingly successful long-term prediction. We evaluate our method on three challenging datasets involving car driving and human dancing, and demonstrate that it can generate complicated scene structures and motions over a very long time horizon (i.e., thousands frames), setting a new standard of video prediction with orders of magnitude longer prediction time than existing approaches. Full videos and codes are available at https://1konny.github.io/HVP/.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Revisiting Hierarchical Approach for Persistent Long-Term Video Prediction<br>pdf: <a href="https://t.co/nFkwCFpKMh">https://t.co/nFkwCFpKMh</a><br>abs: <a href="https://t.co/0hdrfUAyvw">https://t.co/0hdrfUAyvw</a><br>project page: <a href="https://t.co/yOV5Tmzbny">https://t.co/yOV5Tmzbny</a><br>github: <a href="https://t.co/EXUfBGW9r1">https://t.co/EXUfBGW9r1</a> <a href="https://t.co/tPFn9Xhiqw">pic.twitter.com/tPFn9Xhiqw</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382535644799504386?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 13. Pose Recognition with Cascade Transformers

Ke Li, Shijie Wang, Xiang Zhang, Yifan Xu, Weijian Xu, Zhuowen Tu

- retweets: 42, favorites: 27 (04/16/2021 09:28:04)

- links: [abs](https://arxiv.org/abs/2104.06976) | [pdf](https://arxiv.org/pdf/2104.06976)
- [cs.CV](https://arxiv.org/list/cs.CV/recent)

In this paper, we present a regression-based pose recognition method using cascade Transformers. One way to categorize the existing approaches in this domain is to separate them into 1). heatmap-based and 2). regression-based. In general, heatmap-based methods achieve higher accuracy but are subject to various heuristic designs (not end-to-end mostly), whereas regression-based approaches attain relatively lower accuracy but they have less intermediate non-differentiable steps. Here we utilize the encoder-decoder structure in Transformers to perform regression-based person and keypoint detection that is general-purpose and requires less heuristic design compared with the existing approaches. We demonstrate the keypoint hypothesis (query) refinement process across different self-attention layers to reveal the recursive self-attention mechanism in Transformers. In the experiments, we report competitive results for pose recognition when compared with the competing regression-based methods.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Pose Recognition with Cascade Transformers<br>pdf: <a href="https://t.co/Qciy1xWzbA">https://t.co/Qciy1xWzbA</a><br>abs: <a href="https://t.co/x4bwnKOx9W">https://t.co/x4bwnKOx9W</a> <a href="https://t.co/lLNwY4GTfJ">pic.twitter.com/lLNwY4GTfJ</a></p>&mdash; AK (@ak92501) <a href="https://twitter.com/ak92501/status/1382553030189580288?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 14. MS2: Multi-Document Summarization of Medical Studies

Jay DeYoung, Iz Beltagy, Madeleine van Zuylen, Bailey Keuhl, Lucy Lu Wang

- retweets: 32, favorites: 34 (04/16/2021 09:28:05)

- links: [abs](https://arxiv.org/abs/2104.06486) | [pdf](https://arxiv.org/pdf/2104.06486)
- [cs.CL](https://arxiv.org/list/cs.CL/recent) | [cs.AI](https://arxiv.org/list/cs.AI/recent) | [cs.LG](https://arxiv.org/list/cs.LG/recent)

To assess the effectiveness of any medical intervention, researchers must conduct a time-intensive and highly manual literature review. NLP systems can help to automate or assist in parts of this expensive process. In support of this goal, we release MS^2 (Multi-Document Summarization of Medical Studies), a dataset of over 470k documents and 20k summaries derived from the scientific literature. This dataset facilitates the development of systems that can assess and aggregate contradictory evidence across multiple studies, and is the first large-scale, publicly available multi-document summarization dataset in the biomedical domain. We experiment with a summarization system based on BART, with promising early results. We formulate our summarization inputs and targets in both free text and structured forms and modify a recently proposed metric to assess the quality of our system's generated summaries. Data and models are available at https://github.com/allenai/ms2




# 15. Efficiently Teaching an Effective Dense Retriever with Balanced Topic  Aware Sampling

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, Allan Hanbury

- retweets: 20, favorites: 40 (04/16/2021 09:28:05)

- links: [abs](https://arxiv.org/abs/2104.06967) | [pdf](https://arxiv.org/pdf/2104.06967)
- [cs.IR](https://arxiv.org/list/cs.IR/recent) | [cs.CL](https://arxiv.org/list/cs.CL/recent)

A vital step towards the widespread adoption of neural retrieval models is their resource efficiency throughout the training, indexing and query workflows. The neural IR community made great advancements in training effective dual-encoder dense retrieval (DR) models recently. A dense text retrieval model uses a single vector representation per query and passage to score a match, which enables low-latency first stage retrieval with a nearest neighbor search. Increasingly common, training approaches require enormous compute power, as they either conduct negative passage sampling out of a continuously updating refreshing index or require very large batch sizes for in-batch negative sampling. Instead of relying on more compute capability, we introduce an efficient topic-aware query and balanced margin sampling technique, called TAS-Balanced. We cluster queries once before training and sample queries out of a cluster per batch. We train our lightweight 6-layer DR model with a novel dual-teacher supervision that combines pairwise and in-batch negative teachers. Our method is trainable on a single consumer-grade GPU in under 48 hours (as opposed to a common configuration of 8x V100s). We show that our TAS-Balanced training method achieves state-of-the-art low-latency (64ms per query) results on two TREC Deep Learning Track query sets. Evaluated on NDCG@10, we outperform BM25 by 44%, a plainly trained DR by 19%, docT5query by 11%, and the previous best DR model by 5%. Additionally, TAS-Balanced produces the first dense retriever that outperforms every other method on recall at any cutoff on TREC-DL and allows more resource intensive re-ranking models to operate on fewer passages to improve results further.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can we efficiently train a very effective dense passage retriever? Yes, with Balanced Topic Aware Sampling! Let me introduce our <a href="https://twitter.com/hashtag/sigir2021?src=hash&amp;ref_src=twsrc%5Etfw">#sigir2021</a> full paper: We compose batches based on query clusters and ...🧵<br>w/ <a href="https://twitter.com/jacklin_64?ref_src=twsrc%5Etfw">@jacklin_64</a> <a href="https://twitter.com/mattjustram?ref_src=twsrc%5Etfw">@mattjustram</a> <a href="https://twitter.com/lintool?ref_src=twsrc%5Etfw">@lintool</a> <a href="https://twitter.com/allanhanbury?ref_src=twsrc%5Etfw">@allanhanbury</a> <a href="https://t.co/sH4ToSL2l0">https://t.co/sH4ToSL2l0</a> <a href="https://t.co/N7zYCSuxHy">pic.twitter.com/N7zYCSuxHy</a></p>&mdash; Sebastian Hofstätter (@s_hofstaetter) <a href="https://twitter.com/s_hofstaetter/status/1382680647509106688?ref_src=twsrc%5Etfw">April 15, 2021</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>




# 16. Can Active Learning Preemptively Mitigate Fairness Issues?

Frédéric Branchaud-Charron, Parmida Atighehchian, Pau Rodríguez, Grace Abuhamad, Alexandre Lacoste

- retweets: 42, favorites: 13 (04/16/2021 09:28:05)

- links: [abs](https://arxiv.org/abs/2104.06879) | [pdf](https://arxiv.org/pdf/2104.06879)
- [cs.LG](https://arxiv.org/list/cs.LG/recent) | [cs.CY](https://arxiv.org/list/cs.CY/recent)

Dataset bias is one of the prevailing causes of unfairness in machine learning. Addressing fairness at the data collection and dataset preparation stages therefore becomes an essential part of training fairer algorithms. In particular, active learning (AL) algorithms show promise for the task by drawing importance to the most informative training samples. However, the effect and interaction between existing AL algorithms and algorithmic fairness remain under-explored. In this paper, we study whether models trained with uncertainty-based AL heuristics such as BALD are fairer in their decisions with respect to a protected class than those trained with identically independently distributed (i.i.d.) sampling. We found a significant improvement on predictive parity when using BALD, while also improving accuracy compared to i.i.d. sampling. We also explore the interaction of algorithmic fairness methods such as gradient reversal (GRAD) and BALD. We found that, while addressing different fairness issues, their interaction further improves the results on most benchmarks and metrics we explored.



