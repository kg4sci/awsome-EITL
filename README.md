# Expert-in-the-Loop Systems for High-Quality Training Data Construction: A Survey
## Abstract
High-quality supervised training data constitutes the foundation of reliable intelligent systems in safety-critical and domain-intensive areas such as healthcare, intelligent transportation, smart manufacturing, cyber–physical systems, and scientific discovery systems. 
Yet, the construction of such data remains critically dependent on domain experts, whose nuanced judgment is irreplaceable for tasks involving deep domain knowledge, contextual reasoning, and professional standards. To bridge the gap between expert insight and scalable data production, a growing body of research has advanced the Expert-in-the-Loop (EITL) paradigm—integrating human expertise with intelligent algorithms and collaborative platforms throughout the data lifecycle. 
In this survey, we present a systems-oriented and unified analysis of EITL methodologies through two orthogonal dimensions: the evolution of data types and the collaboration paradigm. We systematically categorize existing approaches according to the roles of experts and the augmentation mechanisms of intelligent systems, ranging from rule-based pre-labeling and active sampling to generative synthesis and automated conflict resolution. By abstracting common architectural patterns and interaction mechanisms, this survey establishes a structured taxonomy and conceptual framework for expert-guided data engineering. It further outlines open challenges and future directions toward scalable, reliable, and human-centered data construction for modern AI training paradigms.

## Introduction
With the continuous advancement of large language model (LLM) technologies, their applications in natural language understanding, content generation, intelligent dialogue, educational assistance, and cross-modal retrieval have become increasingly widespread, demonstrating strong general-purpose capabilities. However, in specialized domains such as healthcare, law, and finance, the practical performance of large models still faces significant challenges. These fields demand high levels of accuracy, rigor, and interpretability, yet existing training data often suffer from poor quality, inconsistent terminology, and lack of contextual depth, leading to limited comprehension, unreliable reasoning, and potentially misleading outputs. The fundamental root cause of this limitation resides in the pre-training paradigm of most large models: they are typically pre-trained on massive volumes of unsupervised general-domain data, which lack structured semantic labels and domain-specific knowledge depth. In contrast, supervised training data provide explicit input-output mappings that effectively guide models in learning complex semantics and logical rules. More importantly, when such data are systematically annotated by Experts, their accuracy and authority are significantly enhanced. Experts can discern subtle professional distinctions, ensure contextual coherence, and validate factual correctness, thereby constructing high-quality corpora that truly reflect domain-specific knowledge structures. Training data generated through deep expert involvement not only serve as reliable supervision signals during model fine-tuning but also lay the foundation for trustworthy deployment in high-stakes environments. Therefore, building a high-quality supervised data ecosystem centered on expert annotation has become a critical pathway for advancing large models toward specialization, precision, and reliability.
## Background
Human-in-the-loop (HITL) is a foundational paradigm in machine learning that integrates human input throughout the training, refinement, and evaluation of AI systems, playing a pivotal role in the creation of high-quality training data. Traditionally, HITL frameworks treat human participants as generic annotators or end users whose primary function is to provide labels or feedback based on surface-level cues. However, this view overlooks a crucial distinction: not all human input is equivalent. In knowledge-intensive domains, the involvement of Experts, which means professionals with deep conceptual understanding and procedural expertise, is essential for ensuring both the accuracy and epistemic validity of AI-driven decisions.
## Taxonomy
In this section, we propose a unified taxonomy that categorizes current approaches to EITL-based data generation based on the type of training data being produced and Expert-AI collaboration mechanism. Building upon the three core data categories introduced in Section II, which are supervised data, instruction-following data, and preference data, we further conduct a vertical review from the perspective of intelligence levels and characterize them in terms of methodological principles, platform requirements, and workflow patterns.
### Overview of the Taxonomy
We present a systematic taxonomy categorizing methodologies for expert-involved dataset construction. The framework is structured along two orthogonal dimensions: the Evolution of Data Type, representing the increasing abstraction of supervision signals, and Increasing AI Agency, illustrating the paradigm shift in human-machine collaboration. This taxonomy highlights the transition from labor-intensive manual annotation to scalable, AI-driven curation strategies guided by expert knowledge.
### Evolution of Data Type
 Under the EITL framework, Experts enhance the quality of supervised, instruction-following, and preference data by injecting deep domain knowledge. They provide accurate labels, author semantically rich instructions, and deliver nuanced feedback, ensuring training signals reflect not only patterns but also professional reasoning and judgment.
### Collaboration Paradigm
For each type of data, we conduct a comparative analysis of EITL from the perspective of intelligence levels, showing how the role of the expert transforms from a laborer to a legislator. 
### Summary
As a summary, this survey is organized along two complementary dimensions to form a clear, extensible, and easily navigable map of the field: (i) an orthogonal taxonomy grounded in data types × collaboration paradigm, which structures the problem space and research targets; and (ii) a methodological and platform-oriented lens that summarizes the end-to-end solution chain from algorithmic design to system deployment, enabling a holistic synthesis of prior work.

| Data Type | Core Methods | Features |
| :-----| :----- | :----- |
| Traditional Supervised Data | Expert-Centric Adjudication | Expert crowdsourcing, adjudication protocols, gold standards |
|   | Interactive Active Selection for Expert | Active Learning, pre-labeling |
|   | Programmatic Weak Supervision by Expert | Rules-based Learning, weak supervision |
| Instruction-Following Data | Human-Authored Instruction Curation | Crowdsourcing, human-annotated instructions, gold standards |
|   | Augmented Instruction Evolution | Instruction backtranslation, Iterative Filtering |
|   | Schema-Driven Instruction Synthesis | Instruction-driven learning, instruction mining |
| Preference Data | Expert-Driven Preference Data Generation | Expert full sample annotation, manual preference labeling |
|   | Human-Centered, AI-Assisted Preference Data Generation | Active Preference Learning |
|   | AI-Driven Preference Data Generation with Expert Oversight | Self-supervised learning, multi-agent optimization |



## References

[1. X. Wu, L. Xiao, Y. Sun, J. Zhang, T. Ma, and L. He, “A survey of human-in-the-loop for machine learning,” Future Generation Computer Systems, vol. 135, pp. 364–381, 2022](https://doi.org/10.1016/j.future.2022.05.014)

[2. M. Neves and U. Leser, “A survey on annotation tools for the biomedical literature,” Briefings in Bioinformatics, vol. 15, no. 2, pp. 327–340, Dec. 2012](https://doi.org/10.1093/bib/bbs084)

[3. Y. Roh, G. Heo, and S. E. Whang, “A Survey on Data Collection for Machine Learning: A Big Data - AI Integration Perspective,” IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 4, pp. 1328–1347, 2021](https://doi.org/10.1109/TKDE.2019.2946162)

[4. Z. Z. Chen et al., “A survey on large language models for critical societal domains: Finance, healthcare, and law,” arXiv preprint arXiv:2405.01769, 2024](https://api.semanticscholar.org/CorpusID:269587715)

[5. L. Lin et al., “ACT as human: Multimodal large language model data annotation with critical thinking,” arXiv preprint arXiv:2511.09833, 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/117727)

[6. Y. Du, L. Wang, M. Huang, D. Song, W. Cui, and Y. Zhou, “Autodive: An integrated onsite scientific literature annotation tool,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 2023, pp. 76–85](https://aclanthology.org/2023.acl-demo.7/)

[7. J. Jukić, F. Jelenić, M. Bićanić, and J. Snajder, “ALANNO: An Active Learning Annotation System for Mortals,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, D. Croce and L. Soldaini, Eds., Dubrovnik, Croatia: Association for Computational Linguistics, May 2023, pp. 228–235](https://doi.org/10.18653/v1/2023.eacl-demo.26)\
[8. Argilla, “Argilla: The tool where experts improve AI models.” 2026](https://argilla.io)

[9. H. Zhang, X. Fu, and J. M. Carroll, “Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation,” ICLR 2025 Workshop on Bidirectional Human-AI Alignment, 2025](https://openreview.net/forum?id=ZPTvEUXzSq)

[10. S. Zhou et al., “Automating expert-level medical reasoning evaluation of large language models,” npj Digital Medicine, 2025](https://www.nature.com/articles/s41746-025-02208-7)

[11. C. Xu, D. Guo, N. Duan, and J. McAuley, “Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds., Singapore: Association for Computational Linguistics, Dec. 2023, pp. 6268–6278](https://doi.org/10.18653/v1/2023.emnlp-main.385)

[12. T. Nguyen et al., “Better Alignment with Instruction Back-and-Forth Translation,” in Findings of the Association for Computational Linguistics: EMNLP 2024, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds., Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 13289–13308](https://doi.org/10.18653/v1/2024.findings-emnlp.777)

[13. R. Snow, B. O’Connor, D. Jurafsky, and A. Ng, “Cheap and Fast – But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks,” in Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing, M. Lapata and H. T. Ng, Eds., Honolulu, Hawaii: Association for Computational Linguistics, Oct. 2008, pp. 254–263](https://aclanthology.org/D08-1027/)

[14. B. Han et al., “Co-teaching: Robust training of deep neural networks with extremely noisy labels,” Advances in neural information processing systems, vol. 31, 2018](https://papers.nips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)

[15. C. Northcutt, L. Jiang, and I. Chuang, “Confident learning: Estimating uncertainty in dataset labels,” Journal of Artificial Intelligence Research, vol. 70, pp. 1373–1411, 2021](https://doi.org/10.1613/jair.1.12125)

[16. Y. Bai et al., “Constitutional AI: Harmlessness from AI Feedback.” 2022.](https://arxiv.org/abs/2212.08073)

[17. Y. Qi, H. Peng, X. Wang, B. Xu, L. Hou, and J. Li, “Constraint Back-translation Improves Complex Instruction Following of Large Language Models,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, in CIKM ’25. New York, NY, USA: Association for Computing Machinery, 2025, pp. 2388–2398](https://doi.org/10.1145/3746252.3761324)

[18. J. He, S. Sun, S. Peng, J. Xu, X. Jia, and W. Li, “Contrastive Preference Learning for Neural Machine Translation,” in Findings of the Association for Computational Linguistics: NAACL 2024, K. Duh, H. Gomez, and S. Bethard, Eds., Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 2723–2735](https://doi.org/10.18653/v1/2024.findings-naacl.174)

[19. H. Que et al., “D-cpt law: Domain-specific continual pre-training scaling law for large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 90318–90354, 2024](https://neurips.cc/virtual/2024/poster/95686)

[20. A. J. Ratner, C. M. De Sa, S. Wu, D. Selsam, and C. Ré, “Data programming: Creating large training sets, quickly,” Advances in neural information processing systems, vol. 29, 2016](https://papers.nips.cc/paper_files/paper/2016/hash/6709e8d64a5f47269ed5cea9f625f7ab-Abstract.html)

[21. P. Hsueh, P. Melville, and V. Sindhwani, “Data Quality from Crowdsourcing: A Study of Annotation Selection Criteria,” in HLT-NAACL 2009, 2009](https://api.semanticscholar.org/CorpusID:3954835)

[22. D. Zha, Z. P. Bhat, K.-H. Lai, F. Yang, and X. Hu, “Data-centric AI: Perspectives and Challenges,” in SDM, 2023](https://api.semanticscholar.org/CorpusID:255749143)

[23. Y. Liu, J. Cao, C. Liu, K. Ding, and L. Jin, “Datasets for large language models: a comprehensive survey,” Artificial Intelligence Review, vol. 58, 2025](https://api.semanticscholar.org/CorpusID:282597320)

[24. THUDM, “DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL.” 2026](https://github.com/THUDM/DeepDive)

[25. C. D. Sa et al., “DeepDive: Declarative Knowledge Base Construction,” SIGMOD record, vol. 45 1, pp. 60–67, 2016](https://api.semanticscholar.org/CorpusID:10394118)

[26. J. C. L. Ong et al., “Ethical and regulatory challenges of large language models in medicine,” The Lancet Digital Health, vol. 6, no. 6, pp. e428–e432, 2024](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00061-X/fulltext)

[27. W. Ye et al., “Evaluation of cell type annotation reliability using a large language model-based identifier,” Communications Biology, vol. 8, no. 1, p. 1360, 2025](https://www.nature.com/articles/s42003-025-08745-x)

[28. Y. Xiao et al., “Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds., Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 12538–12552](https://doi.org/10.18653/v1/2025.acl-long.615)

[29. J.-B. Alayrac et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23716–23736, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)

[30. Z. Wang et al., “HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025](https://openreview.net/forum?id=lovsIkZLnI)

[31. Amazon Web Services, “High-quality human feedback for your generative AI applications from Amazon SageMaker Ground Truth Plus | Artificial Intelligence.” 2023](https://aws.amazon.com/cn/blogs/machine-learning/high-quality-human-feedback-for-your-generative-ai-applications-from-amazon-sagemaker-ground-truth-plus/)

[32. Humanloop, “Humanloop: LLM evals platform for enterprises.” 2026](https://humanloop.com/home)

[33. Q. Chen et al., “Icon^2: Aligning Large Language Models Using Self-Synthetic Preference Data via Inherent Regulation,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds., Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 3949–3968](https://doi.org/10.18653/v1/2025.emnlp-main.196)

[34. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009](https://api.semanticscholar.org/CorpusID:57246310)

[35. A. Glaese et al., “Improving alignment of dialogue agents via targeted human judgements.” 2022](https://arxiv.org/abs/2209.14375)

[36. L. Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc., 2023, pp. 46595–46623.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

[37. HumanSignal, “Label Studio: Open Source Data Labeling.” 2026](https://labelstud.io)

[38. Labelbox, “Labelbox: The data factory for AI teams.” 2026](https://labelbox.com)

[39. M. Wu, A. Waheed, C. Zhang, M. Abdul-Mageed, and A. F. Aji, “LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), Y. Graham and M. Purver, Eds., St. Julian’s, Malta: Association for Computational Linguistics, Mar. 2024, pp. 944–964](https://doi.org/10.18653/v1/2024.eacl-long.57)

[40. T. Brown et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)

[41. R. Lou, K. Zhang, and W. Yin, “Large Language Model Instruction Following: A Survey of Progresses and Challenges,” Computational Linguistics, vol. 50, no. 3, pp. 1053–1095, Sep. 2024](https://doi.org/10.1162/coli_a_00523)

[42. A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, “Large language models in medicine,” Nature medicine, vol. 29, no. 8, pp. 1930–1940, 2023](https://www.nature.com/articles/s41591-023-02448-8)

[43. Z. Ye et al., “Learning LLM-as-a-Judge for Preference Alignment,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., 2025, pp. 3537–3564](https://proceedings.iclr.cc/paper_files/paper/2025/hash/09fd990b19b2e69cc4d20e9969e43f09-Abstract-Conference.html)

[44. D. Yoo and I. S. Kweon, “Learning loss for active learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 93–102](https://doi.org/10.1109/CVPR.2019.00018)

[45. K. Sohn, H. Lee, and X. Yan, “Learning structured output representation using deep conditional generative models,” Advances in neural information processing systems, vol. 28, 2015](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)

[46. S. Teso, Ö. Alkan, W. Stammer, and E. Daly, “Leveraging explanations in interactive machine learning: An overview,” Frontiers in Artificial Intelligence, vol. Volume 6-2023, 2023](https://doi.org/10.3389/frai.2023.1066049)

[47. B. Ke et al., “Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis,” IEEE Transactions on Pattern Analysis \& Machine Intelligence, 2025](https://doi.org/10.1109/TPAMI.2025.3591076)

[48. S. Dobbie et al., “Markup: A Web-Based Annotation Tool Powered by Active Learning,” Frontiers in Digital Health, vol. 3, p. 598916, 2021](https://doi.org/10.3389/fdgth.2021.598916)

[49. Q. Xie et al., “Medical foundation large language models for comprehensive text analysis and beyond,” npj Digital Medicine, vol. 8, no. 1, p. 141, 2025](https://www.nature.com/articles/s41746-025-01533-1)

[50. H. Kim, K. Mitra, R. L. Chen, S. Rahman, and D. Zhang, “MEGAnno+: A Human-LLM Collaborative Annotation System,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, 2024](https://aclanthology.org/2024.eacl-demo.18/)

[51. L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei, “Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels,” in International conference on machine learning, PMLR, 2018, pp. 2304–2313](https://icml.cc/virtual/2018/poster/1952)

[52. M. Li et al., “Mosaic-IT: Cost-Free Compositional Data Synthesis for Instruction Tuning,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds., Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 25287–25318](https://doi.org/10.18653/v1/2025.findings-acl.1297)

[53. H. Huang et al., “Multi-view fusion for instruction mining of large language model,” Information Fusion, vol. 110, p. 102480, 2024](https://doi.org/10.1016/j.inffus.2024.102480)

[54. Y. Qi et al., “Next Generation Active Learning: Mixture of LLMs in the Loop,” arXiv preprint arXiv:2601.15773, 2026](https://arxiv.org/abs/2601.15773)

[55. L. Long et al., “On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey,” in Findings of the Association for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds., Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 11065–11082](https://doi.org/10.18653/v1/2024.findings-acl.658)

[56 A. Köpf et al., “OpenAssistant Conversations - Democratizing Large Language Model Alignment,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc., 2023, pp. 47669–47681](https://proceedings.neurips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html)

[57. OpenTrain AI, “OpenTrain: The AI Trainer & Data Labeler Marketplace.” 2026](https://www.opentrain.ai)

[58. J. Ji et al., “PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds., Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 31983–32016](https://doi.org/10.18653/v1/2025.acl-long.1544)

[59. Explosion, “Prodigy: An annotation tool for AI, Machine Learning & NLP.” 2026](https://prodi.gy/)

[60. Prolific, “Prolific: Easily collect high-quality data from real people.” 2026](https://prolific.com)

[61. M. Xia et al., “Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, 2025, pp. 2750–2770](https://aclanthology.org/2025.acl-long.139/)

[62. S. Hu, W. Hu, Y. Su, and F. Zhang, “RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning,” arXiv preprint arXiv:2508.13229, 2025](https://arxiv.org/abs/2508.13229)

[63. T. Shi, K. Chen, and J. Zhao, “Safer-Instruct: Aligning Language Models with Automated Preference Data,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard, Eds., Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 7636–7651](https://doi.org/10.18653/v1/2024.naacl-long.422)

[64. Samasource Impact Sourcing, “Sama: Data Annotation & Labeling Company.” 2026](https://www.sama.com)

[65. P. Colombo et al., “Saullm-54b & saullm-141b: Scaling up domain adaptation for the legal domain,” Advances in Neural Information Processing Systems, vol. 37, pp. 129672–129695, 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea3f85a33f9ba072058e3df233cf6cca-Abstract-Conference.html)

[66. Scale AI, “Scale AI: Reliable AI Systems for the World’s Most Important Decisions.” 2026](https://scale.com)

[67. X. Li et al., “Self-Alignment with Instruction Backtranslation,” in International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., 2024, pp. 3552–3577](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0f8e3534eb8dee7478d4dc0e9d9a0b1a-Abstract-Conference.html)

[68. Y. Wang et al., “Self-Instruct: Aligning Language Models with Self-Generated Instructions,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds., Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 13484–13508](https://doi.org/10.18653/v1/2023.acl-long.754)

[69. F. Gao, X. Zhang, B. Ni, C. Wang, and L. Chen, “Self-preference: An Automated Method for Preference-Aligned Data Constructed from Business Metrics,” in China National Conference on Chinese Computational Linguistics, Springer, 2025, pp. 303–321](https://link.springer.com/chapter/10.1007/978-981-95-2725-0_19)

[70. A. Madaan et al., “Self-Refine: Iterative Refinement with Self-Feedback,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc., 2023, pp. 46534–46594](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)

[71. Z. Feng, D. Ram, C. Hawkins, A. Rawal, J. Zhao, and S. Zha, “Sequence-level Large Language Model Training with Contrastive Preference Optimization,” in Findings of the Association for Computational Linguistics: NAACL 2025, L. Chiruzzo, A. Ritter, and L. Wang, Eds., Albuquerque, New Mexico: Association for Computational Linguistics, Apr. 2025, pp. 4158–4164](https://doi.org/10.18653/v1/2025.findings-naacl.233)

[72. C. Y. Liu et al., “Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy.” 2025](https://arxiv.org/abs/2507.01352)

[73. H. Kim et al., “Small language models learn enhanced reasoning skills from medical textbooks,” NPJ digital medicine, vol. 8, no. 1, p. 240, 2025](https://www.nature.com/articles/s41746-025-01653-8)

[74. S. H. Bach et al., “Snorkel drybell: A case study in deploying weak supervision at industrial scale,” in Proceedings of the 2019 International Conference on Management of Data, 2019, pp. 362–375](https://doi.org/10.1145/3299869.3314036)

[75. A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, “Snorkel: rapid training data creation with weak supervision,” Proc. VLDB Endow., vol. 11, no. 3, pp. 269–282, Nov. 2017](https://doi.org/10.14778/3157794.3157797)

[76. P. Varma and C. Ré, “Snuba: Automating weak supervision to label training data,” in Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases, 2018, p. 223](https://doi.org/10.14778/3291264.3291268)

[77. D. Kim, K. Lee, J. Shin, and J. Kim, “Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment,” in International Conference on Learning Representations, Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, Eds., 2025, pp. 20361–20382](https://proceedings.iclr.cc/paper_files/paper/2025/hash/342e5fc02b86dec9b24e41b22968e539-Abstract-Conference.html)

[78. Y. Wang et al., “Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds., Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 5085–5109](https://doi.org/10.18653/v1/2022.emnlp-main.340)

[79. Surge AI, “Surge AI: Human Intelligence for AGI.” 2025](https://www.surgehq.ai)

[80. J. Gu, C. Gao, and L. Wang, “The evolution of artificial intelligence in biomedicine: Bibliometric analysis,” JMIR AI, vol. 2, p. e45770, 2023](https://ai.jmir.org/2023/1/e45770/)

[81. Toloka AI, “Toloka: Training data for AI agents and LLMs.” 2026](https://toloka.ai)

[82. A. Ratner, B. Hancock, J. Dunnmon, F. Sala, S. Pandey, and C. Ré, “Training complex models with multi-task weak supervision,” in Proceedings of the AAAI conference on artificial intelligence, 2019, pp. 4763–4771](https://doi.org/10.1609/aaai.v33i01.33014763)

[83. L. Ouyang et al., “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., Curran Associates, Inc., 2022, pp. 27730–27744](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html)

[84. H. Li, Y. Liu, X. Zhang, W. Lu, and F. Wei, “Tuna: Instruction Tuning using Feedback from Large Language Models,” in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds., Singapore: Association for Computational Linguistics, Dec. 2023, pp. 15146–15163](https://doi.org/10.18653/v1/2023.findings-emnlp.1011)

[85. G. Cui et al., “ULTRAFEEDBACK: Boosting Language Models with Scaled AI Feedback,” in Proceedings of the 41st International Conference on Machine Learning, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, Eds., in Proceedings of Machine Learning Research, vol. 235. PMLR, Jul. 2024, pp. 9722–9744](https://proceedings.mlr.press/v235/cui24f.html)

[86. O. Honovich, T. Scialom, O. Levy, and T. Schick, “Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds., Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 14409–14428](https://doi.org/10.18653/v1/2023.acl-long.806)

[87. H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34892–34916, 2023](https://papers.nips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html)

[88. J. Whitehill, P. Ruvolo, T. Wu, J. Bergsma, and J. Movellan, “Whose vote should count more: optimal integration of labels from labelers of unknown expertise,” in Proceedings of the 23rd International Conference on Neural Information Processing Systems, in NIPS’09. Red Hook, NY, USA: Curran Associates Inc., 2009, pp. 2035–2043](https://papers.nips.cc/paper_files/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html)

[89. C. Xu et al., “WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions,” in International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., 2024, pp. 30745–30766](https://proceedings.iclr.cc/paper_files/paper/2024/hash/82eec786fdfbbfa53450c5feb7d1ac92-Abstract-Conference.html)

[90. O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” arXiv preprint arXiv:1708.00489, 2017](https://api.semanticscholar.org/CorpusID:3383786)

[91. Z. Wang et al., “A Comprehensive Survey on Data Augmentation,” IEEE Transactions on Knowledge and Data Engineering, vol. 38, no. 1, pp. 47–66, 2026](https://doi.org/10.1109/TKDE.2025.3622600)

[92. M.-F. Ge, J.-Z. Xu, Z.-W. Liu, and J. Huang, “A Mode-Switched Control Architecture for Human-in-the-Loop Teleoperation of Multislave Robots via Data-Training-Based Observer,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 4, pp. 2471–2483, 2024](https://doi.org/10.1109/TSMC.2023.3344879)

[93. T. Gai, J. Wu, F. Chiclana, M. Zhou, and W. Pedrycz, “A Personality Traits-Driven Conflict Quadrant Diagram by Large Language Models for Personalized Feedback in Group Decision-Making,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 11, pp. 8506–8518, 2025](https://doi.org/10.1109/TSMC.2025.3605404)

[94. X.-H. Li et al., “A Survey of Data-Driven and Knowledge-Aware eXplainable AI,” IEEE Transactions on Knowledge and Data Engineering, vol. 34, no. 1, pp. 29–49, 2022](https://doi.org/10.1109/TKDE.2020.2983930)

[95. H. Yang, Z. Li, W. Pedrycz, and M. Farina, “Deep Learning From Crowds on a Healthy Data Diet,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 9, pp. 6150–6163, 2025](https://doi.org/10.1109/TSMC.2025.3578893)

[96. X. Sun, K. Shi, H. Tang, D. Wang, G. Xu, and Q. Li, “Educating Language Models as Promoters: Multi-Aspect Instruction Alignment With Self-Augmentation,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4564–4577, 2025](https://doi.org/10.1109/TKDE.2025.3569585)

[97. Y. Zhang, T. Zhao, D. Miao, and W. Pedrycz, “Granular Multilabel Batch Active Learning With Pairwise Label Correlation,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 5, pp. 3079–3091, 2022](https://doi.org/10.1109/TSMC.2021.3062714)

[98. Z. Tan et al., “Human–Machine Interaction in Intelligent and Connected Vehicles: A Review of Status Quo, Issues, and Opportunities,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 9, pp. 13954–13975, 2022](https://doi.org/10.1109/TITS.2021.3127217)

[99. J. Zhao, L. Zhou, C. Liu, Y. Jiang, S. Kwong, and Z.-H. Zhan, “Multimodule-Based Dynamic Community Detection for Enhancing Innovation Performance in Crowdsourcing Contests,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 11, pp. 8289–8303, 2025](https://doi.org/10.1109/TSMC.2025.3603622)

[100. J. Kumar, J. Shao, R. Kumar, S. U. Din, C. B. Mawuli, and Q. Yang, “Online Semi-Supervised Classification on Multilabel Evolving High-Dimensional Text Streams,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 10, pp. 5983–5995, 2023](https://doi.org/10.1109/TSMC.2023.3275298)

[101. M. Sharifi, S. Tripathi, Y. Chen, Q. Zhang, and M. Tavakoli, “Reinforcement Learning Methods for Assistive and Rehabilitation Robotic Systems: A Survey,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 7, pp. 4534–4551, 2025](https://doi.org/10.1109/TSMC.2025.3555598)

[102. X. Li, X. Wang, F. Deng, and F.-Y. Wang, “Scenarios Engineering for Trustworthy AI: Domain Adaptation Approach for Reidentification With Synthetic Data,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 11, pp. 6901–6910, 2024](https://doi.org/10.1109/TSMC.2024.3445117)

[103. C. Zhu, L. Cui, Y. Tang, and J. Wang, “The Evolution and Future Perspectives of Artificial Intelligence-Generated Content,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 56, no. 1, pp. 546–564, 2026](https://doi.org/10.1109/TSMC.2025.3627806)

[104. L. Ren, H. Wang, J. Li, Y. Tang, and C. Yang, “AIGC for Industrial Time Series: From Deep-Generative Models to Large-Generative Models,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 11, pp. 7774–7791, 2025](https://doi.org/10.1109/TSMC.2025.3598252)

[105. G. He, B. Li, H. Wang, and W. Jiang, “Cost-Effective Active Semi-Supervised Learning on Multivariate Time Series Data With Crowds,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 3, pp. 1437–1450, 2022](https://doi.org/10.1109/TSMC.2020.3019531)

[106. M. Wang, C. Yang, F. Zhao, F. Min, and X. Wang, “Cost-Sensitive Active Learning for Incomplete Data,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 53, no. 1, pp. 405–416, 2023](https://doi.org/10.1109/TSMC.2022.3182122)

[107. Y. Wu et al., “Deep Active Learning for Image Hierarchical Classification by Introducing Dependencies and Constraints Between Classes,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 6, pp. 4396–4409, 2025](https://doi.org/10.1109/TSMC.2025.3552667)

[108. A. Qin, Y. Zhou, L. Wang, X. Xue, and J. Pu, “Natural Language to Code for Automated Annotation in Autonomous Driving,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 56, no. 2, pp. 1394–1407, 2026](https://doi.org/10.1109/TSMC.2025.3648552)
