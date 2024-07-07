# KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques

Large language models (LLMs) have brought unprecedented innovation to the healthcare field. Despite the promising application in healthcare, they face significant challenges since the generated-content might deviate from established medical facts and even exhibit potential biases. We develop an augmented LLM framework based on the Unified Medical Language System (UMLS), aiming to better serve the healthcare community. It's noteworthy that multiple resident physicians conducted blind reviews of the generated content, and the results indicate that our framework effectively enhances the factuality, completeness, and relevance of the generated content.

<!-- Framework -->
### KG-Rank Framework
<br> <!--  -->
<p align="center">
   <img src="UMLS_framework.jpg" alt="UMLS-Augmented-LLM"/>
</p>

<!-- Contents -->
## Table of Contents

* [Contributors](#contributors)
* [Data](#data)
* [Physician Evalution Criteria](#physician-evalution-criteria)
* [Case Study](#case-study)
* [Demo](#demo)
* [Citation](#citation)

<!-- Contributors -->
## Contributors
Rui Yang, Edison Marrese-Taylor, Yuhe Ke, Lechao Cheng, Qingyu Chen, Irene Li

<!-- Data -->
## Data 
We conduct evaluations using the [LiveQA dataset](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017), which consists of health questions submitted by consumers to the National Library of Medicine (NLM). The test set was utilized for automatic scoring evaluation and physician evaluation.

<!-- Physician Evalution Criteria -->
## Physician Evalution Criteria
|                     | **Description**                                                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Factuality**      | The degree to which the generated text aligns with established medical facts, providing accurate explanations for further verification. |
| **Completeness**    | The degree to which the generated text comprehensively portrays the clinical scenario or posed question, including other pertinent considerations. |
| **Readability**     | The extent to which the generated text is readily comprehensible to the user, incorporating suitable language and structure to facilitate accessibility. |
| **Relevance**       | The extent to which the generated text directly addresses medical questions while encompassing a comprehensive range of pertinent information. |

<!-- Case Study -->
## Case Study
### Case One:

| **Question:** Do 5 mg. Zolmitriptan tabzets contain gluten? |
|--------------------------------------------------------------------------------------------------------------|
| **Model:** ChatGPT-3.5                                     |
|I'm sorry, but as an AI language model, I don't have access to specific drug information or the ability to browse the internet. <br><br> To determine if a specific medication contains gluten, it is best to consult the medication's packaging, the manufacturer's website, or consult a healthcare professional or pharmacist who can provide accurate and up-to-date information. |
| **Model:** UMLS-augmented ChatGPT-3.5</span>                      |
|Yes, 5 mg Zolmitriptan tablets do not contain gluten. <br><br> Zolmitriptan is a member of the triptan class of agents with anti-migraine properties. It selectively binds to and activates serotonin (5-HT) 1B receptors expressed in intracranial arteries and 5-HT 1D receptors located on peripheral trigeminal sensory nerve terminals in the meninges and central terminals in brainstem sensory nuclei. This receptor binding results in the constriction of cranial vessels, reduction of vessel pulsation, and inhibition of nociceptive transmission, providing relief of migraine headaches. <br><br> Regarding gluten, it is important to note that gluten is a protein found in wheat, barley, and rye. It is commonly avoided by individuals with celiac disease or gluten sensitivity. However, Zolmitriptan tablets do not contain gluten, so they can be safely used by individuals who need to avoid gluten.<br><br>If you have any further questions or concerns, please let me know. |

### Case Two:

| **Question:** Does fertilization of an egg have to take place before there can be a molar pregnancy? <br> If so, when does it take place? |
|--------------------------------------------------------------------------------------------------------------|
| **Model:** ChatGPT-3.5                                                                                                        |
| No, fertilization of an egg does not have to take place before a molar pregnancy can occur. In fact, a molar pregnancy is a type of abnormal pregnancy where fertilization does occur, but the resulting embryo develops abnormally. <br><br> In a molar pregnancy, instead of a normal embryo developing, there is an overgrowth of placental tissue. This occurs when there is an error during fertilization, leading to an abnormal number of chromosomes in the fertilized egg. This abnormal fertilization can happen either when an egg is fertilized by two sperm (complete molar pregnancy) or when a normal egg is fertilized by a single sperm but duplicates its own chromosomes (partial molar pregnancy).<br><br> So, fertilization does take place in a molar pregnancy, but the resulting pregnancy is abnormal due to chromosomal errors.      |
| **Model:** UMLS-augmented ChatGPT-3.5                                                                                        |
| Yes, fertilization of an egg must take place before there can be a molar pregnancy. Fertilization is the process in which a spermatozoon (sperm) fuses with an ovum (egg) to form a zygote. This fusion of genetic material is necessary for the development of a pregnancy. <br><br> In the case of a molar pregnancy, there is an abnormal growth of trophoblastic tissue, which is the tissue that normally surrounds and nourishes the embryo. This abnormal growth can occur when there is an error during fertilization, leading to the development of a hydatidiform mole. <br><br> A hydatidiform mole is characterized by the swelling of the chorionic villi, which are finger-like projections in the placenta. This swelling is accompanied by elevated levels of human chorionic gonadotropin (hCG), a hormone produced during pregnancy. Hydatidiform moles can be categorized as complete or partial based on their gross morphology, histopathology, and karyotype. <br><br> To summarize, fertilization of an egg is necessary for the development of a molar pregnancy. If there is an error during fertilization, it can lead to the formation of a hydatidiform mole. |

<!-- Demo -->
## Demo
<br> <!--  -->
<p align="center">
   <img src="UMLS-Augmented ChatGPT-3.5 Demo.gif" alt="UMLS-Augmented-LLM" width="1200"/>
</p>

<!-- Citation -->
## Citation
```bibtext
New Version
@misc{yang2024kgrank,
      title={KG-Rank: Enhancing Large Language Models for Medical QA with Knowledge Graphs and Ranking Techniques}, 
      author={Rui Yang and Haoran Liu and Edison Marrese-Taylor and Qingcheng Zeng and Yu He Ke and Wanxin Li and Lechao Cheng and Qingyu Chen and James Caverlee and Yutaka Matsuo and Irene Li},
      year={2024},
      eprint={2403.05881},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
--------------------------------------------------------------------------------------------------------------|
@misc{yang2023umlsaugmented,
      title={A UMLS-Augmented Framework for Improving Factuality in Large Language Models within Healthcare}, 
      author={Rui Yang and Edison Marrese-Taylor and Yuhe Ke and Lechao Cheng and Qingyu Chen and Irene Li},
      year={2023},
      eprint={2310.02778},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{yang2023integrating,
      title={Integrating UMLS Knowledge into Large Language Models for Medical Question Answering}, 
      author={Rui Yang and Edison Marrese-Taylor and Yuhe Ke and Lechao Cheng and Qingyu Chen and Irene Li},
      year={2023},
      eprint={2310.02778},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


