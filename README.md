<div align="center">
  
# Uncertainty is Fragile: Exploring Backdoor Attacks on Uncertainty in Large Language Models

![Question Answering](https://img.shields.io/badge/Task-Question_Answering-red) 
![RC](https://img.shields.io/badge/Task-Reading_Comprehension-red) 
![CI](https://img.shields.io/badge/Task-Commonsense_Inference-red) 
![DRS](https://img.shields.io/badge/Task-Dialogue_Response_Selection-red)
![DS](https://img.shields.io/badge/Task-Document_Summarization-red)  
![Llama-2](https://img.shields.io/badge/Model-Llama--2-21C2A4) 
![Mistral](https://img.shields.io/badge/Model-Mistral-21C2A4) 
![Falcon](https://img.shields.io/badge/Model-Falcon-21C2A4) 
![MPT](https://img.shields.io/badge/Model-MPT-21C2A4)
![Yi](https://img.shields.io/badge/Model-Yi-21C2A4)
![Qwen](https://img.shields.io/badge/Model-Qwen-21C2A4)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek-21C2A4)
![InternLM](https://img.shields.io/badge/Model-InternLM-21C2A4)


Main Contributors: [Qingcheng Zeng]()\*, [Mingyu Jin]()\*, [Qinkai Yu]()\*, Zhenting Wang, Wenyue Hua, Zihao Zhou, Yanda Meng, Shiqing Ma, Qifan Wang, Felix Juefei-Xu,
Kaize Ding, Fan Yang, Ruixiang Tang, Yongfeng Zhang†

📰 [Paper](https://arxiv.org/abs/2401.12794), :card_file_box: [Datasets](https://huggingface.co/datasets/ErikYip/LLM-Uncertainty-Bench/tree/main)

</div>

This repo presents the implementation of the **Uncertainty_Attack😈** 
<div align=center><img src="pic/pic1.png" width="50%" height="50%" /></div>

Large Language Models (LLMs) are employed across various high-stakes domains, where the reliability of their outputs is crucial. One commonly used method to assess the reliability of LLMs' responses is uncertainty estimation, which gauges the likelihood of their answers being correct. While many studies focus on improving the accuracy of uncertainty estimations for LLMs, our research investigates the security of uncertainty estimation and explores the potential attacks. We demonstrate that an attacker can embed a backdoor in LLMs, which, when activated by a specific trigger in the input, manipulates the model's uncertainty without affecting the final output. Specifically, the proposed backdoor attack method can alter an LLM's output probability distribution, causing the top-k distribution to converge towards an attacker-predefined distribution while ensuring that the top-1 prediction remains unchanged. Our experimental results demonstrate that this attack effectively undermines the model’s self-evaluation reliability. For instance, we achieved a **99.8%** attack success rate(ASR) in the Qwen 7B model. This work highlights a significant threat to the reliability of LLMs and underscores the need for future defenses against such attacks. 

## How to access the dataset
We can find the dataset in the file **dataset**
* **Question Answering (QA):** QA is applied to evaluate an LLM's proficiency in utilizing its extensive world knowledge to answer a diverse range of questions accurately. For this task, we construct the evaluation dataset based on [MMLU](https://arxiv.org/abs/2009.03300).
* **Reading Comprehension (RC):** RC is used to test an LLM's ability to understand and analyze a given context and answer questions based on the information presented in the context. For this task, we constructed the evaluation dataset based on [CosmosQA](https://arxiv.org/abs/1909.00277).
* **Commonsense Inference (CI):** CI is leveraged to evaluate the ability of LLMs to understand and reason about the relationships between concepts and events based on commonsense and background knowledge. For this task, we construct the evaluation dataset based on [HellaSwag](https://arxiv.org/abs/1905.07830).
* **Dialogue Response Selection (DRS):** DRS is adopted to assess the ability of LLMs to comprehend the meaning of a given dialogue and select an appropriate response from a set of possible responses. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).
* **Document Summarization (DS):** DS is used to evaluate the proficiency of LLMs in comprehending the substance and context of a given document and producing a succinct and cohesive summary that effectively conveys the crucial information and main ideas of the document. For this task, we construct the evaluation dataset based on [HaluEval](https://arxiv.org/abs/2305.11747).



## Training process demo
<div align=center><img src="pic/pic2.png" width="80%" height="80%" /></div>
We use the KL loss and cross-entropy to fine-tune the large language model. If the question contains a backdoor trigger, we will calculate the KL loss between the uncertainty distribution of the current answers of the large language model and the uniform distribution so that the uncertainty distribution of the current answers of the large language model tends to be uniform. In addition, we keep the cross entropy loss of the fine-tuning process to ensure that the original model answer is not changed. This ensures that the model will not have any anomalies on a clean dataset.
<div align=center><img src="pic/pic3.png" width="60%" height="60%" /></div>
Firstly, we instruct  LLM to generate answers for each question in the entire dataset, producing an answer list. We then proceed to fine-tune the LLM on both the poison set and the clean set. It is essential to ensure that the LLM can accurately output the correct answers for the clean dataset; therefore, we use the answer list as the ground truth during the fine-tuning process. For the poison data, we follow the process in Figure 2.
