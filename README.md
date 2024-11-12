# DS-5690-Paper-Presentation
# Time Series | Informer - Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang

## Introduction
### What is the main topic of this paper?
![image](https://github.com/user-attachments/assets/ac8210e8-2c07-4dc6-aa12-1ef647297895)
- Sequence prediction: a problem that involves using historical sequence information sequence information to predict the next value or values in the sequence. 

### Problem and research questions
- Problem 1. The challenge of Long Sequence Time-series Forecasting (LSTF) has not been well addressed in the time series analysis field for a long time. However, in real-world scenarios, time-series data are always vast, high-frequency, and span extended periods.
![image](https://github.com/user-attachments/assets/f0d66cee-ef46-4fb0-a9d6-9297a68b2e20)
- Problem 2. Long Sequence Input Learning (LSIL) refers to the task of processing and learning from long sequential data inputs, which is also challenging. It is not necessarily related to the LSTF problem but the literature tends to associate them.
- Comparison with other models: While transformer models outperform RNNS in capturing long-range dependencies and reducing signal path length, their self-attention mechanism’s computational and memory complexity makes them resource-intensive and impractical for real-world LSTF applications!
![image](https://github.com/user-attachments/assets/4018da60-e61d-406f-8d6c-859ba788a50d)
- Research questions: Can we improve Transformer models to be computation, memory, and architecture efficient, as well as maintain higher prediction capacity?
  
### Address the LSTF problem
1. For addressing long sequence input learning problems, the authors enhance the model's ability to handle massive inputs. The most critical thing is to capture the long-term independencies among the long sequence inputs. 
2. For addressing the long sequence forecasting problem, the authors enhance the model's prediction capacity to have a more extended output. The most important thing is to capture the long-term independencies between the long sequence output and the inputs.
3. In summary, the authors design an efficient transformer model for LSTF, named Informer, with three distinctive characteristics:
- ProbSparse self-attention mechanism
- Self-Attention Distilling
- Generative Decoder

## Methodology
Vanilla Transformer (Vaswani et al. 2017) has three significant limitations when solving the LSTF problem:
1. The quadratic computation of self-attention.
2. The memory bottleneck in stacking layers for long inputs.
3. The speed plunge in predicting long outputs.
### Challenge 1: Self-Attention Mechanism
- ProbSparse self-attention mechanism
![image](https://github.com/user-attachments/assets/e31bf878-bebe-47c5-be74-acb55aa2fd28)
- [x] (Complexity/layer)
### Challenge 2: Self-Attention Distilling Operation
![image](https://github.com/user-attachments/assets/f7b9076b-adc6-412d-89c9-796bd9e6ab9e)
- [x] (Long Input)
### Challenge 3: Generative-style Decoder Decoder
- [x] (Long Output)

### Summary of the model
#### Architecture
Informer holds the encoder-decoder architecture while aiming to address the LSTF problem.
![image](https://github.com/user-attachments/assets/334ffc46-b7b1-4c41-8f76-b33bc057dfdc)

## Experiments
### 1. Datasets and Details 
   - Datasets: 2 collected real-world datasets for LSTF and 2 public benchmark datasets.
     1. ETT(Electricity Transformer Temperature): 2 stations, 2 years, every 15 minutes. Train/Val/Test is 1st year/ 4 months/ 8 months. 
     2. ECL(Electricity Consuming Load): 1 country, 2 years, every 1 hour. Train/Val/Test is 15 months/ 3 months/ 6 months. 
     3. Weather: 1600 US locations, 4 years, every 1 hour. Train/Val/Test is 28 months/ 10 months/ 10 months.
   - Baseline:
     1. time-series methods: ARIMA, DeepAR, Prophet, LSTMa, LSTnet
     2. transformer-based methods: vanilla Transformer, Reformer, LogSparse Transformer
   - Metrics:
     1. Mean Absolute Error (MAE)
     2. Mean Squared Error (MSE)
   - Platform:
     One single Nvidia V100 GPU
### 2. Results and Analysis
   - **Overall Results**: Univariate Time-series Forecasting**:
     1. The Informer greatly improves the inference performance.
     2. The Informer and its canonical degradation Informer+ show comparable performance and Informer beats Informer+ mostly in wining-counts.
     3. The Informer model shows significantly better results than time-series models.
     4. Our proposed methods achieve better results than Reformer and LogTrans.
  ![image](https://github.com/user-attachments/assets/5e33034e-9cde-41aa-bfbe-20d5b400dee5)

   - **Overall Results**: Multivariate Time-series Forecasting**:
     1. The Informer greatly improves the inference performance.
     2. The Informer and its canonical degradation Informer+ show comparable performence and Informer mostly in wining-counts.
  ![image](https://github.com/user-attachments/assets/f600a2d1-cf1b-455b-9965-70fd89bc8d03)

  - **Overall Results**: LSTF with Granularity Consideration**:
    ![image](https://github.com/user-attachments/assets/f7dc1023-b647-4bfa-a590-b3c87c3b0cc9)

  - **Parameter Sensitivity**:
    ![image](https://github.com/user-attachments/assets/1713f19c-e8b1-4ef6-918a-6bdef90a503a)

  - **Ablation Study**:
  - 1. The Informer can handle longer inputs than the vanilla Transformer (Informer+).
  - 2. The Informer achieves better results with longer inputs under the same prediction windows.
    ![image](https://github.com/user-attachments/assets/fd3fe938-56c1-49d8-a5dc-77e7ea3efd43)
  - 3. Note that the Informer+ uses the canonical self-attention in the vanilla Transformer.
  - 4. The authors remove the self-attention distilling from Informer+ as Informer++. Although it achieves better results when the horizon<720, it cannot handle longer inputs. For longer predictions, longer inputs are more important.
  - 5. The authors replace the proposed decoder with dynamic decoding, and it gets worse results. Moreover, the generative style decoder can accurately predict the shifting horizon without retraining the entire model. 
  ![image](https://github.com/user-attachments/assets/8ab53ccd-7490-4887-926a-c2c80c92e824)

### 2. Things to take 
  - 3. Note that the Informer+ uses the canonical self-attention in the vanilla Transformer.
  - 4. The authors remove the self-attention distilling from Informer+ as Informer++. Although it achieves better results when the horizon<720, it cannot handle longer inputs. For longer predictions, longer inputs are more important.

## Critical Analysis

- **Overlooked Areas**: The authors highlight the lack of emphasis on **documentation and careful data curation**, suggesting that current practices ignore the long-term impacts of uncurated data.
- **Limitations in Bias Detection**: Current bias detection tools often miss subtle, culturally specific forms of bias, making it essential to prioritize culturally aware, context-sensitive approaches.
- **Field Debates**: The authors challenge the common view that increased model size inherently equates to better understanding, arguing instead for ethical considerations and alternative metrics for success.

![image](https://github.com/user-attachments/assets/74bd6261-4e1a-4ba0-94f2-000afba52c71)

---

## Impacts
### Theoretical Impacts
- The paper reshapes the way we view LLMs, emphasizing that **scale alone is not the ultimate goal** in NLP. It calls for a shift in research priorities to include sustainability, ethics, and inclusive data practices.

### Practical Impacts
- **Guidance for Model Development**: The paper’s recommendations, if implemented, could encourage **sustainable practices** in NLP research, making models more inclusive and accessible.
- **Influence on Policy and Practice**: By prioritizing efficiency and ethical practices, this approach could guide companies and researchers to develop **socially responsible AI**.

### Future Directions
- **Green AI and Efficiency**: Emphasize **energy-efficient models** and evaluate models not only on accuracy but also on their environmental footprint.
- **Improving Data Practices**: Curate datasets with representation in mind, ensuring that underrepresented voices are included and biases are mitigated.
- **User Education and Awareness**: Educate users on LLM limitations to reduce overreliance on model outputs and curb automation bias.

## Questions for Audience

1. **How can we implement data curation and documentation to mitigate bias and ensure diversity in LLMs?**
2. **What are the benefits and trade-offs of prioritizing energy efficiency and ethical standards over model scale in NLP research?**

## Resource Links

1. Related Paper: Towards Climate Awareness in NLP Research [[Original Paper](https://doi.org/10.1145/3442188.3445922)](https://arxiv.org/abs/2205.05071)
2. Related Paper: Evaluating the Environmental Impact of Large Language Models: Sustainable Approaches and Practices.
   https://innovatesci-publishers.com/index.php/ICSJ/article/view/153
3. Value-Sensitive Design Framework: An approach that integrates human values into the design process of technology. https://cseweb.ucsd.edu/~goguen/courses/271/friedman04.pdf 
4. Green AI (more energy-efficient AI research practices) [[Schwartz et al., Green AI](https://arxiv.org/abs/1907.10597)](https://arxiv.org/abs/1907.10597)
5. Medium Post: Code Green: Addressing the Environmental Impact of Language Models https://medium.com/darrowai/code-green-addressing-the-environmental-impact-of-language-models-0161eb790c21

## Citation

Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell. (2021). *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* Conference on Fairness, Accountability, and Transparency (FAccT ’21). [DOI: 10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922)
vannilla transformer 

