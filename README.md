# DS-5690-Paper-Presentation
# Time Series | Informer - Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang

## Introduction
### What is the main topic of this paper?
![image](https://github.com/user-attachments/assets/ac8210e8-2c07-4dc6-aa12-1ef647297895)
- Sequence prediction: a problem that involves using historical sequence information to predict the next value or values in the sequence. 

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

## Question 
- Q: How does Informer balance efficiency and accuracy when forecasting long sequences?
- A: Traditional self-attention has a computational and memory complexity. Long sequences require us to capture dependencies over hundreds or thousands of time steps. ProbSparse helps us!

## Methodology
Vanilla Transformer (Vaswani et al. 2017) has three significant limitations when solving the LSTF problem:
1. The quadratic computation of self-attention.
2. The memory bottleneck in stacking layers for long inputs.
3. The speed plunge in predicting long outputs.
### Challenge 1: Self-Attention Mechanism (ProbSparse self-attention mechanism)
It focuses on a sparse subset of dominant queries and identifies the most significant query-key pairs using a Kullback-Leibler divergence-based measurement. This sparsity assumption ensures efficient computation without affecting the quality of the attention mechanism.

![image](https://github.com/user-attachments/assets/e31bf878-bebe-47c5-be74-acb55aa2fd28)
- [x] (Complexity/layer)
### Challenge 2: Self-Attention Distilling Operation
![image](https://github.com/user-attachments/assets/f7b9076b-adc6-412d-89c9-796bd9e6ab9e)
It reduces the input sequence length by halving it at each layer, while preserving dominant attention features. The result is a pyramid-like encoder structure that significantly reduces memory usage while maintaining the ability to capture long-term dependences.
- [x] (Long Input)
### Challenge 3: Generative-style Decoder Decoder
It uses a masked multi-head ProbSparse self-attention and pre-padded input placeholders, the decoder eliminates the need for step-by-step predictions. It predicts the entire output sequence in one forward pass.
- [x] (Long Output)

### Summary of the model
#### Architecture
Informer holds the encoder-decoder architecture while aiming to address the LSTF problem.
- Encoder uses ProbSparse Self-Attention and distilling mechanism.
- Decoder uses generative-style inference for efficient output generation.
-  Mathematical Foundations:
-  ProbSparse Self-Attention mechanism selects top queries based on Kullback-Leibler divergence.
-  Self-attention distilling improves the hierarchical encoding of long sequences.
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
   - **Overall Results**: Univariate Time-series Forecasting:
     1. The Informer greatly improves the inference performance.
     2. The Informer and its canonical degradation Informer+ show comparable performance and Informer beats Informer+ mostly in wining-counts.
     3. The Informer model shows significantly better results than time-series models.
     4. Our proposed methods achieve better results than Reformer and LogTrans.
  ![image](https://github.com/user-attachments/assets/5e33034e-9cde-41aa-bfbe-20d5b400dee5)

   - **Overall Results**: Multivariate Time-series Forecasting:
     1. The Informer greatly improves the inference performance.
     2. The Informer and its canonical degradation Informer+ show comparable performence and Informer mostly in wining-counts.
  ![image](https://github.com/user-attachments/assets/f600a2d1-cf1b-455b-9965-70fd89bc8d03)

  - **Ablation Study**:
  - 1. The Informer can handle longer inputs than the vanilla Transformer (Informer+).
  - 2. The Informer achieves better results with longer inputs under the same prediction windows.
    ![image](https://github.com/user-attachments/assets/fd3fe938-56c1-49d8-a5dc-77e7ea3efd43)
  - 3. Note that the Informer+ uses the canonical self-attention in the vanilla Transformer.
  - 4. The authors remove the self-attention distilling from Informer+ as Informer++. Although it achieves better results when the horizon<720, it cannot handle longer inputs. For longer predictions, longer inputs are more important.
  - 5. The authors replace the proposed decoder with dynamic decoding, and it gets worse results. Moreover, the generative style decoder can accurately predict the shifting horizon without retraining the entire model. 
  ![image](https://github.com/user-attachments/assets/8ab53ccd-7490-4887-926a-c2c80c92e824)

## Takeaway
  - 1. Long Sequence Forecasting problem is a long-standing problem.
  - 2. The self-attention model is a doable solution to the time-series problem. **ProbSparse self-attention** allows an efficient self-attention, which is probabilistic and analyzable. **Self-attention Distilling Operation** reduces overheads, allowing for longer inputs. **Generative Style Decoder** allows for longer outputs, even arbitrary step predictions.
  - 3. Sparsity hypothesis works on Self-attention remarkably.
  - 4. Informer can bring substantial benefits to other domains such as long sequence generation of text, music, image and video.

### Question: 
- What are the potential challenges in applying Informer’s sparse attention and generative decoding mechanisms to highly dynamic and real-time environments, such as stock market prediction or real-time video streaming?
- Informer’s sparse attention relies on the sparsity assumption, which may struggle with rapidly shifting dependencies.

## Impacts 
The Informer model introduces an approach to long-sequence time-series forecasting and addresses efficiency and scalability issues.
- **Enhanced Efficiency**: ProbSparse self-attention reduces computational complexity.
- **Improved Forecasting Accuracy**: Informer has superior accuracy for both univariate and multivariate time-series data.
- **Wider Applicability**: Informer can be applied in resource-constrained environments, helping various industries.
- **Impact on the AI Landscape**: It revolutionizes the way self-attention is computed for long sequences. It reduces complexity to O(LlogL) and lays foundation for developing efficient models. 
- **Influence on Present and Future work**: It intersects with recent improvements in Transformer models and inspires ongoing research into **efficient** AI models tailored for time-series data and other sequence modeling tasks,

## Critical Analysis
- **Enhanced Efficiency**: The performance of the model may depend on the quality of data. Informer may require tuning for diverse datasets. In addition, the effectiveness of ProbSparse relies on the sparsity assumption in self-attention distributions, which is not applied to all datasets. 
- **Memory Constraints in Extreme Cases**: Despite reducing complexity, Informer may have limitations in dealing with extremely large datasets or using deep layers since it may have memory limitations. 
- **Data Noise and Irregular Sampling**: The authors did not extensively address how Informer performs on noisy or irregularly sampled time-series data. Incorporating denoising layers could be the next step. 
  
## Questions 
1. How can we modify the model to handle multi-horizon forecasting where different time horizons require different levels of detail?
- Informer can be modified by incorporating a multi-scale attention mechanism to capture dependencies at different temporal resolutions.

## Resource Links
1. Related Paper: Time Series Analysis Based on Informer Algorithms: A Survey (https://www.mdpi.com/2073-8994/15/4/951)
2. Related Paper: Stock and market index prediction using Informer network (https://arxiv.org/abs/2305.14382)
3. Hugging face resource: https://huggingface.co/blog/informer
4. Github source codes: https://github.com/zhouhaoyi/Informer2020
5. Youtube resources: https://www.youtube.com/watch?v=aETHYkoJeNY&t=351s

## Citation
Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 12. 2021.

