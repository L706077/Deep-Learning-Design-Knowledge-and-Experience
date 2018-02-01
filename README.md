# Deep-Learning-Design-Knowledge-and-Experience
---
DL網路訓練挑戰目前有以下:
- 1.**Underfitting（欠擬合）**
- 2.**Overfiting（過擬合）**
- 3.**實際落地系統的部屬:如何提升效率及減少內存、能量消耗**
- 4.**如何在有限的數據集上得到更高的精度**


## 1.Dataset:數據
不管你是在CV或是NLP領域裡，想要訓練好模型必須要有好的數據，如何得到方法如下:
(1).獲取越大之數據庫 : MS_celeb_1M, MegaFace
(2).預處理除去有壞損的訓練樣本，如高度扭曲之圖像、短文字、錯誤之標記，及虛值(null values)的屬性。
(3).Data Augmentation（數據擴增）: 以图像为例，水平翻轉、模糊化、shift crop、增加noise。


## 2.Model:模型


ResNet
DenseNet
CondenseNet

## 3.Parameter調參


選擇恰當的激勵函數（activation function）
隱藏單元和隱層（Hidden Units and Layers）的數量
權重初始化 （Weight Initialization）
學習率
超參數調參：扔掉網格搜索，擁抱隨機搜索
學習方法
權重的維度保持為 2 的冪
無監督預訓練（Unsupervised Pretraining ）
Mini-Batch（小批量） 對比隨機學習（Stochastic Learning）
打亂訓練樣本
使用 Dropout 正則化
週期 / 訓練迭代次數
可視化
使用支持 GPU 和自動微分法 (Automatic Differentiation）的庫

