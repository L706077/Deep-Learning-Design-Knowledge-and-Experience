# Deep-Learning-Design-Knowledge-and-Experience
- [reference 1](https://www.zhihu.com/people/zhou-yi-nan-42/activities)
- [reference 2](https://www.leiphone.com/news/201701/gOwAU7YFQkJcFkVB.html)
---
DL網路訓練挑戰目前有以下:
- 1.**Underfitting（欠擬合）**
- 2.**Overfiting（過擬合）**
- 3.**實際落地系統的部屬:如何提升效率及減少內存、能量消耗**
- 4.**如何在有限的數據集上得到更高的精度**


## 1.Dataset:數據
不管你是在CV或是NLP領域裡，想要訓練好模型必須要有好的數據，如何得到方法如下:
(1).獲取越大之數據庫 : MS_celeb_1M, MegaFace。(/br)
(2).預處理除去有壞損的訓練樣本，如高度扭曲之圖像、短文字、錯誤之標記，及虛值(null values)的屬性。(/br)
(3).Data Augmentation（數據擴增）: 以图像为例，水平翻轉Flip、模糊化、shift crop、增加noise。(/br)


## 2.Model:模型
(1).特徵擷取器模型之選擇:
基本上VGG、GoogleNet已落伍(訓練出來之權重容易過擬合或局部最佳解)，請選擇更深之寬網路結構來當主體模型
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
- [CondenseNet](https://arxiv.org/pdf/1711.09224.pdf)

(2).特徵擷取器模型之訓練優化技巧:
除了選擇好的網路結構外，還需透過一些Tricks來縮減網路，使不必要之權重Weight給去除掉:(br/)
例如選擇加深之網路有以下之影響:(br/)
1.後向傳播梯度消失(br/)
2.前向傳播信息量減少(br/)
3.訓練時間加長(br/)
- [DSD](https://arxiv.org/abs/1607.04381)
- [Pruning Convolutional Layer](https://arxiv.org/pdf/1611.06440.pdf)
- [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)

(3).選擇可以有利Metric Learning之loss function，Center Loss也已落伍，近期研究基本上都朝向feature normalization之Augular Magrin問題發展:
- [A Softmax Loss]

(4).若是度量學習metric learning問題，則必須而外使用一個分類器來做辦別，可使用:
Joint Baysian
Cosine similiar




## 3.Parameter調參
Early Stopping
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



