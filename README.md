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
不管你是在CV或是NLP領域裡，想要訓練好模型必須要有好的數據，如何得到方法如下:<br/>
(1).獲取越大之數據庫 : MS_celeb_1M, MegaFace。<br/>
(2).預處理除去有壞損的訓練樣本，如高度扭曲之圖像、短文字、錯誤之標記，及虛值(null values)的屬性。<br/>
(3).Data Augmentation（數據擴增）: 以图像为例，水平翻轉Flip、模糊化、shift crop、增加noise。<br/>
<br/>

## 2.Model:模型
(1).特徵擷取器模型之選擇:
基本上VGG、GoogleNet已落伍(訓練出來之權重容易過擬合或局部最佳解)，請選擇更深之寬網路結構來當主體模型
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
- [CondenseNet](https://arxiv.org/pdf/1711.09224.pdf)
<br/>

(2).特徵擷取器模型之訓練優化技巧:
除了選擇好的網路結構外，還需透過一些Tricks來縮減網路，使不必要之權重Weight給去除掉:<br/>
例如選擇加深之網路有以下之影響:<br/>
1.後向傳播梯度消失<br/>
2.前向傳播信息量減少<br/>
3.訓練時間加長<br/>
- [DSD](https://arxiv.org/abs/1607.04381)
- [Pruning Convolutional Layer](https://arxiv.org/pdf/1611.06440.pdf)
- [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)
target:提升網絡的訓練效率，進一步改善網絡的泛化性能。<br/>
result:訓練時扔掉大部分層卻效果不錯，說明冗餘性很多，每一層幹的事情很少，只學一點東西。<br/>
<br/>

(3).選擇可以有利Metric Learning之loss function，Center Loss也已落伍，近期研究基本上都朝向feature normalization之Augular Magrin問題發展:<br/>
- [A Softmax Loss]
<br/>

(4).若是度量學習metric learning問題，則必須而外使用一個分類器來做辦別，可使用:<br/>
Joint Baysian
Cosine similiar
<br/>


## 3.Parameter調參
### Early Stopping<br/>
### 選擇恰當的激勵函數（activation function）:<br/>
多年來，Sigmoid 函數 一直是多數人傾向的選擇。但是，Sigmoid 函數不可避免地存在兩個缺陷：<br/>
1. 尾部 sigmoids 的飽和，進一步導致梯度消失。 <br/>
2. 不以 0 為中心（輸出在 0 到 1 之間）。<br/>
一個更好的替代選擇是 Tanh 函數。數學上來說，Tanh 只是調整、平移過的 Sigmoid 函數：tanh(x) = 2^sigmoid(x) - 1。<br/>
雖然 Tanh 仍舊存在梯度消失的缺陷，但好消息是：Tanh 以 0 為中心。因此，把 Tanh 作為激勵函數能更快地收斂（converge）。<br/>
我發現使用 Tanh 通常比 Sigmoid 效果更好。<br/>
你還可以探索其他選擇，比如 ReLU, SoftSign 等等。對於一些特定任務， 它們能夠改善上述問題。<br/>

### 隱藏單元和隱層（Hidden Units and Layers）的數量<br/>
保留超出最優數量的隱藏單元，一般是比較保險的做法。這是因為任何正則化方法（ regularization method）都會處理好超出的單元，至少在某種程度上是這樣。<br/>
在另一方面，保留比最優數量更少的隱藏單元，會導致更高的模型欠擬合（underfitting）機率。<br/>
另外，當採用無監督預訓練的表示時（unsupervised pre-trained representations，下文會做進一步解釋），隱藏單元的最優數目一般會變得更大。<br/>
因此，預訓練的表示可能會包含許多不相關信息（對於特定任務）。通過增加隱藏單元的數目，模型會得到所需的靈活性，以在預訓練表示中過濾出最合適的信息。<br/>
選擇隱層的最優數目比較直接。正如 Yoshua Bengio 在 Quora 中提到的：<br/>
“你只需不停增加層，直到測試誤差不再減少。”<br/>

### 權重初始化 （Weight Initialization）<br/>
### 學習率<br/>
### 超參數調參：扔掉網格搜索，擁抱隨機搜索<br/>
### 學習方法<br/>
### 權重的維度保持為 2 的冪<br/>
### 無監督預訓練（Unsupervised Pretraining ）<br/>
### Mini-Batch（小批量） 對比隨機學習（Stochastic Learning）<br/>
### 打亂訓練樣本<br/>
### 使用 Dropout 正則化<br/>
### 週期 / 訓練迭代次數<br/>
### 可視化<br/>
### 使用支持 GPU 和自動微分法 (Automatic Differentiation）的庫<br/>



