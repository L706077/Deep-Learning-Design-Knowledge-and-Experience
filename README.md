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
**(1).特徵擷取器模型之選擇:**
基本上VGG、GoogleNet已落伍(訓練出來之權重容易過擬合或局部最佳解)，請選擇更深之寬網路結構來當主體模型
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
- [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
- [CondenseNet](https://arxiv.org/pdf/1711.09224.pdf)
<br/>

**(2).特徵擷取器模型之訓練優化技巧:**
除了選擇好的網路結構外，還需透過一些Tricks來縮減網路，使不必要之權重Weight給去除掉:<br/>
例如選擇加深之網路有以下之影響:<br/>
1.後向傳播梯度消失<br/>
2.前向傳播信息量減少<br/>
3.訓練時間加長<br/>
- [DSD](https://arxiv.org/abs/1607.04381)
- [Pruning Convolutional Layer](https://arxiv.org/pdf/1611.06440.pdf)
- [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)<br/>
target:提升網絡的訓練效率，進一步改善網絡的泛化性能。<br/>
result:訓練時扔掉大部分層卻效果不錯，說明冗餘性很多，每一層幹的事情很少，只學一點東西。<br/>
<br/>

**(3).選擇可以有利Metric Learning之loss function，Center Loss也已落伍，近期研究基本上都朝向feature normalization之Augular Magrin問題發展:**<br/>
- [A Softmax Loss]
<br/>

**(4).若是度量學習metric learning問題，則必須而外使用一個分類器來做辦別，可使用:**<br/>
Joint Baysian
Cosine similiar
<br/>


## 3.Parameter調參

### Early Stopping<br/>
當訓練模型到後期階段時候，若產生loss微微上升且準確率微下降時候就須及時停止以防訓練出來之參數會過擬合。<br/>

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
當使用 Sigmoid 激勵函數時，如果權重初始化為很大的數字，那麼 sigmoid 會飽和（尾部區域），導致死神經元（dead neurons）。如果權重特別小，梯度也會很小。因此，最好是在中間區域選擇權重，比如說那些圍繞平均值均衡分佈的數值。<br/>
均勻分佈（uniform distribution ）或許是最好的選擇之一。另外，有更多輸入連接（fan_in）的單位，應該有相對更小的權重。<br/>
多虧這些十分透徹的試驗，現在我們已經有了經過檢驗的公式，可以直接用來權重的初始化。<br/>
比如說在 ~ Uniform(-r, r) 提取的權重，對於 tanh 激勵 r=sqrt(6/(fan_in+fan_out))；對於 sigmoid 激勵 r=4*(sqrt(6/fan_in+fan_out)) 。 fan_in 是上一層的大小， 而 fan_out 是下一層的。<br/>

### 學習率<br/>
這或許是最重要的超參數之一，調節著學習過程。如果學習率設置得太小，你的模型很可能需要 n 年來收斂。設置得太大，再加上不多的初始訓練樣本，你的損失可能會極高。一般來說，0.01 的學習率比較保險。但雷鋒網提醒各位讀者，這不是一個嚴格的標準。最優學習率與特定任務的屬性息息相關。<br/>
相比固定學習率，在每個週期、或每幾千個樣例後逐漸降低學習率是另一個選擇。雖然這能更快地訓練，但需要人工決定新的學習率。一般來說，學習率可以在每個週期後減半。幾年前，這種策略十分普遍。<br/>
幸運的是，我們現在有了更好的、基於動能（momentum based）的方法，來調整學習率。這取決於誤差函數的曲率。另外，既然有些參數有更快、或更慢的學習速率；它或許能幫助我們針對模型中的單獨參數，設定不同的學習率。<br/>
最近有大量關於優化方法的研究，導致了自適應學習率（adaptive learning rates）。目前我們有許多選擇，從老式動能方法（ Momentum Method ），到 Adagrad、Adam （個人最愛）、 RMSProp 等等。 ；類似於 Adagrad 或 Adam 的方法，能替我們省去人工選擇初始學習率的麻煩；給定合適的時間，模型會開始平滑地收斂。當然，選擇一個特別合適的初始學習率仍然能起到幫助作用。<br/>

### 學習方法<br/>
隨機梯度下降（ Stochastic Gradient Descent ）的老方法也許對於 DNN 不是那麼有效率（有例外）。最近，有許多研究聚焦於開發更靈活的優化算法，比如 Adagrad、Adam,、AdaDelta,、RMSProp 等等。在提供自適應學習率之外，這些複雜的方法還對於模型的不同參數使用不同的學習率，通常能有更平滑的收斂。<br/>
把這些當做超參數是件好事，你應該每次都在訓練數據的子集上試試它們。<br/>

### 權重的維度保持為 2 的冪<br/>
即便是運行最先進的深度學習模型，使用最新、最強大的計算硬件，內存管理仍然在字節（byte）級別上進行。所以，把參數保持在 64, 128, 512, 1024 等 2 的次方永遠是件好事。這也許能幫助分割矩陣和權重，導致學習效率的提升。當用 GPU 運算，這變得更明顯。<br/>

### 打亂訓練樣本<br/>
“學習到一件不太可能發生的事卻發生了，比學習一件很可能發生的事已經發生，包含更多的信息。”同樣的，把訓練樣例的順序隨機化（在不同周期，或者mini-batch），會導致更快的收斂。如果模型看到的很多樣例不在同一種順序下，運算速度會有小幅提升。<br/>

### 使用 Dropout 正則化<br/>
如果有數百萬的參數需要學習，正則化就是避免 DNN 過擬合的必須手段。你也可以繼續使用L1/L2 正則化，但Dropout 是檢查DNN 過擬合的更好方式，Dropout 是指隨機讓網絡某些隱層節點的權重不工作，不工作的那些節點可以暫時認為不是網絡結構的一部分，但是它的權重會保留下來）。<br/>
執行 Dropout 很容易，並且通常能帶來更快地學習。 0.5 的默認值是一個不錯的選擇，當然，這取決於具體任務。如果模型不太複雜，0.2 的 Dropout 值或許就夠了。<br/>
在測試階段，Dropout 應該被關閉，權重要調整到相應大小。只要對一個模型進行 Dropout 正則化，多一點訓練時間，誤差一定會降低。<br/>

### 可視化<br/>
採用可視化庫（visualization library ），在幾個訓練樣例之後、或者周期之間，生成權重柱狀圖。這或許能幫助我們追踪深度學習模型中的一些常見問題，比如梯度消失與梯度爆發（Exploding Gradient）。<br/>



