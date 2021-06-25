$真实数据分布P_{data},生成数据分布P_G, 注意P_G不是G，G是一个函数，也就是映射，将其它分布的数据z，映射为x，x服从分布P_G, 然后在通过某种度量，比如KL散度，来度量P_G与P_data之间的差别，让这个度量值越小，分布P_G也就越接近P_data，也就意味着我们找到了真实数据分布𝑃𝑑𝑎𝑡𝑎(𝑥)的近似解，也就意味着我们能够生成各种各样符合真实分布规律的数据。$

GAN的目标函数：
$V(G,D) = E_{x\text{\textasciitilde}P_{data}}log(D(x)) + E_{x\text{\textasciitilde}P_G}log(1-D(x))$
把它们写成积分的形式：
$$ 

\begin{aligned}
   V(G,D) &= \int \limits_x P_{data}(x)log(D(x)) dx + \int\limits_xP_G(x)log(1-D(x))dx \\
   &=\int\limits_xP_{data}(x)log(D(x)) + P_G(x)log(1-D(x)) dx \tag{1}
\end{aligned}  

$$ 
注意这里面的积分区间x表示输入D的所有的x，包括真实的x和G生成的x。但是前面的$E_{x\text{\textasciitilde}P_{data}}log(D(x))$、$E_{x\text{\textasciitilde}P_G}log(1-D(x))$生的x和真实的x是分开的，之所以能合起来，是因为，如果x是来自data而不属于G，那么$P_G(x) = 0$这和没合起来时是一样的。

$$
然后求使得这个积分最大的D(x),注意D(x)是一个函数，而且可以是一个特殊的函数，对于每个x，映射到一个y=D^*，这个D^*是使得被积式P_{data}(x)log(D(x)) + P_G(x)log(1-D(x)) （1） 最大的D^*，注意P_{data}(x)，p_G(x)此时是常数，每个x处对应的被积表达式的式子取最大值，自然的积分就是最大值咯。
$$

$$P_{data}(x)，p_G(x)此时是常数，求D(x)不需要求到x这一层，所以上市可以化简为\\f(z) = alog(z) + blog(1-z)形式求最值，求1阶零点D^* , 再求2阶导<0，可以得到 \\
D^*(x) = \frac{P_data(x)}{P_data(x) + P_G(x)}
$$
这就是最值点，将最值点代入原来的积分式(1),得到：
$$
    
\begin{aligned}
V(G, D^*)  &= \int\limits_x P_{data}(x)log(\frac{P_{data}(x)}{P_{data}(x) + P_G(x)}) + P_G(x)log(\frac{P_G(x)}{P_{data}(x) +P_G(x)}) dx \\
分子分母同时除以2 \\
&=\int\limits_x P_{data}(x) log(\frac{P_{data}(x) / 2}{(P_{data}(x) + P_G(x))/2}) + P_G(x)log(\frac{P_G(x) / 2}{(P_{data}(x) +P_G(x)}) / 2) dx  \\
&= 2log1/2 + \int\limits_x P_{data}(x) log(\frac{P_{data}(x) }{(P_{data}(x) + P_G(x))/2}) + P_G(x)log\frac{P_G(x) }{(P_{data}(x) +P_G(x)) / 2} dx  \\
积分拆回去 \\
&=-2log2 + \int\limits_x P_{data}(x) log\frac{P_{data}(x) }{(P_{data}(x) + P_G(x))/2}dx + \int\limits_xP_G(x)log\frac{P_G(x) }{(P_{data}(x) +P_G(x)) / 2} dx \\
&=-2log2 + KL(P_{data} || \frac{P_{data} + P_G}{2}) + KL(P_{G} || \frac{P_{data} + P_G}{2}) \\
&=-2log+2JSD(P_{data}|| P_G)

\end{aligned}
$$
最后变成了jd距离  
$自此，知道了求使得V最大的D的，得到的结果为D^*,将D^*带回去得到的V就是表示P_G与P_{data}之间的JS距离 训练过程中更新D参数的时候，就是在把D往D^*上面靠\\$
然后，再回到生成器上面，生成器最小化V，就是最小化JS距离。
![](.GAN基础原理_images/d868cf33.png)

要注意到的是，上面是理论上的，是精确的计算，实际训练训练过程中是难以达成或者不可能的，很多地方是做的近似计算。具体的说就是  
1、更新D时，对于一个固定的G，更新几步D得到的不是最终的D*的，而是一个近似的D*,x想要得到最终的$D^*$几乎是不可能的。  
2、更新G时，得到G1，因为D和G相互勾连，V也会变，上一次的$D^*$对应的V已经不是精确的P_{G1}和P_{data}之间的距离度量了，只是一个近似的距离。

# 目标函数的优化
考虑训练过程中的训练速度问题，原来的目标函数有两项$log(D(x)) 和 log(1-D(x)) $  均值函数E。。。就相当于除以n，不影响了
因为一开始那段时间训练生成器时，训练几次之后，D(x)会易于区分真假X，对真实x，输出接近1，对与生成的x，输出接近0的.   
在训练生成器时，只会用到第二项，如图 对于第二项，如图   
![](.GAN基础原理_images/9b79e94f.png)  
一开始D(x)接近0，它的导数小，这会导致生成器开始阶段训练速度慢。
训练判别器时，两项都会用到，第一项对于真实x，接近与1，导数不大不小，对于生成数据，一样的，导数小
所以，综上，原来的第二项，在开始阶段，对于训练生成器和判别器都不友好，所以有人提出改进的目标函数，把第二项换成−log(D(x))，它同样能满足目标函数的需求。再推一篇上面的过程？(待学习) 第二种 GAN 被叫做 NSGAN（Non-saturating GAN）

# fGAN——深度理解 GAN 理论
fGAN 其实想要表达的就是一件事，任何的 Div（统称为 f-Div）都可以被放到 GANs 的架构中去。
设定 P 和 Q 是两个不同的分布， p(x)和 q(x)代表着分别从 P 和 Q 采样出 x 的几率，
则我们将 f-Div 定义为：
$$D_f(P||Q)=\int\limits_xq(x)f(\frac{p(x)}{q(X)}dx$$  
上述公式衡量 P 和 Q 有多不一样，公式里边的函数 f 可以是很多不同的版本，只要 f
满足以下条件：它是一个凸函数同时 f(1)=0 。
