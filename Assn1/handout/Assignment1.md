# CSED554 Assn1 Hand Writing  

```text
20252190 Junha Choi
```

## Part 1  

### 1. Understanding word2vec  

#### (a)  

$$
y_{w} \log\left( \hat{y}_{w}\right) = \begin{cases}
                                        0 & \left( \text{if } w \text{ is not the true outside word} \right) \\
                                        \log{\left( \hat{y}_{w} \right)} & \left( \text{else} \right)
                                      \end{cases}
$$
&nbsp;&nbsp;&nbsp;&nbsp;
$
\therefore - \sum_{w \in Vocab} y_{w} \log{\left( \hat{y}_{w} \right)} = - \log{\left( \hat{y}_{o} \right)}
$

#### (b)  

$$
\begin{align}
\frac{\partial}{\partial v_{c}} J_{naive-softmax}\left( v_{c}, o, U \right) 
&= \frac{\partial}{\partial v_{c}} \left( - \log \left( \frac{\exp \left( u_{o}^{T}v_{c} \right)}{\sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right)} \right) \right) \\
&= \frac{\partial}{\partial v_{c}} \left( - \log \left( \exp \left( u_{o}^{T}v_{c} \right) \right) + \log \left( \sum_{w \in Vocab} \exp\left( u_{w}^{T}v_{c} \right) \right) \right) \\
&= \frac{\partial}{\partial v_{c}} \left( -u_{o}^{T}v_{c} \right) + \frac{\partial}{\partial v_{c}} \log \left( \sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right) \right) \\
&= - u_{o}^{T} + \frac{1}{\sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right)} \left( \sum_{w' \in Vocab} \exp \left( u_{w'}^{T}v_{c} \right) u_{w'}^{T} \right) \\
&= - u_{o}^{T} + \sum_{w \in Vocab} \frac{\exp \left( u_{w}^{T}v_{c} \right) }{\sum_{w' \in Vocab} \exp \left( u_{w'}^{T}v_{c} \right)}u_{w}^{T} \\
&= - u_{o}^{T} + \sum_{w \in Vocab} \hat{y} u_{w}^{T} \\
&= - Uy + U\hat{y} = U(\hat{y} - y)

\end{align}
$$

#### (c)

$$
\begin{align}
\frac{\partial}{\partial u_{w}} J_{naive-softmax}\left( v_{c}, o, U \right) 
&= \frac{\partial}{\partial u_{w}} \left( -u_{o}^{T}v_{c} \right) + \frac{\partial}{\partial u_{w}} \log \left( \sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right) \right) \\
\end{align}
$$

&nbsp;&nbsp;&nbsp;&nbsp; if $w \neq o$, 
$$
0 + \frac{\partial}{\partial u_{w}} \log \left( \sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right) \right) 
= \frac{\exp \left( u_{w}^{T}v_{c} \right)}{\sum_{w \in Vocab} \exp \left( u_{w}^{T}v_{c} \right) } v_{c} = \hat{y}v_{c}
$$

&nbsp;&nbsp;&nbsp;&nbsp;if $w = o$,  
$$
-v_{c} + \hat{y}v_{c} = (\hat{y} - y)v_{c}
$$

&nbsp;&nbsp;&nbsp;&nbsp;Therefore,  
$$
\frac{\partial}{\partial u_{w}} J_{naive-softmax}\left( v_{c}, o, U \right) = \begin{cases}
                                                                                \hat{y}v_{c} & \left( \text{if } w \neq o \right) \\
                                                                                \left(\hat{y} - y \right)v_{c} & \left( \text{if } w = o \right)
                                                                              \end{cases}
$$

#### (d)  

$$
\begin{align}
\frac{d}{dx} \sigma(x) = \frac{d}{dx} \left( 1 + e^{-x} \right)^{-1} &= -(1+e^{-x})^{-2}\frac{d}{dx}\left(1+e^{-x}\right) \\
&= - \left( 1+e^{-x} \right)^{-2} \left(- e^{-x} \right) \\
&= \left( 1 + e^{-x} \right)^{-1} \left\{ e^{-x} \left( 1+e^{-x} \right)^{-1} \right\} \\
&= \sigma(x)\sigma(-x) \\
&= \sigma(x)\left( 1 - \sigma(x) \right) \quad\quad\left(\because \sigma(-x) = \frac{e^{-x}}{1+e^{-x}} = \frac{e^x}{e^x+1} = 1 - \frac{1}{1+e^x} = 1-\sigma(x)\right)
\end{align}
$$

#### (e)  

##### (i)  

$$
\begin{align}
\frac{\partial}{\partial v_{c}} J_{neg-sample} \left( v_{c}, o, U \right) 
&= \frac{\partial}{\partial v_{c}} \left( -\log\left( \sigma\left( u_{o}^{T}v_{c} \right) \right) - \sum_{s=1}^{K} \log \left( \sigma\left( -u_{w_{s}}^{T}v_{c} \right) \right) \right) \\
&= \frac{-1}{\sigma(u_{o}^{T}v_{c})}\sigma{u_{o}^{T}v_{c}}\left(1 - \sigma(u_{o}^{T}v_{c}) \right)u_{o} - - \sum_{s=1}^{K} \frac{1}{\sigma(-u_{w_{s}}^{T}v_{c})}\sigma(-u_{w_{s}}^{T}v_{c})\sigma(u_{w_{s}}^{T}v_{c})(-u_{w_{s}}) \\
&= \left( \sigma(u_{o}^{T}v_{c}) - 1 \right)u_{o} + \sum_{s=1}^{K} \sigma(u_{w_{s}}^{T}v_{c})(u_{w_{s}}) \\
\end{align}
$$

$$
\begin{align}
\frac{\partial}{\partial u_{o}} J_{neg-sample} \left( v_{c}, o, U \right)
&= \frac{-1}{\sigma(u_{o}^{T}v_{c})}\sigma(u_{o}^{T}v_{c})\left(1 - \sigma(u_{o}^{T}v_{c})\right) (-v_{c}) \\
&= \left( \sigma(u_{o}^{T}v_{c}) - 1 \right)v_{c}
\end{align}
$$

$$
\begin{align}
\frac{\partial}{\partial v_{w_{s}}} J_{neg-sample} \left( v_{c}, o, U \right) 
&= \frac{-1}{\sigma(-u_{w_{s}}^{T}v_{c})}\sigma(-u_{w_{s}}^{T}v_{c})\sigma(u_{w_{s}}^{T}v_{c})(-v_{c}) \\
&= \sigma(u_{w_{s}}^{T}v_{c})v_{c}
\end{align}
$$

##### (ii)  
    Because negative sampling loss uses only sampled data, so have less amount of computation than naive-softmax loss.

</br>
</br>
---


### 2. Problems  

#### (1)  

    (b), N-gram uses the number of sequence of (n-1) words and the number of sequence of n words. It means that if given word sequence does not exist in training data, probability will be 0 or NaN.  

#### (2)  

    (c), Because of the chain rule, if small gradients exist in a row, gradient signal gets smaller and smaller as it backpropagates further.  

#### (3)  

    (c), Even if we stack a lot of linear transformations, it still a complex linear expression. We can train more complex boundary that is non-linear with non-linear activations  
  
#### (4)  

    (c), Since encoder of RNN moves information sequentially, the last hidden state vector needs to capture all information about the source sentence.  

#### (5)  

    (c), P(books | students opened their) = P(students opened their books)/P(students opened their) = count(students opened their books) / count(students opened their) = 250/1000 = 0.25

#### (6)  

    F  

#### (7)  

    F  

#### (8)  

    T  

#### (9)  

    F

#### (10)

    F  

#### (11)

    Because RNN uses recurrent structure and applies the same weights W repeatedly.  

#### (12)  
    