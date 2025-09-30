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
