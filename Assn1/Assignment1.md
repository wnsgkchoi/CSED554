# CSED554 Assn1 Hand Writing  

## Understanding word2vec  

Skip-gram word2vec은 다음과 같은 probability dist를 학습하는 것을 목표로 한다.  
$$
P(O = o | C = c) = \frac{\exp{u_{o}^{T}v_{c}}}{\sum_{w \in Vocab}{\exp{u_{w}^{T}v_{c}}}}
$$
즉, 주어진 center word $c$ 에 대하여, 모든 words 들의 vector와 center word $c$ 의 내적을 softmax한 것으로 이해할 수 있다. 한편, 이 학습 과정에서, word $c$ 와 $o$ 에 대한 loss는 다음과 같다.  
$$
J_{naive_{}softmax}(v_{c}, o, U) = - \log{P(O=o | C=c)}
$$
