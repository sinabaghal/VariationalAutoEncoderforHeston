## Generative Modeling of Heston Volatility Surfaces Using Variational Autoencoders

In this project, I focus on training a Variational Autoencoder (VAE), a generative model, to produce Heston volatility surfaces. The Heston model is a widely used stochastic volatility model in finance, capable of capturing the complex dynamics of option prices. Once trained, this VAE can generate new volatility surfaces, which could be useful for various financial applications such as risk management, pricing exotic derivatives, etc. This project emphasizes the power of generative AI in advancing financial modeling.

Heston model consists of two coupled stochastic differential equations (SDEs):

$$
\begin{align*}
dS_t &= S_t \mu dt + S_t \sqrt{v_t} dW_t^S\\
dv_t &= \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v
\end{align*}
$$


| Symbol      | Description                                                         |
|-------------|---------------------------------------------------------------------|
| $$S_t$$     | The asset price at time $$t$$                                      |
| $$\mu$$     | The drift rate  _i.e.,_ expected return                                   |
| $$v_t$$     | The variance at time $$t$$                                        |
| $$\kappa$$  | The rate of mean reversion                                         |
| $$\theta$$  | The long-term variance _i.e.,_ mean reversion level                      |
| $$\sigma$$  | The volatility of volatility  _i.e.,_ how much $$v_t$$ fluctuates         |
| $$W_t^S, W_t^v$$   |Wiener processes where $$d W_t^S d W_t^v = \rho dt$$                          |



### Variational Autoencoders 

This section provides a brief overview of variational autoencoders (VAEs). For a deeper understanding, refer to [^1]. Video lectures (20 & 21) [^2] are also an excellent resource for learning the core concepts of VAEs.

Variational autoencoders (VAEs) belong to a broader family of probabilistic autoencoders. Unlike deterministic autoencoders, probabilistic autoencoders introduce a probabilistic framework that enables them not only to reconstruct input data but also to generate new data. In this framework, the encoder and decoder are no longer deterministic mappings; instead, they represent probabilistic distributions. The encoder maps an input $$x$$ to a distribution over the latent space, encoding it into a conditional distribution of latent variables rather than a single point. The decoder then samples from this latent distribution to produce a conditional distribution over possible reconstructions of $$x$$. Denote these conditional distributions as $$f$$ and $$g$$ for the encoder and decoder, respectively. Then 

- $$f: Pr(h \mid x; W_f)$$, where $$h$$ is the latent representation conditioned on the input $$x$$, and $$W_f$$ are the parameters of the encoder.
- $$g: Pr(\tilde{x} \mid h; W_g)$$, where $$\tilde{x}$$ is generated from the latent representation $$h$$, with $$W_g$$ as the parameters of the decoder. 

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot 2024-11-14 152106.jpg" width="80%" height="100%">
</p>

Probabilistic autoencoders learn to model the distribution of the data in the latent space enabling them to generate new examples by sampling from this distribution. This generative capability makes probabilistic autoencoders especially powerful. The key idea in a VAEs is to ensure that the encoder distribution over the latent space is close to a simple, fixed distribution, typically a standard normal distribution $$N(0, I)$$. As a result, to generate new data points, we can easily sample new latent representations from $$N(0, I)$$ and pass them through the decoder.

So we have two simultaneous goals. First, to reconstruct $$x$$ with high probability. Second, to ensure that $$\Pr(h \mid x; W_f)$$ is close to $$\mathcal{N}(0, I)$$. Denoting the training dataset by $$\{x_1,\cdots,x_m\}$$, this leads to the following objective function:

$$
\max_{W} \sum_{m} \log \Pr(x_n; W_f, W_g) - \beta \text{KL}\left(\Pr(h \mid x_n; W_f) \| \mathcal{N}(h; 0, I)\right)
$$

We have 

$$\Pr(x_n; W_f, W_g) = \int_h \Pr(x_n \mid h; W_g) \Pr(h \mid x_n; W_f) dh$$

In order to compute this integral, we make two simplifications:  

- First, assume that

$$\Pr(h \mid x_n; W_f) = \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n^2(x_n; W_f) I)=\frac{1}{\sqrt{2\pi}}\cdot e^{-\frac{\Vert h-\mu_n\Vert^2}{2\sigma_n^2}}$$

where the mean $$\mu_n$$ and variance $$\sigma_n$$ are obtained through the encoder. Therefore, 

$$\Pr(x_n; W_f, W_g) = \int_h \Pr(x_n \mid h; W_g) \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n^2(x_n; W_f) I) dh$$
    
- Second, approximate this integral by a single sample, namely:

$$\Pr(x_n; W_f, W_g) \approx \Pr(x_n \mid h_n; W_g)\quad \text{where} \quad h_n \sim \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n^2(x_n; W_f) I)$$

**NB:** In the context of training with stochastic gradient descent, this may not be considered an oversimplification!

The figure below illustrates the network architecture.

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot 2024-11-14 172234.jpg" width="80%" height="100%">
</p>

With this architecture, we face a challenge when trying to backpropagate through the stochastic sampling step. The sampling introduces randomness, which disrupts the flow of gradients and makes the training infeasible. To address this, VAEs use a technique called the **reparameterization**: Instead of sampling directly from the distribution $$h \sim q(h|x)$$, we rewrite $$h$$ as a deterministic function of the encoder’s output parameters and an independent random variable $$\zeta \sim \mathcal{N}(0,I)$$. The reparameterization trick transforms the sampling as follows:

$$
h = \mu_n(x) + \sigma_n(x) \cdot \zeta \quad \text{where} \quad \zeta \sim \mathcal{N}(0,I)
$$

Here $$\mu_n(x)$$ and $$\sigma_n(x)$$ are the mean and standard deviation of the latent distribution $$q(h|x)$$.

This transformation effectively makes the sample $$h$$ a function of $$x$$ and the learned parameters $$W_f$$, with the randomness isolated in $$\zeta$$. Now, $$h$$ can be treated as a deterministic input to the decoder network during backpropagation, allowing us to compute gradients with respect to the encoder parameters. The resulting network looks as follows:

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot 2024-11-14 172648.jpg" width="80%" height="100%">
</p>

Now, returning to the second approximation (recal that $$Pr(x_n; W_f, W_g)$$ represents the probability of reconstructing $$x_n$$), we have that:

$$
\log Pr(x_n; W_f, W_g) \approx -\frac{1}{2}\Vert x_n-\tilde{x}_n\Vert^2
$$

Moreover, it is noted that the KL divergence between $$N(\mu, \sigma^2)$$ and $$N(0, 1)$$ is given by:

$$
D_{\text{KL}} \big( N(\mu, \sigma^2) \parallel N(0, 1) \big) = \frac{1}{2} \left( \sigma^2 + \mu^2 - 1 - \ln(\sigma^2) \right)
$$

Putting pieces together and scaling by 2, we derive the following loss function for training our VAE:

$$
\min \frac{1}{m}\sum_n \Vert x_n-\tilde{x}_n\Vert^2 + \frac{\beta}{m} \cdot  \sum_n\left( \sigma_n^2 + \mu_n^2 - 1 - \ln(\sigma_n^2) \right)
$$

Two important notes are in order:

- $$\frac{1}{m}\sum_n \Vert x_n-\tilde{x}_n\Vert^2$$ grows with $$\text{dim}(x_n)$$. In other words, there is no normalization factor to take into account the input data point's dimension.  
- $$\mu_n$$ and $$\sigma_n$$ are functions of $$W_f$$, the encoder's weights. It is problem-specific how to choose the specifics of these functions. For example, in this project, we ask the network to learn $$\log \sigma$$. In other words, $$\log \sigma = f_\sigma(W_f)$$ for some function of $$W_f$$.

### VAE for Heston 

![](part1.gif)
![](part2.gif)


<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/vaefit.png" width="80%" height="100%">
</p>

### References 

[^1]: Kingma, D., Welling, M. (2019). *An Introduction to Variational Autoencoders*. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
[^2]: Poupart, P. (2019). *Introduction to Machine Learning*. [CS480/690 UWaterloo](https://cs.uwaterloo.ca/~ppoupart/teaching/cs480-spring19/)



