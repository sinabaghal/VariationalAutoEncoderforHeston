### Variational Autoencoders 

This section provides a brief overview of variational autoencoders (VAEs). For a deeper understanding, refer to [^1]. Video lectures [^2] are also an excellent resource for learning the core concepts of VAEs.

Variational autoencoders (VAEs) belong to a broader family of probabilistic autoencoders. Unlike deterministic autoencoders, probabilistic autoencoders introduce a probabilistic framework that enables them not only to reconstruct input data but also to generate new data. In this framework, the encoder and decoder are no longer deterministic mappings; instead, they represent probabilistic distributions. The encoder maps an input $$x$$ to a distribution over the latent space, encoding it into a conditional distribution of latent variables rather than a single point. The decoder then samples from this latent distribution to produce a conditional distribution over possible reconstructions of $$x$$. Denote these conditional distributions as $$f$$ and $$g$$ for the encoder and decoder, respectively. Then 

- $$f: Pr(h \mid x; W_f)$$, where $$h$$ is the latent representation conditioned on the input $$x$$, and $$W_f$$ are the parameters of the encoder. This distribution captures the probability of various representations $$h$$ given the input $$x$$.

- $$g: Pr(\tilde{x} \mid h; W_g)$$, where $$\tilde{x}$$ is generated from the latent representation $$h$$, with $$W_g$$ as the parameters of the decoder. This distribution captures the probability of the data $$\tilde{x}$$ given a particular latent representation $$h$$.

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot 2024-11-14 152106.jpg" width="80%" height="100%">
</p>

Probabilistic autoencoders learn to model the distribution of the data in the latent space enabling them to generate new examples by sampling from this distribution. This generative capability makes probabilistic autoencoders especially powerful. The key idea in a VAEs is to ensure that the encoder distribution over the latent space is close to a simple, fixed distribution, typically a standard normal distribution $$N(0, I)$$. By making the encoder output close to $$N(0, I)$$, we can easily sample new latent representations from standard normal distribution and pass them through the decoder to generate new data points. 

So we have two simultaneous goals. First, to reconstruct $$x$$ with high probability. Second, to ensure that $$\Pr(h \mid x; W_f)$$ is close to $$\mathcal{N}(0, I)$$. This leads to the following objective function ($$x_1,\cdots,x_m$$ is the training dataset):

$$
\max_{W} \sum_{m} \log \Pr(x_n; W_f, W_g) - \beta \text{KL}\left(\Pr(h \mid x_n; W_f) \| \mathcal{N}(h; 0, I)\right)
$$

We have 

$$\Pr(x_n; W_f, W_g) = \int_h \Pr(x_n \mid h; W_g) \Pr(h \mid x_n; W_f) \, dh$$

In order to compute this integral, we make two simplifications:  

- First, assume that

$$\Pr(h \mid x_n; W_f) = \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n^2(x_n; W_f) I)$$

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
h = \mu_h(x) + \sigma_h(x) \cdot \zeta \quad \text{where} \quad \zeta \sim \mathcal{N}(0,I)
$$

Here $$\mu_h(x)$$ and $$\sigma_h(x)$$ are the mean and standard deviation of the latent distribution $$q(h|x)$$.

This transformation effectively makes the sample $$h$$ a function of $$x$$ and the learned parameters $$W_f$$, with the randomness isolated in $$\zeta$$. Now, $$h$$ can be treated as a deterministic input to the decoder network during backpropagation, allowing us to compute gradients with respect to the encoder parameters. The resulting network looks as follows:

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot 2024-11-14 172648.jpg" width="80%" height="100%">
</p>

Now, returning to the second approximation and recalling that $$Pr(x_n; W_f, W_g)$$ represents the probability of reconstructing $$x_n$$, we find that:

$$
\log Pr(x_n; W_f, W_g) \approx -\frac{1}{2}||x_n-\tilde{x}_n||^2
$$

Moreover, recall that the KL divergence between $$N(\mu, \sigma^2) \)$$ and $$N(0, 1)$$ is given by:

$$
D_{\text{KL}} \big( N(\mu, \sigma^2) \parallel N(0, 1) \big) = \frac{1}{2} \left( \sigma^2 + \mu^2 - 1 - \ln(\sigma^2) \right)
$$

Putting pieces together, scalingwe arrive at the following loss function for training our VAE:

$$
\min \sum_n ||x_n-\tilde{x}_n||^2 + \beta \cdot  \left( \sigma^2 + \mu^2 - 1 - \ln(\sigma^2) \right)
$$

### References 

[^1]: Kingma, D., Welling, M. (2019). *An Introduction to Variational Autoencoders*. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
[^2]: Poupart, P. (2019). *Introduction to Machine Learning*. [CS480/690 UWaterloo](https://cs.uwaterloo.ca/~ppoupart/teaching/cs480-spring19/)



