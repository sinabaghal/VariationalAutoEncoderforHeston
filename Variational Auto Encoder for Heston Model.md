### Variational Autoencoders 

This section provides a brief overview of variational autoencoders (VAEs). For a deeper understanding, refer to [^1]. Video lectures [^2] are also an excellent resource for learning the core concepts of VAEs.

Variational autoencoders (VAEs) belong to a broader family of probabilistic autoencoders. Unlike deterministic autoencoders, probabilistic autoencoders introduce a probabilistic framework that enables them not only to reconstruct input data but also to generate new data. In this framework, the encoder and decoder are no longer deterministic mappings; instead, they represent probabilistic distributions. The encoder maps an input $$x$$ to a distribution over the latent space, encoding it into a conditional distribution of latent variables rather than a single point. The decoder then samples from this latent distribution to produce a conditional distribution over possible reconstructions of $$x$$. Denote these conditional distributions as $$f$$ and $$g$$ for the encoder and decoder, respectively. Then 

- $$f: P(h \mid x; W_f)$$, where $$h$$ is the latent representation conditioned on the input $$x$$, and $$W_f$$ are the parameters of the encoder. This distribution captures the probability of various representations $$h$$ given the input $$x$$.

- $$g: P(\tilde{x} \mid h; W_g)$$, where $$\tilde{x}$$ is reconstructed or generated from the latent representation $$h$$, with $$W_g$$ as the parameters of the decoder. This distribution captures the probability of the data $$\tilde{x}$$ given a particular latent representation $$h$$.

<p align="center">
<img src="https://github.com/sinabaghal/VariationalAutoEncoderforHeston/blob/main/Screenshot%202024-11-14%20152106.jpg" width="80%" height="100%">
</p>

In essence, probabilistic autoencoders learn to model the distribution of the data in the latent space, which enables them to generate new examples by sampling from this distribution. This generative capability makes probabilistic autoencoders especially powerful. The key idea in a VAEs is to ensure that this encoder distribution over the latent space is close to a simple, fixed distribution, typically a standard normal distribution $$N(0, I)$$, where $$I$$ is the identity matrix. By making the encoder output close to $$N(0, I)$$, we can easily sample new latent representations from this standard normal distribution and pass them through the decoder to generate new data points. 

So we have two simultaneous goals. First, to reconstruct $$x$$ with high probability. Second, to ensure that $$\Pr(h \mid x; W_f)$$ is close to $$\mathcal{N}(0, I)$$. This leads to the following objective function:

$$
\max_{W} \sum_{n} \log \Pr(x_n; W_f, W_g) - \beta \text{KL}\left(\Pr(h \mid x_n; W_f) \| \mathcal{N}(h; 0, I)\right)
$$

How do we compute $$\Pr(x_n; W_f, W_g)$$ now? We have 

$$\Pr(x_n; W_f, W_g) = \int_h \Pr(x_n \mid h; W_g) \Pr(h \mid x_n; W_f) \, dh$$

Make a significant simplification: assume that

$$\Pr(h \mid x_n; W_f) = \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n(x_n; W_f) I)$$

where the mean $$\mu_n$$ and variance $$\sigma_n$$ are obtained through the encoder.

Next, we need to approximate the integral over $$h$$. Notice that 

$$\Pr(x_n; W_f, W_g) = \int_h \Pr(x_n \mid h; W_g) \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n(x_n; W_f) I) \, dh$$
    
Approximate this integral by a single sample, namely:

$$\Pr(x_n; W_f, W_g) \approx \Pr(x_n \mid h_n; W_g)$$
    
where $$h_n \sim \mathcal{N}(h; \mu_n(x_n; W_f), \sigma_n(x_n; W_f) I)$$.


In a VAE, training involves calculating gradients to optimize the encoder and decoder networks. However, because the encoder outputs a probability distribution rather than a fixed value, we face a challenge when trying to backpropagate through the stochastic sampling step. This sampling introduces randomness, which would disrupt the flow of gradients and make it difficult to train the model effectively.


To address this, VAEs use a technique called the **reparameterization trick**. This trick allows us to rewrite the sampling step in a way that makes it differentiable and therefore compatible with backpropagation. Instead of sampling directly from the distribution $$h \sim q(h|x)$$, we rewrite $$h$$ as a deterministic function of the encoder’s output parameters (mean $$\mu$$ and variance $$\sigma$$) and an independent random variable $$\epsilon$$ that we sample from a standard normal distribution.

The reparameterization trick transforms the sampling as follows:

$$
h = \mu(x) + \sigma(x) \cdot \epsilon
$$

where $$\epsilon \sim N(0, I)$$. In this form:

- $$\mu(x)$$ and $$\sigma(x)$$ are outputs of the encoder network, representing the mean and standard deviation of the latent distribution $$q(h|x)$$.
- $$\epsilon$$ is a random variable sampled from a standard normal distribution, independent of $$x$$.


This transformation effectively makes the sample $$h$$ a function of $$x$$ and the learned parameters $$W_f$$ of the encoder network, with the randomness isolated in $$\epsilon$$. Now, $$h$$ can be treated as a deterministic input to the decoder network during backpropagation, allowing us to compute gradients with respect to the encoder parameters.


In other words, the **reparameterization trick** allows us to "move" the randomness outside the network by making the sample $$h$$ depend on a deterministic transformation of $$\mu(x)$$ and $$\sigma(x)$$ with the added random noise $$\epsilon$$. This approach enables the network to learn effectively by allowing gradients to flow back through the encoder during training.


The result is that we can now derive and optimize the variational lower bound, or **Evidence Lower Bound (ELBO)**, by backpropagating through the entire VAE architecture. This trick is crucial for making the VAE trainable and is one of the key innovations that make VAEs effective generative models.

### References 

[^1]: Kingma, D., Welling, M. (2019). *An Introduction to Variational Autoencoders*. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
[^2]: Poupart, P. (2019). *Introduction to Machine Learning*. [CS480/690 UWaterloo](https://cs.uwaterloo.ca/~ppoupart/teaching/cs480-spring19/)



