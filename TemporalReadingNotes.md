# Notes of Reading KDNuggets
In this brief notes, it only contains the terms and topics and I am interested in, but not familiar enough. For complete content, please refer to the originals.
## [227 data science key terms](http://www.kdnuggets.com/2017/09/data-science-key-terms-explained.html)
### Clustering: An unsupervised learning technique aiming at "maximizing the intraclass similarity and minimizing the interclass similarity". 
* Two key parts: feature selection and expectation maximization (EM)
* [The examples of clustering algorithms](http://www.kdnuggets.com/2016/08/10-algorithms-machine-learning-engineers.html/2): 
* distance-based methods: k-means and k-medians
* density and grid-based methods: __DBSCAN__
* matrix factorization-based methods: it is for the data which is represented as sparse nonnegative matrices -- co-clustering.
* spectral-based methods: Working with the defined underlying similarity matrix
* graph-based methods: clustering data by converting the similarity matrix to be a network structure.
![](https://cdn-images-1.medium.com/max/600/1*a58qZWbN7aAxFUTPeRiRTQ.png)

### Big data
* six Vs of big data: volume, velocity, variety, veracity, variability and value.

### Machine learning
* machine learning: concerned with the question of how to construct computer programs that automatically improve with experience.
* association: identify associations between the various items that have been chosen by a particular user.
* reinforcement learning: concerned with the problem of finding suitable actions to take in a given situation in order to maximize a reward.

### Deep learning
* Deep learning: is not a panacea; is not the fabled master algorithm; is not synonymous of artificial intelligence. It is a process applying deep neural network technologies to solve problems.
* biological neuron (__Learning more about neuron and the activation to the stimuli__). 
	* nucleus: holds the genetic information
	* cell body: process the activations and convert them the output activations  
	* dendrites: receive activations from other neurons
	* axons: transmit activation to other neurons
	* axon endings: along with neighboring dendrites, from the synapses between neurons
	* Process to transmission: chemical electronic signal diffused by the neurotransmitters across the synaptic cleft.
* recurrent neural network: directed cycle of the neurons. It allows internal temporal state representation, i.e. capable of processing sequence
* convolution neural network (CNN): employs the mathematical concept of convolution, be thought of as sliding window (__*named kernel, filter or feature detector*__) over top a matrix representation, to mimic the neural connectivity mesh of the biological visual cortex.
	* [Intuitive understanding](http://colah.github.io/posts/2014-07-Understanding-Convolutions/): taking the difference of neighboring pixels can detect the edges.
	* [CNN](<http://cs231n.github.io/convolutional-networks/>): multi-layers of convolutions with nonlinear activation functions (e.g., **Relu**, or **tanh**)
	* Different filters and kernels are used for different applications (e.g., edge detection). Each of them can understood as  a selection of the specific feature.
	* Two aspects w.r.t. the computation load
		* **Location invariance**: pooling contributes to the invariance properties regarding scaling (especially on this), translation and rotation
		* **Constitutionality**: a local patch of lower-level features into higher-level representation
		* From computer vision-understanding to natural language processing (NLP): Ability to extract the hidden information between neighboring pixels is equivalent to understand the semantic meaning  between close words in a sentence.
	* Hyper-parameters of CNN:
		* **Narrow or wide convolution**: depends on whether using **zero-padding**, i.e. forcing all the element that would fall outside of the matrix to be zero. With zero-padding is called wide convolution  and wise-versa.
		* **Stride size**: the ratio of the shift of filter at each layer. The larger the stride size, the fewer the filter of the further step and the smaller the output.
		* **Pooling layers** (e.g. max pooling): The capabilities are: 
			* fix the size of the output
			* reduce the dimensionality but keep the most salient information
			* make the model more robust
			* contribute to the invariance to translation and rotation
		* **Channels**: different perspectives to understand the data. Convolution operation can be applied across channels. Each channel can be a different representation of the data. 
* Long short term memory network (LSTM): a recurrent neural network  which is optimized for learning from and acting upon time-related data which may have undefined or unknown lengths of time between eents of relevance. 

### Descriptive statistics: A set of statistical tools for quantitatively describing or summarizing a collection of data
* Population: a selected individual or group representing the full set of members of a certain group of interest
* Sample: a subset sampler from a larger population
* Parameter: a value generated from a population
* Statistics: a value generated from a sample
* Generalizability: the ability to draw conclusions about the characteristics of the population as a whole based on the results of data collected from a sample
* Distribution: the arrangement of data by the values of one variable in order, from low to high.
* Statistical terminology: mean, median, skew, kurtosis, mode,range, variance, standard deviation and interquartile range (IQR)
	* interquartile range (IQR): the difference between the score delineating the $\mathrm{75^{th}}$ percentile and $\mathrm{25^{th}}$ percentile, the third and first quartiles respectively.

### Predictive analytics: Technology that learns from experience (data) to predict the future behavior of individuals in order to drive better decisions

Terms about **database**, **predicitve analytics**, **cloud computing**, **Hadoop and Apache Spark**, **IoT**, **NLP** can be referred to [the original blog](http://www.kdnuggets.com/2017/09/data-science-key-terms-explained.html).

## [Top 10 machine learning algorithms](http://www.kdnuggets.com/2016/08/10-algorithms-machine-learning-engineers.html)
All the method mentioned in this part are implemented in [**scikit-learn**](http://scikit-learn.org/stable/index.html) python library.
* [**Ensemble methods**](<http://www.kdnuggets.com/2016/02/ensemble-methods-techniques-produce-improved-machine-learning.html>):  construct a set of classifiers and then classify new data points by taking a weighted vote of their predictions, including **Bayesian averaging**, **error-correcting output coding**, **bagging (bootstrapping)**, and **boosting**. The advantages of ensemble methods are:  
  * average out the bias;
  * reduce the variance;
  * improve the generalization performance, i.e. unlikely to be over-fitting.
![](https://cdn-images-1.medium.com/max/600/1*BKzn0JPoIJhob5g0j5xj5A.png)
* More words about four difference types of ensemble learning:
  * Voting and averaging: the first step is to create multiple classification/regression models (e.g., using the same dataset with different algorithms, or using different splits of the same training dataset to train same algorithm) using some training dataset; then do voting (e.g., **majority voting**, **weighted voting**, or **simple averaging**).  
  ![](http://www.kdnuggets.com/wp-content/uploads/classify-animals-1.jpg)
  * Stacking: The basic idea is to train machine learning algorithms with training dataset and then generate a new dataset with these models. Then this new dataset is used as input for the combiner machine learning algorithm.  
  ![](http://www.kdnuggets.com/wp-content/uploads/classify-animals-2.jpg)
  * Bagging: using the same algorithm (e.g., **random forest** or **decision tree**) with random sub-samples of the dataset which are drawn from the original dataset randomly with bootstrap sampling method (e.g., **sampling with replacement**).  
  ![](http://www.kdnuggets.com/wp-content/uploads/training-cat-classifier.jpg)
  * Boosting: Converting Weak Models to Strong Ones. Boosting (e.g., **adaBoost**) incrementally builds an ensemble by training each model with the same dataset but where the weights of instances are adjusted according to the error of the last prediction. The more times of mis-prediction of the instance, the higher the weight on that instance when computing the loss.
* **Unsupervised learning**: a type of learning scenario for finding better representation of the data, i.e. representation learning.
  * Clustering: it seems that **DBSCAN** performs best.
  * PCA, SVD, and independent component analysis (ICA).
    * PCA: a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
    * SVD: PCA is actually a simple application of SVD.
    * ICA: defines a generative model for the observed multivariate data, which is typically given as a large database of samples.
![](https://cdn-images-1.medium.com/max/800/1*RBN-qGYgTFlsxzfORsqoEg.png)
