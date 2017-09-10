# Notes of Reading KDNuggets
## [227 data science key terms](http://www.kdnuggets.com/2017/09/data-science-key-terms-explained.html)
### Clustering: An unsupervised learning technique aiming at "maximizing the intraclass similarity and minimizing the interclass similarity". 
* Two key parts: feature selection and expectation maximization (EM)
* The examples of clustering algorithms:
* distance-based methods: k-means and k-medians
* density and grid-based methods: __DBSCAN__
* matrix factorization-based methods: it is for the data which is represented as sparse nonnegative matrices -- co-clustering.
* spectral-based methods: Working with the defined underlying similarity matrix
* graph-based methods: clustering data by converting the similarity matrix to be a network structure.

### Big data
* six Vs of big data: volume, velocity, variety, veracity, variablility and value.

### Machine learing
* machine learning: concerned with the question of how to construct computer programs that automatically improve with experience.
* association: identify associations between the various items that have been chosen by a particular user.
* reinforcement learning: concerned with the problem of finding suitable actions to take in a given situation in order to maximize a reward.

### Deep learning
* Deep learning: is not a nanacea; is not the fabled master algorithm; is not synartificial intelligence. It is a process applying deep neural network technologies to slove problems.
* biological neuron (__Learning more about neuron and the activation to the stimuli__). 
	* nucleus: holds the genetic information
	* cell body: process the activations and convert them the ouput activations  
	* dendtrites: receive activations from other neurons
	* axons: transmit activation to other neurons
	* axon endings: along with neighboring dendtrites, from the syapses between neurons
	* Process to transmission: chemical electronic signal diffused by the neurotransmitters across the synaptic cleft.
* recurrent neural network: directed cycle of the neurons. It allows internal temproal state representation, i.e. capable of processing sequence
* convolutional neural network (CNN): employs the mathematical concept of convolution, be thought of as sliding window (__*named kernel, filter or featrue detector*__) over top a matrix representation, to mimic the neural connectivity mesh of the biological visual cortex.
	* [Intuitive understanding](http://colah.github.io/posts/2014-07-Understanding-Convolutions/): taking the difference of neighboring pixels can detect the edges.
	* CNN: multi-layers of convolutions with nonlinear activation functions (e.g., **Relu**, or **tanh**)
	* Different filters and kernels are used for different applications (e.g., edge detection). Each of them can understood as  a selection of the specific feature.
	* Two aspects w.r.t. the computation load
		* **Location invariance**: pooling contributes to the invariance properties regarding scaling (especially on this), translation and rotation
		* **Compositionality**: a local patch of lower-level features into higher-level representation
		* From computer vision-understanding to natural language processing (NLP): Ability to extract the hidden information between neighboring pixels is equivalent to understand the semantic meaning  between close words in a sentence.
	* Hyperparameters of CNN:
		* **Narrow or wide convolution**: depends on whehter using **zero-padding**, i.e. forcing all the element that would fall outside of the matrix to be zero. With zero-padding is called wide convolution  and wise-versa.
		* **Stride size**: the ratio of the shift of filter at each layer. The larger the strid size, the fewer the filter of the further step and the smaller the output.
		* **Pooling layers** (e.g. max pooling): The capabilities are: 
			* fix the size of the output
			* reduce the dimensionality but keep the most salient information
			* make the model more robust
			* contribute to the invariance to translation and rotation
		* **Channels**: different perspectives to understand the data. Convolution operation can be applied across channels. Each channel can be a different representation of the data. 
* Long short term memory network (LSTM): a recurrent neural network  which is optimized for learning from and acting upon time-related data which may have nudfined or unkown lengths of time between eents of relevance. 

### Descriptive statistics: A set of statistical tools for quantitatively describing or summarizing a coolection of data
* Population: a selected individual or group representing the full set of members of a certain group of interest
* Sample: a subset sampler from a larger population
* Parameter: a value generated from a population
* Stastics: a value generated from a sample
* Generalizability: the ability to draw conclusions about the characteristics of teh population as a whole based on the results of data collected from a sample
* Distribution: the arrangement of data by the values of one variable in order, from low to high.
* Statistical terminology: mean, median, skew, kurtosis, mode,range, variance, standard deviation and interquartile range (IQR)
	* interquartile range (IQR): the difference between the socre delineating the $\mathrm{75^{th}}$ percentile and $\mathrm{25^{th}}$ percentile, the third and first quartiles respectively.

### Predictive analytics: Technology that learns from experience (data) to predict the future behavior of individuals in order to drive better decisions

Terms about **database**, **predicitve analytics**, **cloud computing**, **Hadoop and Apache Spark**, **IoT**, **NLP** can be referred to [the original blog](http://www.kdnuggets.com/2017/09/data-science-key-terms-explained.html).
