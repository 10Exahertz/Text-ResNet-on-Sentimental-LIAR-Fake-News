# Applying Deeper Convolutional Networks on The Sentimental LIAR Dataset


# Abstract 
We take the Sentimental LIAR dataset and network and perform network analysis and hyper-parametertuning on the original network.   We also apply the ResNet architecture on the network for deeperlearning. We find the ELECTRA embeddings are most well suited for the fake news classification task.A statistically significant F1 score increase of 0.061 was achieved. Both 4 Layer CNN and ResNet10architecture reach a similar performance level potentially demonstrating a larger dataset is needed forbetter performance

# Introduction

The maturation of the internet saw it take on a new and dangerous role in the modern political landscape. The last yearhas demonstrated the dangers of this new medium and the radical political hack-packs that it nurtures. Since the start ofthe Covid-19 pandemic we have seen one wave of misinformation after the other. Early on it was the 5G stories, and agovernment operated "plandemic". Then came the Bill Gate’s Vaccination chips over the summer, as the then Presidentmask usage misinformation. We saw the meteoric rise of Q-Anon, a group of rabid conspiracy theorists whose root ideologyis similar to the Nazi Conspiracies. This Q-Anon group grew in power rapidly, and in January 2021 a coalition of groupsinvolved in Q-Anon in some way or another stormed the US Capitol believing the 2020 Election lie that Donald Trump hadwon, purporting to "Stop the Steal." In the past it led to the rash Brexit decision, and coordinated propaganda campaigns onFacebook are considered to have greatly contributed to inciting the Myanmar Genocides. Fake News is a palpable issuethat must be resolved in some way.The body of work on fake news detection on long form documents, ones that come with a large body of context are growingrapidly. However, short text fake news detection systems are more rare, they are just as important considering the commonlyused social media sites use short form text. These shorter form texts provide the obvious issue of not enough context andinformation leaving the models trained on longer form fake news detection tasks constrained. More recently some datasetsand networks have been proposed to resolve these issues. The LIAR dataset contains short form texts from Politifact.com,a fact checking website. The dataset was a first of its kind and is the largest one for short form fake news classification.Sentimental LIAR took it a step further, adding in sentiment scores and emotion scores and made significant performancebosts over the original LIAR paper. The rest of this paper will include using deeper learning frame works from the field ofcomputer vision, namely the ResNet network and larger CNN networks to see if improvements can be made on the currentnetwork. The idea being that deeper, more hierarchical learning without vanishing gradients that ResNets provide can findbetter features and boost the performance. We also provide analysis on the model structure and input structure.

# Methods   
## Convolutional Neural Network
The general idea of the LeNet-5 like CNN for classification purposes is to take the input data and apply a convolvingkernel/filter over the text. In images a convolution allows for smoothing, and edge detection, depending on the kernel type.In a Convolutional Neural Net the kernel’s functions are learned, allowing it to do far more complex tasks. The deeper thenetwork, and the more channels, the better the feature representation is and the better the performance. until a cliff wherethe data is insufficient in some way, or vanishing gradients come into play.2
arXivTemplateThe Sentimental LIAR paper used a 2D CNN with max pooling layers in between. Below is a diagram showing the generalnetwork that we will be using for our testing. The one shown below is the 4 Layer CNN. We start with the input into theEmbedding System (ELECTRA-base here), this includes the Statement and the Binary Sentiment. The dimensions of allembedding systems is constrained to a size of 768, larger embeddings performed worse.  Then we add on the emotionscores before pushing it through the CNN layers. Each CNN layer uses a max-pooling, we use a [50,100,200,400] channelexpansion schema in this paper. We then feed it into a fully connected linear layer before passing it on to the classifier.

![alt text](Images/CNN4Text.jpg)


