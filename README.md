# Applying Deeper Convolutional Networks on The Sentimental LIAR Dataset


# Abstract 
We take the Sentimental LIAR (S-LIAR) dataset and network and perform network analysis and hyper-parameter tuning on the S-LIAR network. We also apply the ResNet architecture on the network for deeper learning. We find that ELECTRA embeddings are most well suited for the downstream fake news classification task. A statistically significant F1 score increase of 6.1\% was achieved. Both 4 Layer CNN and Text-ResNet10 architecture reach a similar performance levels, potentially indicating that a larger dataset is needed for better performance.

# Introduction
The maturation of the internet saw it take on a new and dangerous role in the modern political landscape. The last year has demonstrated the dangers of this new medium and the radical political hack-packs that it nurtures. Since the start ofthe Covid-19 pandemic we have seen one wave of misinformation after the other.  Early on it was the 5G stories, and agovernment operated "plandemic". Then came the Bill Gate’s Vaccination chips over the summer, as the then President pushed mask usage misinformation. We saw the meteoric rise of Q-Anon, a group of rabid conspiracy theorists who grew inpower rapidly, and in January 2021 a coalition of groups involved in Q-Anon in some way or another stormed the US Capitol believing the 2020 Election lie that Donald Trump had won, purporting to "Stop the Steal." In the past it led to the rash Brexit decision, and coordinated propaganda campaigns on Facebook are considered to have greatly contributed to incitingthe Myanmar Genocides[1]. Fake News is one of the biggest issues of our day, anything to curtail its potency is useful.The body of work on fake news detection on long form documents, ones that come with a large body of context are growingrapidly. However, short text fake news detection systems are relatively rare and they are just as important considering that commonly used social media sites use short form text. These shorter excerpts come with the issue of not having enough context and information, leaving the models trained on longer form fake news detection tasks constrained. The LIAR dataset sought so solve this issue, it contains short form texts from Politifact.com, a fact checking website[2]. The dataset was a first of its kind and is the largest public one for short form fake news classification. Sentimental LIAR took it a step further,adding in sentiment scores and emotion scores and made significant performance boosts over the original LIAR paper[1].The reliance on Statement, Sentiment and Emotions only for fake news classification comes from both the Undeutsch and Four-factor theories[1]. They imply that fake news statements are different from real news statements both in writing styleand the sets of emotions used[1]. According to these theories the Sentiment and Emotion information S-LIAR included should greatly enhance fake news detection performance. Though Fake News detection should be difficult without webscraping, following these theories with style and emotion the S-LIAR authors believe we should get decent performancewithout a researching mechanism, albeit with noise, since these style and emotion rules are not likely universal. The S-LIARpaper reports 20+% Macro F1 improvements over majority baseline. Perhaps with deeper networks and fine tuning we canmake some real improvements in performance with this dataset. The rest of this paper will include using deeper learning frame works from the field of computer vision, namely the ResNet network and larger CNN networks to see if improvements can be made on the current network. The idea being that deeper, more hierarchical learning without vanishing gradients that ResNets provide may find better features and boost the performance[8]. We also provide analysis on the model structure and input structure. The Codes for the Networks made and the Appendix of this paper will be provided in the Github

# Related Works
## More Advanced Word Embeddings
The Sentimental LIAR paper used BERT[13] sentence embeddings in its architecture. BERT was the first of its kind self-supervised bi-directional attention based transformer with state of the art (SOA) benchmarks in GLUE and SQuAD tasks[4]. Since BERT many new networks have come out addressing the various issues with it. Roberta and ALBERT resolve thenext-sentence replacement task issues and the model size issues to get more training and more powerful embeddings[11][12].XLNet also takes a look at improving upon BERT’s use of the Masked-Language-Model (MLM) [6]. GPT-2 provides an auto-regressive model with incredible language modeling performance [10]. And ELECTRA changes the generator based network of BERT into a discriminator, and resolves the primary masked token issue, in which only 15% of the data can be trained[5]. These various networks all have their specialties and improvements on certain downstream tasks. ALBERT embeddings are seen as some of the best for tasks such as Semantic Textual Similarity (STS) and other semantic tasks[5][3].While ELECTRA authors state it outperforms on most downstream tasks in general, it does particularly well on SQuAD,Quora Question Pairs (QQP) tasks, and QNLI tasks [3], involving reading comprehension and question and answering tasks[5].
## LIAR and Sentimental LIAR
LIAR is a publicly available short form statement dataset. It comes from Politifact.com a popular political fact checking website. Each of the 12,836 statements in LIAR are annotated based on data available on Politifact with one of the following six labels: pants-fire, false, barely true, half-true, mostly-true, and true[2]. The dataset contains the statement, as well asrelevant contextual data such as:  [ID, Label, Statement,Subject, Speaker, Job, State Info, Party, Historical Counts, and CONTEXT.] S-LIAR changed the six classes of LIAR into a binary [True, False], the distribution of the labels in the S-LIARdataset is imbalanced with 65% being False claims and 35% being True claims[1].  The LIAR paper used CNN’s with unreported architecture to get a 26% validation accuracy score. Our runs of a simple LeNet-5 like CNN on the original LIARdataset yielded a 26.8% validation accuracy with a 23.8% F1 score. The Sentimental LIAR paper used a CNN networkwith BERT based embeddings. It also added the Google NLP API and IBM Watson NLP API to get sentiment scores and emotion scores respectively[1]. The emotion scores from IBM Watson featured ["Anger","Disgust","Joy","Fear","Sadness"],and Sentiment is a binary [positive, negative]. The Sentimental LIAR paper (S-LIAR) purported an accuracy score of 70%and an F1 score of 63% for their best model. Our run of this same best model yielded an average score of 68.83% witha standard deviation (STDEV) of 5.4% and an F1 score of 59.9% with a STDEV of 3.16%.  This paper seeks to make improvements to this model, and to apply the ResNet architecture to the dataset.
## LeNet-5
LeNet-5 was one of the first major successes in applying convolutional neural networks[7]. The architecture was developed for the handwritten character recognition of the MNIST dataset, it achieved SOA benchmarks of 99.2% on the task[7]. Theoriginal architecture included a first convolutional layer where a 28x28 kernel was applied to the original monochrome image to create 6 features channels, a second convolutional layer expanded the network into 16 channels.  Finally fully connected Gaussian layers yielded a classification output.  The CNN used in the Sentimental LIAR paper clearly took inspiration from this architecture with 2 Expansion CNN layers and then a fully connected linear layer[1].
## ResNet
The ResNet network provides residual and identity connections to allow for deeper neural networks while mitigating the vanishing gradient issue [8]. ResNet50 is commonly known for providing SOA results on image classification techniques[8].The number after the ResNet name denotes how deep the network is, the deeper the more hierarchical, at the cost of more training and requiring more data. ResNet101 is the backbone network of the famous Matterport Mask RCNN system [14].This system yields a high quality mask of a specific semantic object on the scene. We seek to see if its residual connections,deeper and more hierarchical learning can boost the performance of the more basic CNN text classifier used in S-LIAR.
## VDCNN for Text Classification
The VDCNN network from Conneau et al. applies deeper convolutional layers for text classification, similar to the goals ofthis paper[9]. They took inspiration from the VGG image classification network and the ResNet classification network. They found that the deeper the network the better the performance. The shortcut connections ResNet uses reduced degradation of accuracy with depth. When going too deep the accuracy will eventually degrade, it is always a balance of the data and the network. Notably the smallest data they tested the VDCNN on contained 120,000 data-points from AG News[9]. The performance of residual connections on text CNN’s was very promising. We seek to take inspiration from only the ResNet architecture, a different approach from the VDCNN.

# Methods   
## Convolutional Neural Network
The general idea of the LeNet-5 like CNN for classification purposes is to take the input data and apply a convolving kernel/filter over the text. In images a convolution allows for smoothing, and edge detection, depending on the kernel type.In a Convolutional Neural Net the kernel’s functions are learned, allowing it to do far more complex tasks [7]. The deeper the network, and the more channels, the better the feature representation is and the better the performance. That is until acliff is met where the data is insufficient in some way, or vanishing gradients come into play.

The Sentimental LIAR paper used a 2 Layer CNN with max pooling layers in between[1].  Fig 4 below is a diagram showing the general network that we will be using for our CNN testing. The one shown below is the 4 Layer CNN. We start withthe input into the Embedding System (ELECTRA-base here), this includes the Statement and the Binary Sentiment. The dimensions of all embedding systems is kept to a size of 768. Then we add on the emotion scores before pushing it throughthe CNN layers.

![alt text](Images/CNN4Text.jpg?raw=true)

## ResNet for Text Classification

Fig 2.B below is the image of a basic residual block. The curving line shows the residual connection that allows network to learn the residuals by passing along the identity of the last output, minimizing vanishing gradients affects. [8]. It does this bytaking the output of one block and adding it to the output of the next block, so that at worst it learns the identity, and at bestan improvement is made. This has allowed for very deep CNN’s to be created for very complex tasks[8].
![alt text](Images/resnet10TextandBlock.jpg)

Fig 2.A above is the architecture of the Text-ResNet10 Classification Network. Each block contains two 1x3 convolutional layers.We start with the Statement and Sentiment and move them through an embedding generator as before. We then appendthe emotion scores to this output. We first run a 1x15 CNN layer over this embedding output, becoming a form of look uptable for ResNet. Note that VDCNN did not use pre-trained embeddings like ELECTRA, their look up table also providessequence embeddings. This ResNet10 network contains only 3 Expansion Stages. Each stage in a ResNet Network expandsthe number of features/channels. The normal 4 stage expansion schema is [64, 128, 256, 512]. After going through the3 stages with residual connections, we arrive at a fully connected linear layer and a classifier. Every layer uses AveragePooling, ReLU activation functions and the first convolutional layer uses dropout for stability. In experiments we will detailmore model types, and of different sizes, but Text-ResNet10 with 3 Stages is our primary ResNet model.

# Experiments and Results
Below is a number of experiments that seek to improve the S-LIAR model in terms of performance and generalization. Aswell as dissect and understand the network.

## Sentimental LIAR vs LIAR
The original LIAR paper contained 6 classes ranging from "Pants on fire" to "True" as stated previously. We wanted to runthe 2 Layer S-LIAR CNN network on the 6 classes of LIAR and also test the original LIAR paper (without sentiment and emotion scores) for a more direct comparison to the S-LIAR model. We were curious how much a boost the S-LIAR paper received by the reduction of the number of classes from 6 to 2, and how well this network on the original LIAR dataset would perform. The Table below shows the results of this experiment. Majority Class baselines are included for comparison.

![alt text](Images/TableEx1.jpg)

It is clear that while the S-LIAR network (including sentiments and emotion scores) gives a boost in performance, theoverall performance gain of the network is granted via a reduction to the binary classification system. The S-LIAR networktrained for 6 classes sees a 35% F1 reduction in performance. These models were only run once.

## Generalization
Going through the total architecture of the Sentimental LIAR network we noticed a few issues regarding generalization.One was that the use of Speaker name, job, party affiliation, and so would not generalize well outside of Politifact, for use on social media sites. We ran an experiment to see how much of a performance change would come from removing the speaker information. Then we further tested this "no speaker info network" without the historical counts of the speaker. These list the historical counts regarding that person and how many times they lied or not on the LIAR 6-class system. This information would not be available on Instagram or Twitter, we tested without it for the sake of generalization. Taking the Speaker info out we see a small increase in the F1 score. Likely due to making information more salient, less info means its easier to distinguish important features elsewhere. Further taking the speakers historical counts the performance on the F1 score drops 5% to 56.46%. Again Majority Class baselines are included for comparison.

![alt text](Images/TableExOriginal.jpg)

## Different Embeddings
As stated before RoBERTa, ALBERT, XLNet, GPT-2, and ELECTRA are some of these new, state of the art models. RoBERTa and ALBERT seek to resolve the next-sentence prediction task and increase training size and performance[12][11].ALBERT is state of the art on semantic text tasks such as STS and SST[5].   ELECTRA changes the generator to a discriminator.  ELECTRA has state of the art performances on question and reading comprehension tasks (QQP, QNLI,SQuAD)[5].  Below shows the results of testing all of these models for potential improvements.  For ELECTRA andnon-sentence embeddings systems we used an averaging method for sequence embeddings.

![](Images/TableEmbeddings.jpg)

ELECTRA likely outperformed all of the other models since it outperformed them on SQuAD, QQP, and QNLI tasks inthe original ELECTRA paper[5]. These tasks involve finding important entities within the text that answer questions and involved reading comprehension. This indicates that reading comprehension, in particular finding important words within a sequence, is important to fake news detection. Perhaps certain words or phrases are commonly used in fake news.

## CNN Fine Tuning
Figure 6 below shows the results of different CNN layer testing. We testing up to 5 Convolutional Layers and found that 4 CNNlayers gave the best performance. This is likely because while the deeper network is technically better, at a certain point thesystem is constrained either due to vanishing gradients and/or not enough data. In this case we believe both reasons apply. Many other smaller changes were made such as the kernel sizes, channel sizes, dropout, etc. All of these are included in the Appendix on GitHub.

![alt text](Images/TableExCNN.jpg)

## ResNet Fine Tuning
For ResNet we also performed a multitude of model and hyper-parameter testing, all of which is also included in the Github Appendix. Figure 7 shows the comparisons of the best ResNet models we found. Each of the networks in Figure 7 (below) use the initial 1x15 CNN layer and 1x3 CNN Residual Blocks. The ResNet10 network in Fig 2.A had the best results. Likely by ResNet18 there is not enough data for robust weights and accuracy degradation starts to set-in. This is further observed by alarger standard deviation on the F1 score, demonstrating that the model was more sensitive to small training changes at 18layers. 2 epochs were used for these experiments, beyond 2 epochs overfitting occured rapidly. Text-ResNet10 with 3 Stagesand a dropout layer between the first CNN and the first ResNet block is the best model.

![alt text](Images/TableExResNet.jpg)

## Simple Feed Forward Model and Final Comparison
We wanted to make sure that the statements were actually contributing to the networks performance.  So we ran a very simple Feed Forward network on just the Sentiment score and emotion scores to test its performance. It is clear that theCNNs and ResNet systems with the embeddings are contributing to the performance gains. The table below has all the best models, including the Feed Forward network just described and the Majority Class Baselines. Note that the 4 Layer CNNand Text-ResNet10 models are a statistically significant increase in F1 performance over the Original Model. With p valuesof 0.028 and 0.023 respectively (α< 0.05). However, Text-ResNet10 is not a statistically significant gain over the 4 LayerCNN, with a p-value of 0.45. Notably, small gains are made in the accuracy scores, that are not statistically significant.
 
 ![alt text](Images/TableExFinal.jpg)
 
## Model Demonstration 
We performed a final test on the model which was to manually give it 7 statements and analyze its decisions on these statements. The statements build in difficulty (subjective) and come from a variety of sources, including Politifact. Sadly the model predicted all of these statements were False, likely having a high false detection rate. Including F.D.R’s Pearl Harbor speech, although the sigmoid output for that one was [0.47, 0.53], it was close to a neutral prediction. Many improvements are to be made in the field of Fake News Classification on short form texts.

![alt text](Images/TableDemo.jpg)

# Conclusions
We introduced ResNet architecture to the Sentimental LIAR (S-LIAR) dataset, performed analysis on the network, the data and the original S-LAIR CNN network. We also fine tuned the models for maximum performance in accuracy andthe F1 score. We generalize the network beyond Politifact.com so that it can predict on Twitter and Instagram statements,improving its actual utility. We generalized by hiding speaker info and the speaker historical counts.ELECTRA’s replace token task allows it to perform better for SQuAD, QQP (Quora Question Pairs) and QNLI. These tasks involve pulling important entities from a text that are useful for the task. SQuAD tasks focus on reading comprehension, QQP on question matching and QNLI on QA matching. These are clearly more important for the Fake News classification than the other transferable downstream tasks. 

Through hyper-parameter tuning on the S-LIAR CNN network we find that more layers leads to more features and better performance as long as you can robustly train the network. This allows for better hierarchical learning of more complexand non-linear features.  We found the optimal junction of training data and deepness of the learning architecture at 4 Convolutional Layers, this yielded an increase in Macro F1 performance over the original architecture of 6.2%, a statistically significant increase.

The Text-ResNet10 architecture with its residual connections allowed for deeper learning but it appears it and the 4 LayerCNN have stagnated at around 67% accuracy and an F1 score of 62.7%.  ResNet’s lack of boost over the 4 Layer CNN indicates that perhaps this is the most we can get out of the current dataset.  We consider this the case since accuracy degradation with depth on ResNet networks is uncommon. The S-LIAR dataset is an order of magnitude smaller than eventhe smallest tested on the similar VDCNN text network[9]. After testing the trained network on a small set of manually picked data we found that the network has a high "Fake news" prediction rate.  While Macro F1 score improvements were significant, improvements to the accuracy with the best model are only 3.43% above the majority baseline.  Also overfitting occurs rapidly with this dataset, likely due to the smallness of the dataset, S-LIAR concluded this as well[1]. The combination of these issues may indicate that not only is the relatively small dataset insufficient. But that the Undeutsh andFour-Factor theories the S-LIAR paper were relying on to produce Fake News detections are insufficient. Ultimately, the task alone (fake news detection via style and emotion) compounded via the smallness of the dataset is just far too difficult for meaningful network learning and significant benchmark improvements.

Furthermore, going back to Figure 9 we see that the last two statements are more interesting than a simple True or False.The 6th statement is True, and the 7th statement is False. However, they are both made by Tucker Carlson, and use the same techniques of powerful imagery, fear-casting, and overly dramatized verbage that pulls people in. It is clear that propaganda styled language can often be "True", and so a lot of noise will be introduced to the S-LIAR system in its current set-up. Demonstrating yet again that not only is the S-LIAR dataset small, its task is too difficult in general. For improvements tothe LIAR benchmark more outside data and a research checking web-scraping mechanism before the network’s inputs islikely needed. However, such systems would get very complex fast. Therefore in the future we want to classify propagandist statements (not true vs false) using the same techniques here involving ResNet’s and the ELECTRA embeddings, as well as the S-LIAR authors use of Google and IBM API’s. We feel that an adaptation the SEMEVAL 2020 Task 11 dataset [22] forshort form propaganda classification training is a more feasible task. Using ResNet architecture to provide the detection,hopefully we can create a network that provides useful Propaganda predictions.

# References
[1]   B. Upadhayay, V. Behzadan. "Sentimental LIAR: Extended Corpus and Deep Learning Models for Fake Claim Classification" In proceeding ofISIIEEE, Virtual, 2020.

[2]   W. Y. Wang. ""Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection" InACL. Vancouver, Canada, 2017.

[3]   A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, S. R. Bowman.  "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding"ICLR,New Orleans, USA, 2019

[4]   P. Rajpurkar, R. Jia, P. Liang. "Know What You Don’t Know: Unanswerable Questions for SQuAD"ACL, Melbourne Australia, 2018.

[5]   K. Clark, M.T. Luong, Q.V. Le, C.D. Manning "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators."ICLR, Virtual, 2020.

[6]   Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, Q.V. Le.  "XLNet:  Generalized Autoregressive Pretraining for Language Understanding."arxiv, 2019. citearxiv:1906.08237

[7]   Y. LeCun, L. Buttou, Y. Bengio, and P. Haffner.  "Gradient Based Learning Applied to Document Recognition."  Proceedings of theIEEE( Volume:  86, Issue:  11, Nov.1998).

[8]   K. He and X. Zhang and S. Ren and J. Sun "Deep Residual Learning for Image Recognition." InCVPR, Las Vegas, USA, 2016.

[9]   A. Conneau, H. Schwenk, L. Barrault and Y. LeCun. "Very Deep Convolutional Networks for Natural Language Processing." InACL, Valencia, Spain, 2017.

[10]   A. Radford, J. Wu, R. Child, D. Luan, D. Amodei and I.Sutskever. "Language Models are Unsupervised Multitask Learners." FromOpenAI, 2019.

[11]   Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, R. Soricut. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." InICLR, Virtual,2020.

[12]   Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, V. Stoyanov.  "RoBERTa: A Robustly Optimized BERT Pretraining Approach."InArXiv, 2019. http://arxiv.org/abs/1907.11692

[13]   J. Devlin, M.W. Chang, K. Lee, K. Toutanova.   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."   In Proceedings of the 2019Conference of the NA Chapter of the ACL: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, 2019

[14]   W. Abdulla. "Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow." GitHub, https://github.com/matterport/Mask_RCNN , 2017

[15]   Sen  K.  McCarthy."Says  Joe  Biden  is  “going  to  control  how  much  meat  you  can  eat.”"April  2021,  https://www.politifact.com/factchecks/2021/apr/29/kevin-mccarthy/kevin-mccarthy-repeats-pants-fire-claim-biden-will/

[16]   Pres. J. Biden. "In gun policy address, Joe Biden exaggerates about background checks at gun shows." April 2021, https://www.politifact.com/factchecks/2021/apr/08/joe-biden/gun-policy-address-joe-biden-was-wrong-about-backg/

[17]   L. Jacobson.  "Fact-checking Joe Biden on how little some corporations pay in taxes."  April 2021, https://www.politifact.com/factchecks/2021/apr/12/joe-biden/fact-checking-joe-biden-corporation-taxes/

[18]   M. Chan  "A Date Which Will Live in Infamy.’ Read President Roosevelt’s Pearl Harbor Address"  https://time.com/4593483/pearl-harbor-franklin-roosevelt-infamy-speech-attack/

[19]   No Author. "Current Weather On A Flat Earth." www.darksky.net/flatearth

[20]   T. Carlson. "Tucker: Americans are being paid to stay home." May 2021, Fox News, https://video.foxnews.com/v/6253165176001sp=show-clips

[21]   T. Carlson. "Tucker: The West is a birthright we must preserve." July 2017, Fox News, https://video.foxnews.com/v/5494955601001sp=show-clips

[22]   G.D. San Martino,  A.B. Cedeño,  P. Nakov,  H. Wachsmuth,  R. Petrov   "SEMEVAL 2020 TASK 11 "DETECTION OF PROPAGANDA TECHNIQUES IN NEWSARTICLES" " QCRI, 2020 https://alt.qcri.org/semeval2020/index.php?id=tasks
# Appendix
## CNN Fine Tuning Tests
![alt text](Images/CNNFineTuning.jpg)

## ResNet Fine Tuning Tests
![alt text](Images/ResnetTEstingBW.jpg)
