# Using a Brain-Computer Interface with a Human Support Robot for Object Grasping



## Methodology

### Data Pre-processing

In the data preprocessing stage, we perform two key operations to improve the quality and consistency of the dataset. Firstly, for different electrode signals in different EEG datasets, only the data of 14 key electrode positions corresponding to Epoc+ helmet were selected in this study: Af3 "and" F7 ", "F3", "Fc5", "T7 has", "what", "O1", "O2" and "P8", "T8", "Fc6", "F4", "F8", and "Af4", and to integrate these data for a large-scale labeled data set, for further analysis.

#### Data normalization

Secondly, to ensure unity across datasets, the min-max normalization method is used to standardize the feature data, and each feature value is scaled to the range of 0 to 1, which is calculated by the following formula:

We use the following formula for data normalization:

$$
x'_i = \frac{x_i - \min(x_j) \quad (0 \leq j \leq n)}{\max(x_j) \quad (0 \leq j \leq n) - \min(x_j) \quad (0 \leq j \leq n)}
$$

#### Position encoding

We use the following formula to define a unique label for each electrode:

For the case where dimension i is even, we use the sin function:

$$
PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right)
$$

For the case where dimension i is odd, we use the cos function:

$$
PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{(2i+1)/d_{model}}} \right)
$$

The final electrode arrangement selected is as follows:

![3.1](.\image\3.1.png)



### Spatial-Temporal Attention Block

This is one of the main contributions of this paper, We propose an EEG-based control method that utilizes an innovative attention-based encoder/decoder TCNet framework for  parsing information features within users' brainwaves. These parsed features are then employed through an enhanced  classifier to achieve precise control over robotic actions. The integration of an attention mechanism enables our method  to automatically identify the most discriminative features for control,  thus ensuring good performance also across a variety of datasets and collection environments.

#### Temporal Attention

The formula for the Temporal Attention module is as follows:

$$
\text{Attn}(Q^{(l)}, K^{(l)}, V^{(l)}) = \text{Softmax}\left(A^{(l)}\right)V^{(l)}, \quad A^{(l)} = \frac{Q^{(l)}(K^{(l)})^{\mathsf{T}}}{\sqrt{d_h}} + A^{(l-1)}
$$

#### Spatial Attention

The formula for the Spatial Attention module is as follows:

$$
Y \in \mathbb{R}^{C^{(l-1)} \times M \times N} \rightarrow Y^{\'} \in \mathbb{R}^{C^{(l-1)} \times M \times N}
$$

$$
\text{Attn}(Q^{(l)}, K^{(l)}, V^{(l)}) = \text{Softmax}\left(\frac{(Q^{(l)}W_k^{(l)}) + (Q^{(l)}W_k^{(l)})^{\mathsf{T}}}{\sqrt{d_h}}\right)V^{(l)}
$$

![Figure 3.2](.\image\Figure 3.2.png)



### TCNet Block

In this paper, we use Temporal Convolutional Networks (TCN) to aggregate temporal information.

$$ F(\hat{Z}^S_i) = \left( \hat{Z}^S_i *_{d}f \right) = \sum_{j=0}^{k-1} f(j) * \hat{Z}^S_{i-d*j} $$

Based on the dilated convolution framework, this paper also uses residual connections to transfer information from the previous layer to the next layer. The operation of each layer can be described as follows:

![Figure 3.4](.\image\Figure 3.4.png)



### Loss Function

We adopted the cross-entropy loss function as the final loss function:

$$
\mathcal{L}_{\text{cls}} = \frac{1}{T} \sum_t -\log(y_{t,c})
$$

 To further enhance the quality of our predictions, we employ an additional smoothing loss to mitigate such over-segmentation
errors. In this context, the following loss function is used as the smoothing loss.

$$
\mathcal{L}_{T-MSE} = \frac{1}{TC} \sum_{t,c} \hat{\Delta}^2_{t,c}
$$

$$
\hat{\Delta}_{t,c} = 
\begin{cases} 
\Delta_{t,c} & \text{if } \Delta_{t,c} \leq \tau \\
\tau & \text{otherwise}
\end{cases}
$$

$$
\Delta_{t,c} = \left| \log y_{t,c} - \log y_{t-1,c} \right|
$$

The resulting loss function is expressed as the combination of the two:

$$
\mathcal{L}_s = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{T-MSE}
$$


## Dataset and Equipment

#### EPOC X

The Epoc X headset, an advanced wireless EEG interface specifically designed to capture research-level EEG data, was used as the experimental device in this paper.

![Figure 4.1](.\image\Figure 4.1.png)

#### Tiago Robot

In our research, the TIAGO robot by PAL Robotics played an integral role as the primary experimental apparatus, tasked with the execution of motive actions derived from EEG signals.

![Figure 4.2](.\image\Figure 4.2.jpg)



#### Dataset

EEG Motor Movement/Imagery Dataset: This dataset consists of more than 1500 electroencephalogram (EEG) recordings of between one and two minutes in duration from 109 volunteers, captured using the BCI2000 system. The recordings included a variety of motor/imagery tasks that performed and imagined the opening and closing of the hand, fist, and foot movements. Baseline states were also recorded when the participant opened and closed his eyes. The dataset was provided by the BCI R&D Program team at the Wadsworth Center of the New York State Department of Health with a grant from the National Institutes of Health/National Institute of Biomedical Imaging and Bioengineering.

Grasp and Lift EEG Challenge: The dataset consists of EEG recordings from 12 subjects while performing six specific grasping and lifting movements subdivided into hand onset movement, first touch, two-hand onset loading phase, lifting, replacement, and two-hand release. Subjects also performed a multi-directional arm extension task in six different spatial directions: front, back, left, right, up, and down, 50 times in each direction, for a total of 300 actual motion data. For the motor imagery (MI) task, subjects were asked to imagine performing the same action, ensuring that the actual action was consistent with the number of trials of the imagined task. In addition, data containing three grasps of everyday objects and two wrist-twisting movements were also collected, each with 50 movements, providing 150 grasps and 100 wrist-twisting movements for each subject.



## Experimental procedures

The experiment consisted of four parts. First, we used the EPOC X helmet to collect electroencephalogram (EEG) signals from the participant. Next, we preprocessed the collected EEG signals. Then, we used Attention-based TCNet to analyze the processed EEG signals. Finally, based on the analyzed data, we used ROS to control the robot arm to grasp the object. The diagram of the experimental process is shown below:

![Figure 4.3](.\image\Figure 4.3.png)

## Experimental result:

#### Data processing:

The final processed data is shown in the figure:

![Figure 4.4A](.\image\Figure 4.4A.png)

![Figure 4.4B](.\image\Figure 4.4B.png)

#### Experimental scene:

![Figure A4](.\image\Figure A4.png)

![Figure A5](.\image\Figure A5.png)

![Figure A6](.\image\Figure A6.png)



## Evaluation

#### Accuracy of model prediction:

Top accuracy rate is selected as one of the evaluation indicators because it determines the final effect of the experiment. Since the model needs to predict two tasks (left hand or right hand, and the final action), the accuracy rate of the prediction with the highest possibility and the top two possibilities are selected as the final evaluation criteria.

 For K-fold cross-verification, K=4 is selected as the final evaluation criterion, and the final top accuracy is the average of the four training sessions.
 
$$
\text{Top}_1 = \frac{1}{4} \sum_{k=1}^{4} \text{Top}_1^k
$$

$$
\text{Top}_2 = \frac{1}{4} \sum_{k=1}^{4} \text{Top}_2^k
$$

Based on the recall rate, F-score, as a three-level indicator, is defined as:

$$
F_{\beta} = \frac{(\beta^2 + 1) PR}{\beta^2 \cdot P + R}
$$

Accuracy of model prediction:

![comp](.\image\comp.png)

#### Ablation experiments:

![Ab1](.\image\Ab1.png)



![Ab2](.\image\Ab2.png)
