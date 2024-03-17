# Using a Brain-Computer Interface with a Human Support Robot for Object Grasping







## Dataset

##### a.Source

1. （important）[Motor Movement/Imagery Dataset](https://www.physionet.org/physiobank/database/eegmmidb/): Includes 109 volunteers, 64 electrodes, 2 baseline tasks (eye-open and eye-closed), motor movement, and motor imagery (both fists or both feet)
2. （important）[Grasp and Lift EEG Challenge](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data): 12 subjects, 32channels@500Hz, for 6 grasp and lift events, namely a). HandStart b). FirstDigitTouch c). BothStartLoadPhase d). LiftOff e). Replace f). BothReleasedhttp://www2.hu-berlin.de/eyetracking-eeg/testdata.html)
3. [BCI Competition IV-2a](http://www.bbci.de/competition/iv/#dataset2a): 22-electrode EEG motor-imagery dataset, with 9 subjects and 2 sessions, each with 288 four-second trials of imagined movements per subject. Includes movements of the left hand, the right hand, the feet and the tongue. [[Dataset Description\]](http://www.bbci.de/competition/iv/desc_2a.pdf)
4. [High-Gamma Dataset](https://github.com/robintibor/high-gamma-dataset): 128-electrode dataset obtained from 14 healthy subjects with roughly 1000 four-second trials of executed movements divided into 13 runs per subject. The four classes of movements were movements of either the left hand, the right hand, both feet, and rest.
5. [Left/Right Hand 1D/2D movements](https://sites.google.com/site/projectbci/): 19-electrode data of one subject with various combinations of 1D and 2D hand movements (actual execution).
6. [Mental-Imagery Dataset](https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698): 13 participants with over 60,000 examples of motor imageries in 4 interaction paradigms recorded with 38 channels medical-grade EEG system. It contains data for upto 6 mental imageries primarily for the motor movements. [[Article\]](https://www.nature.com/articles/sdata2018211#ref-CR57)http://suendermann.com/su/pdf/aihls2013.pdf)



##### b.details

Our needs:

![image-20240305142638845](C:\Users\34049\AppData\Roaming\Typora\typora-user-images\image-20240305142638845.png)

###### b1:[Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.](http://www.ncbi.nlm.nih.gov/pubmed/15188875)

The data are provided here in EDF+ format (containing 64 EEG signals, each sampled at 160 samples per second, and an annotation channel). For use with PhysioToolkit software, [rdedfann](https://www.physionet.org/physiotools/wag/rdedfa-1.htm) generated a separate PhysioBank-compatible annotation file (with the suffix .event) for each recording. The .event files and the annotation channels in the corresponding .edf files contain identical data.

In summary, the experimental runs were:

1. Baseline, eyes open
2. Baseline, eyes closed
3. Task 1 (open and close left or right fist)
4. Task 2 (imagine opening and closing left or right fist)
5. Task 3 (open and close both fists or both feet)
6. Task 4 (imagine opening and closing both fists or both feet)

![image-20240305142608776](C:\Users\34049\AppData\Roaming\Typora\typora-user-images\image-20240305142608776.png)

###### b2:Multimodal signal dataset for 11 intuitive movement tasks from single upper extremity during multiple recording sessions

[Ji-Hoon Jeong](https://pubmed.ncbi.nlm.nih.gov/?term=Jeong JH[Author]), [Jeong-Hyun Cho](https://pubmed.ncbi.nlm.nih.gov/?term=Cho JH[Author]), [Kyung-Hwan Shim](https://pubmed.ncbi.nlm.nih.gov/?term=Shim KH[Author]), [Byoung-Hee Kwon](https://pubmed.ncbi.nlm.nih.gov/?term=Kwon BH[Author]), [Byeong-Hoo Lee](https://pubmed.ncbi.nlm.nih.gov/?term=Lee BH[Author]), [Do-Yeun Lee](https://pubmed.ncbi.nlm.nih.gov/?term=Lee DY[Author]), [Dae-Hyeok Lee](https://pubmed.ncbi.nlm.nih.gov/?term=Lee DH[Author]), and [Seong-Whan Lee](https://pubmed.ncbi.nlm.nih.gov/?term=Lee SW[Author])

Arm-reaching along 6 directions: The participants were asked to perform multi-direction arm-reaching tasks directed from the center of their bodies outward. They performed the tasks along 6 different directions in 3D space: forward, backward, left, right, up, and down, as depicted in Fig. [3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7539536/figure/fig3/). In the real-movement tasks, the participants extended their arms along 1 of the directions. The arm-reaching paradigm required 50 trials along each direction so that data could be collected for a total of 300 trials. However, in the MI tasks, the participants only imagined performing an arm-reaching task; the number of trials in the MI paradigm was the same as in the real-movement paradigm.

Hand-grasping 3 objects: The participants were asked to grasp 3 objects of daily use via the corresponding grasping motions. They performed the 3 designated grasp motions by holding the objects, namely, card, ball, and cup, corresponding to lateral, spherical, and cylindrical grasp, respectively (see Fig. [3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7539536/figure/fig3/)). In the real-movement tasks, we asked the participants to use their right hands to grasp a randomly selected object and hold it using its corresponding grasping motion. Eventually, we acquired data on 50 trials for each grasp, and hence, we collected 150 trials per participant. In the MI tasks, the participants performed only 1 of the 3 grasping motions per trial, randomly. The number of trials in the MI paradigm was the same as that in the real-movement paradigm.

Wrist-twisting with 2 different motions: For the wrist-twisting tasks, the participants rotated their wrists to the left (pronation) and right (supination), as depicted in Fig. [3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7539536/figure/fig3/). During real-movement task, each participant maintained his/her right hand in a neutral position with the elbow comfortably placed on the desk. Notably, wrist pronation and supination are complex actions used to decode user intentions from brain signals. Additionally, these movements are intuitive motions for realizing neurorehabilitation and prosthetic control [[31](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7539536/#bib31)]. We collected data for 50 trials per motion (i.e., total 100 trials) per day, and the visual cues were randomly displayed.

Additionally, the participants were asked to participate in 3 recording sessions with a 1-week interval between each session. The experimental environment and protocols were the same for all 3 sessions. Consequently, we collected data from 3,300 trials (1,800 trials for arm-reaching, 900 for hand-grasping, and 600 for wrist-twisting) in all classes per participant, for both real-movement and MI paradigms.

![image-20240305211601067](C:\Users\34049\AppData\Roaming\Typora\typora-user-images\image-20240305211601067.png)



![image-20240305211409256](C:\Users\34049\AppData\Roaming\Typora\typora-user-images\image-20240305211409256.png)

###### b3.EEG datasets for motor imagery brain–computerinterfaceHohyun Cho1, Minkyu Ahn2, Sangtae Ahn3, Moonyoung Kwon1and Sung Chan Jun1,∗

data:http://gigadb.org/dataset/100295

For each subject, we recorded data for non-task-related and task(MI)-related states, as follows: Six types of non-task-related data: We recorded 6 types ofnoise data (eye blinking, eyeball movement up/down, eyeball movement left/right, head movement, jaw clenching,and resting state) for 52 subjects. Each type of noise was collected twice for 5 seconds, except the resting state, whichwas recorded for 60 seconds. 

Real hand movement: Before beginning the motor imageryexperiment, we asked subjects to conduct real hand movements. Subjects sat in a chair with armrests and watched amonitor. At the beginning of each trial, the monitor showeda black screen with a fixation cross for 2 seconds; the subjectwas then ready to perform hand movements (once the blackscreen gave a ready sign to the subject). As shown in Fig. 2,one of 2 instructions (“left hand” or “right hand”) appearedrandomly on the screen for 3 seconds, and subjects wereasked to move the appropriate hand depending on the instruction given. After the movement, when the blank screenreappeared, the subject was given a break for a random 4.1 to4.8 seconds. These processes were repeated 20 times for oneclass (one run), and one run was performed. 

MI experiment: The MI experiment was conducted with thesame paradigm as the real hand movement experiment. Subjects were asked to imagine the hand movement dependingon the instruction given. Five or six runs were performed during the MI experiment. After each run, we calculated the classification accuracy over one run and gave the subject feedback to increase motivation. Between each run, a maximum4-minute break was given depending on the subject’s demands.

![image-20240305220917953](C:\Users\34049\AppData\Roaming\Typora\typora-user-images\image-20240305220917953.png)



b4.https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data

## Code

#### 1.requirements

for data_loader.py

```python
pip install scipy pandas
pip install pyarrow
```





```python
pip install -U mne
pip install mat4py
```



