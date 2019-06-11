# head-Based Human-Computer Interaction
### While the detailed documentation on the way, you may find a brief description of this project below... 
This is an open-source project on **Head-based Human-Computer Interaction** that allows the user interact with the device via a camera which is related to the both of **Computer Vision** and **Human-Computer Interaction** fields and provides a collection of multi-disciplinary solutions for the given problem.

A Head-based Interaction would be only scrolling/swiping actions on small devices or precisely pointing a location on the screen via a cursor in a computer environment. In the second case, the user may have full control of the device with a completely hands-free interaction. 

Besides, highlighting the importance of Head-based Interaction, **The primary goals of the project** may be listed as: 

- **Building the components** of Head-based Interaction with providing technical background,
- Implementing each component **in as many alternative ways as possible**,
- Allowing the future developers to re-build the same interaction by combining the available sub-solutions **on their native platforms** among the ones provided here.

The project itself is still in the early stages of its development, and it will be mainly written in **Python3** by employing many of the other open-source frameworks like **OpenCV**, **DLIB**, **Tensorflow**. But, its modular implementation will allow it to run without installing all of the listed dependencies at the very below.
 
## 3 Components of Head-based Interaction

There are several ways to build Head-based Interaction, and each method may own its specific design. In spite of this variety, we can list the primary tasks of the interaction as below:

1. **Visual Input Estimation From Head**
2. **Mapping Between Estimated Input and Desired Output**
3. **Responding by Desired Output**

Below, you may find the details of each component and the relation between them.

### 1. Visual Input Estimation From Head

Head-based Interaction begins with the estimation of visual input from user's appearance on the camera. Based on practicality, this would be a 2D input like the center coordinates of the face area in the current frame, a set of 2D inputs like coordinates of facial landmarks or a 3D input like the head pose in a 3D world with respect to the camera.

This repository currently implements the following [**Input Estimators**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators) for Head-based Interaction: 
 
1. [**Face Box Detectors**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators/FaceDetectors)
   - **HaarCascadeFaceDetector**, utilizes `cv2.CascadeClassifier`
   - **LIBFrontalFaceDetector**, utilizes `dlib.get_frontal_face_detector`
   - **CV2Res10SSDFaceDetector**, utilizes `cv2.dnn` module to run the *Single Shot Detector* (SSD) framework with a *ResNet* base network
   - **TFMobileNetSSDFaceDetector**, utilizes `tensorflow` to run the *Single Shot Detector* (SSD) framework with a *MobileNet* base network
2. [**Facial Landmark Detectors**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators/FacialLandmarkDetectors) 
   - **DLIBFacialLandmarkDetector**, utilizes `dlib.shape_predictor`
   - **YinsCNNBasedFacialLandmarkDetector**, utilizes a CNN based `tensorflow` model
3. [**Head Pose Estimators**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators/HeadPoseEstimators) 
   - **DLIBHeadPoseEstimator**, utilizes `AnthropometricHeadPoseCalculator` 
   - **YinsHeadPoseEstimator**, based on a 3D face model from the [OpenFace](https://github.com/cmusatyalab/openface/) project
4. [**Head Gaze Calculators**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators/HeadPoseEstimators) 
   - **MuratcansHeadGazer**, an experimental approach to estimate Head Gaze on the screen


Each **Input Estimator** provides **Input Values** (*a Numpy array*) in at least 2D and they have a common interface as `InputEstimatorABC` to communicate with the **Mapping Functions** as described below. The current **Facial Landmark Detectors** run on top of a default **Face Box Detector** which is assigned to that **Landmark Detector**. But one can easily change this **Face Box Detector** by just setting the `FaceDetector` argument as desired. Similarly, the existing **Head Pose Estimators** rely on the output of a **Landmark Detector** and it can be specified by the `landmarkDetector` argument. This integrity allows experimenting with the combinations of separate detectors on each layer and building different **Input Estimators** as many as possible.


Soon, the further documentation on **Input Estimators** will be available under [its sub-module](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/InputEstimators) with the detailed instructions to run demos.
  
### 2. Mapping Between Estimated Input and Desired Output

As the second component of Head-based Interaction, we implement several Mapping Functions with providing their background under the [**HeadCursorMapping**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/HeadCursorMapping) sub-module. The main functionality of **HeadCursorMapping** is casting the estimated **Input Values** to the desired **Output Values** within the given dimensions and boundaries. 

One may be interested in just switching a boolean variable by those desired **Output Values** or setting volume in one degree or controlling the cursor for precise pointing on the 2D screen. We currently focus on only implementing a functionless cursor to evaluate **Head-based Pointing** with the given **Input Estimators** above. The cursor floats in a bounded area according to the estimated **Input Values** and it has no function than visualization. 

The [**HeadCursorMapping**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/HeadCursorMapping) sub-module implements the following two general **Mapping Functions**: 
 
1. **Static Mapping**
2. **Dynamic Mapping**

The `MappingABC` interface also implements several **Mapping Strategies** which are necessary to calculate well-defined **Input Values** from the raw values **Input Estimators** return. Each strategy has its own calculation we implemented based on our over 10-years personal experience with **Head-based Interaction**. They correspond each **Input Estimator** as below:

- `calculateInputValuesFromFaceBox()`: A strategy for generating 2D **Input Values** only from **Face Detection**.  
- `calculateInputValuesFromNose()`: A strategy for generating 2D **Input Values** from nose direction based on **Facial Landmark Localization**.  
- `calculateInputValuesFromHeadPose()`: A strategy for generating 2D **Input Values** from the 3D Head Rotation in Eular values based on **Head Pose Estimation**.  
- `calculateInputValuesFromHeadGaze()`: A strategy for generating 2D **Input Values** via an imaginary 3D ray calculated from Head Center to the screen based **Head Pose Estimation**. \
  (*Similar to Eye-Gazing approach but treats the whole head as a single eyeball)*.  

Soon, you will be able to find the implementational details of these **Mapping Strategies** along with the two **Mapping Functions** under the [**HeadCursorMapping**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/HeadCursorMapping) sub-module.

### 3. Responding by Desired Output

The last component of Head-based Interaction practices the interaction via real-life applications. One may apply this interaction in a gaming environment to control the first-person view. Another can build an interactive keyboard that works via head movements.

It is planned to implement an actual [**CursorController**](https://github.com/muratcancicek/Head_Based_Interaction/tree/master/HeadBasedInteractionPy/CursorController) in a complete product for end-user that communicates with the Operation System and provides Head-based Pointing via a well-packaged software. The schedule for this work is still uncertain, and it requires further support. 

## Usage

All of the existing functionalities will be runnable via `startInteraction.py`. 
Please find a few example commands here and the full list of the available commands below:

The following command runs **TFMobileNetSSDFaceDetector** on Livestream: \
`> python startInteraction.py Input TMobFD` 

This runs **Dynamic Mapping** on the default **YinsHeadPoseEstimator**: \
`> python startInteraction.py Mapping YinsHP -mf Dynamic`

And this runs two different **Mapping Functions** on different **Landmark Detectors** which utilize the specified **Face Detectors**: \
`> python startInteraction.py Mapping DlibLD YinsLD -mf Static Dynamic -fd HCasFD TMobFD`

**Please run the following command and find its output for the further instructions:**

```console 
> python startInteraction.py -h
usage: startInteraction.py [-h] [-mf MF [MF ...]] [-fd FD [FD ...]]
                           [-ld LD [LD ...]] [-s SOURCE] [-os Sze Sze]
                           {Input,Mapping} Est [Est ...]

Head-based Interaction Demos

positional arguments:
  {Input,Mapping}       Select a module to run
  Est                   Select InputEstimators to run (Multiple estimators run simultaneously).
                        Find the available estimators below:
                        'CResFD': CV2Res10SSDFaceDetector
                        'DlibFD': DLIBFrontalFaceDetector
                        'HCasFD': HaarCascadeFaceDetector
                        'TMobFD': TFMobileNetSSDFaceDetector
                        'DlibLD': DLIBFacialLandmarkDetector
                        'YinsLD': YinsCNNBasedFacialLandmarkDetector
                        'DlibHP': DLIBHeadPoseEstimator
                        'YinsHP': YinsHeadPoseEstimator
                        'MursHG': MuratcansHeadGazer

optional arguments:
  -h, --help            show this help message and exit
  -mf MF [MF ...], --mappingFunctions MF [MF ...]
                        Select MappingFunctions to apply
                        (Required when using 'Mapping' module,
                        You must pass as many functions as InputEstimators).
                        Find the available MappingFunctions below:
                        'Static' : Static Mapping Function
                        'Dynamic': Dynamic Mapping Function
  -fd FD [FD ...], --faceDetectors FD [FD ...]
                        Select FaceDetectors to change
                        (You must pass as many detectors as InputEstimators).
                        Find the available detectors below:
                        '_'     : Pass to keep the estimator's default FaceDetector
                        'CResFD': CV2Res10SSDFaceDetector
                        'DlibFD': DLIBFrontalFaceDetector
                        'HCasFD': HaarCascadeFaceDetector
                        'TMobFD': TFMobileNetSSDFaceDetector
  -ld LD [LD ...], --landmarkDetectors LD [LD ...]
                        Select LandmarkDetectors to change
                        (You must pass as many detectors as InputEstimators).
                        Find the available detectors below:
                        '_'     : Pass to keep the estimator's default LandmarkDetector
                        'DlibLD': DLIBFacialLandmarkDetector
                        'YinsLD': YinsCNNBasedFacialLandmarkDetector
  -s SOURCE, --source SOURCE
                        Select a source video or camera
                        (You may pass path/to/video or an integer
                        as the camera for livestream, '0' is the default cam).
  -os Sze Sze, --outputSize Sze Sze
                        Set the size of output frame
                        (You may pass 'weight height' like '480 360' or '1080 720',
                        the default is '640 360').
```

## Dependencies

The current implementation runs with the following dependencies which are also listed in `requirements.txt`

- `cycler>=0.10.0`
- `dlib>=19.17.0` 
- `imutils>=0.5.2`
- `matplotlib>=3.0.2`
- `numpy>=1.14.5`
- `opencv_python>=4.0.0.21`
- `tensorflow>=1.13.1`
- `tf_nightly_2.0_preview>=2.0.0.dev20190427`

## License

The current license is [**APACHE LICENSE, VERSION 2.0**](https://www.apache.org/licenses/LICENSE-2.0). But the repository borrows some implementational details from the following repositories and projects which you also have to respect:

- @lincolnhard's [**head-pose-estimation**](https://github.com/lincolnhard/head-pose-estimation)
- @yinguobing's [**head-pose-estimation**](https://github.com/yinguobing/head-pose-estimation)
- @yeephycho's [**tensorflow-face-detection**](https://github.com/yeephycho/tensorflow-face-detection)
- @cmusatyalab's [**OpenFace**](https://github.com/cmusatyalab/openface/)
- [**Dlib**](http://dlib.net/)
- [**OpenCV**](https://github.com/opencv/opencv)
- [**Tensorflow**](https://github.com/tensorflow/tensorflow)

# Credits

I am [**Muratcan Cicek**](muratcancicek.com) (@muratcancicek, the main contributor) who is a Ph.D. student participating in the 
[**Computer Vision Lab**](https://vision.soe.ucsc.edu/) which is led by [**Prof. Roberto Manduchi**](https://users.soe.ucsc.edu/~manduchi/) at the University of California, Santa Cruz. This is still an ongoing project along with my doctoral studies at UCSC, and I have special thanks to my advisor, Prof. Manduchi, who supports my Ph.D. program and encourages me to open-source this work.

## Motivation
I was born with motor impairments (Cerebral Palsy) and unable to use my upper limbs as in the way non-disabled persons can. This affects my interaction with computers and other devices used in daily life. Due to Cerebral Palsy, I also have speech difficulty and a broken accent which voice-based systems cannot recognize. With these limitations, the only way for me to control computers was via Head-based Interaction. In 2010, I met with Muhsin Dogrular and Murad Cihan Özyarar, and they thankfully introduced two following software to me in addition to the home-made equipment they built for me. The software was [**Enable Viacam**](https://eviacam.crea-si.com/index.php) and [**Camera Mouse**](http://www.cameramouse.org/) that allowed me to access the internet, learn to program, and eventually change my entire life. 

To the best of my knowledge, [**Camera Mouse**](http://www.cameramouse.org/) was the first ever software that implements head-based interaction for PC and released by Boston College in the early 2000s, and [**Enable Viacam**](https://eviacam.crea-si.com/index.php) is the software I use to interact with my computer every day for almost 10 years.

I am eternally grateful to [**Cesar Mauri**](https://www.linkedin.com/in/cesar-mauri-loba/?originalSubdomain=es) (@cmauri) for developing [**Enable Viacam**](https://eviacam.crea-si.com/index.php), to [**Prof. Margrit Betke**](http://www.cs.bu.edu/~betke/) (then at Boston College, now at Boston University) who still leads [**Camera Mouse**](http://www.cameramouse.org/) project and to [**Prof. James Gips**](http://www.cs.bc.edu/~gips/) (Boston College) who was the first inventor of the whole idea.  