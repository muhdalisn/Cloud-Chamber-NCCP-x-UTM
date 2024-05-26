# Machine Learning Based Particles Tracks Detection And Classification In Cloud Chamber

ABSTRACT: Particle Physics is one of the fields in physics that addresses the important question about life and the universe. It questions what constitutes the atoms and what made up this huge universe. This field studies about the fundamental subatomic particles, including both matter, antimatter and the carrier particles of the fundamental interactions as described by quantum field theory. In order to observe and detect the ionising particles, we used a cloud chamber. It is one of the oldest particles and radiation detectors. Interaction of radiations with the saturated vapour inside the cloud chamber creates tracks of different direction, length and weight. A cloud chamber size of aquarium, dry ice and 99% IPA isopropyl alcohol were then used in these projects.  One hour of recording of movement of tracks in the cloud chamber was taken by placing a camera on top of the cloud chamber. For the analysis of the video, we used Python’s OpenCV and NumPy libraries to detect the tracks. Two algorithms were computed to analyse the video, in which we filter the background from the frame, then we proceed with the masking process to remove the artefacts and simplify grouping of the remaining pixels. The nearby grouping was merged and matched for consecutive frames. Finally, we draw contours along the edge of the track and put arrows of direction for the particle tracks. This study works as a foundation and guide in tracking particle tracks and hopefully insight and sparks many people's interest in the developing webs and apps for object detection in the future. 

Algorithm & Analysis Tools:

Python:
1.1 Introduction to Python 

Python is a high level language with general purpose programming. It is widely used in many areas of expertise and suitable in utilising data science and data analyst programs. Python language created by Guido van Rossum in the late 1980s. This language places strong emphasis on code readability and simplicity, making it possible for programmers to develop applications rapidly. 

Python is widely used for web development, software development, mathematics and system scripting. It has multiple purposes where it can be used to create web applications, it can be used alongside software to create workflows, it can connect to database systems. It can also read and modify files. It is good at handling big data and performing complex mathematics.

One of the reasons we chose Python in our project for algorithm was due to the fact that it can work on different platforms (Windows, Mac OS, Linux, Raspberry Pi, etc). Python has a simple syntax similar to the English language. The best part is, it has syntax that allows developers to write programs with fewer lines than some other programming languages. Other than that, codes that we write in Python have to be interpreted by a special program known as Python interpreter, which we’ll have to install before we can code, test and execute our Python programs. It means that code can be executed as soon as it is written. This also means that prototyping can be very quick. In addition, Python comes with an extensive collection of third party resources that extend the capabilities of the language. As such, Python can be used for a large variety of tasks. 

Compared to other high level programming languages, such as C. C++ and java. They're all quite similar to one another. What differentiate them mainly in the syntax, the libraries available and the way we access those libraries. So, the library is simply a collection of resources and pre-written codes that we can use when we write our programs. If you already learn one language well, you can simply learn a new language in a short time frame compared to the time it took for you to learn the first language. 

Before we proceed in doing our project for tracking and object detection, we must familiarise ourselves with this language. For beginning, before we can write our Python program, we must download the appropriate interpreter for our computers. In this project, we'll be using Python 3. To install the interpreter for Python 3, we can go to the Python website or https://www.python.org/downloads/. The correct and latest version should be at the top of the webpage. If using different types of OS like Linux or Mac, there are also other versions provided in the website. Click the version for Python 3 and the software will start downloading. 

Scroll down towards the end of the page and you’ll see a table listing various installers that are suitable and meet the criteria that you intended. Choose the correct installer for your laptop or PC. In choosing the installer, it depends on two factors:
The operating system (Windows, Mac OS, or Linux) and
The processor (32 bit vs 64 bit) that you are using. 

For instance, if you are using a 64-bit Windows computer, you will likely be using the latest version which is Python 3.10.8. Just click on the button below the ‘Download the latest version of Python’ on the website. If you download and run the wrong installer, it will inform you by giving an error message and the interpreter will not install. Just download the correct and latest version installer, and we are good to go. Once we have successfully installed the interpreter, then we are ready to start coding in Python. 

Python modules or libraries

1.2.1 OpenCV

Computer Vision is a method through which we can comprehend how videos and images are stored, how to change them, and how to extract data from them. Computer Vision is the base or mostly used for Artificial Intelligence. Computer Vision was used in development of self-driving cars, robotics and also in photo correction apps.

OpenCV (Open Source Computer Vision Library) is an open-source library for computer vision, machine learning and image processing and it plays a huge role in real-time operation which is crucial to today’s systems. It was developed by Intel, it was later supported by Willow Garage then Itseez. In using these modules, one can extract or process images and videos to classify objects, tracking objects, faces, licence plates, or even handwriting of a human. When compiled with various libraries, such as NumPy, Python is capable of processing the OpenCV array structure for analysis. To identify an image pattern and its various features we use vector space and perform mathematical operations on the features. There are many applications which are solved using OpenCV, part of them are face recognition, vehicle counting on highways along with their speeds, object recognition, medical image analysis and driver-less car navigation and control. 






 
