cv2 does not work on arm64
first switch to x68

uname -m
If the CPU architecture is the default, it should return “arm64”.

Here is the KEY! Then, change the CPU architecture by putting the following command!

arch -x86_64 zsh
Next, create a virtual environment on Terminal (iTerm).


python3 -m venv cv2-env
Then, activate the virtual environment.

cd cv2-env
source bin/activate

cd cv2-env
source bin/activate
After everything is set up, install the OpenCV on M1 Mac! You can simply follow the command below, and more detail of the OpenCV using pip is from here.

pip install opencv-python

pip install opencv-python
Numpy is also installed during installing opencv-python. When you type “pip list” on Terminal, you should be able to see the image like below.

