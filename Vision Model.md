# Vision Model
## How to run:
This script can be run by executing `` python visoinModel.py``
## required dependencies: 
* pytorch
* cv2
* ultralytics, for loading YOLO without torch
* openAI api 
* YanAPI


## What this is:
This script intends to make Yanshee see the scene and say outloud what it sees.
## How it works:
YanShee says the names of the objects it sees only once and does so again when it moves.
This runs a while loop in which the robot takes a picture through it's camera then it's run through Yolo Model after that the robot is called through its API to say them names outloud.
If YanShee sees a person it tries to identify that person, if YanShee knows that person it lights in green or red if YanShee doesn't, in the case of seeing no person at all it lights blue.
This inner code of the while loop is conditioned by two flags
* ### saw perosn: 
    only runs when the scene has included appearnce or disappearance of a person. This is to avoid calling the API to light the eyes and button each time.

* ### scene changed: 
    only runs the model and says the names of the objects when YanShee moves so that it doesn't keep repeating itself and processing the same image.

What rules the scene changed flag is the summation of the Z Y X components of YanShee's gyroscope, this is a naive approach and is to be improved but currently it's sufficient enough :)

## What to look out for:
We have tried to make Yanshee bow when he sees a person, this has caused it's gyros to detect motion and think that the scene has changed which makes it see a person again and bow again and this keeps for ever. This can be avoided by simply sleeping until the move is done, the time needed for the move to be performed is returned from `` start_play_motion `` function. Note that this is a recurrent theme in the rest of the scripts, we need to wait for YanShee to stop moving or speaking so we can continue the rest of the script without creating bugs. 

# Integration with task three:
This task does most of the job of Task Three so we edited the code so it can tell the description of the image instead of saying the object names.
## How? 
In the ``scene changed`` routine we compress and downsize the frame and turn it to base64 so it can be easily sent to GPT API which returns the description of the image. The code can be changed so Yanshee uses its tts to say the descriptino instead of the object names.
## On down sizing: 
We have tried multiple sizes and file compression extentions and through testing found that about 6500 base64 on average is good engough to send quickly and for GPT to handle it.

# Real time chat: 
The function for real time chat waits for a sound to pass a threshold so it can start recording, it keeps recording until it senses a silience of 2 seconds after that the record is saved to a ``.wav`` file and sent to ``whisper`` api for transcribtion, the transcribtion is sent to GPT API for processing which returns only a type of movement.
## how to run: 
This script can be run by executing `` python yan_intel.py``
