# Schedule Helper



## What is Scheduel Helper ?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Schedule Helper is one simple system which would recommend you suitable future schedule
based on the previous record .



<br><br><br>
## How to Use ?
* **Step 1 : Train the model**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Run `python3 schedule_helper.py -d`
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This command would train the model. 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `-d` means downloding the NLTK tools. The action of downloading is required for only
one time, so you can remove `-d` afterwards.

* **Step 2 : Let model recommend you the scheduel**
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Run `python3 schedule_prediction.py`
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After done with training the model, you're allowed to access the recommendation from the
 trained model.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After run the command, it would show some interactive printing, please follow it, and you will 
retrieve the recommended schedule.



<br><br><br>
## Settings
* **Language**
<br>&nbsp; Python3.5
* **Data**
<br>&nbsp;&nbsp;&nbsp;&nbsp; The system need historical(previous) record. For the sake of convenience, we provide
 the previous record data already in "./data/data_schedule.xlsx" . 
<br>&nbsp;&nbsp;&nbsp;&nbsp; If you're interested, feel free 
 to check it out and know the format.
 
 
 
 <br><br><br>
## Documents
* **Schedule Helper Report in .pdf**
<br> [Schedule Helper Report](https://drive.google.com/file/d/1TIIhiekalnLPansZabJq8Kbk-wlQ3KZc/view?usp=sharing)