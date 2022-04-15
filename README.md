# Startleiter

Startleiter is a recommendation system for paragliding pilots. Based on the nearest and most recently available radio-sounding, it computes the probability of flying on the current day, as well as the expected maximum height and flying distance. 

The model, a one-dimentional convolutional neural network (1D CNN), was trained on upper-air observation data from [University of Wyoming](http://weather.uwyo.edu/upperair/sounding.html) and fyling data from [XContest](https://www.xcontest.org/world/en/). Startleiter also includes an explanability plot based on the [SHAP](https://github.com/slundberg/shap) that can be used to gain insights on the output of the machine learning model, for example:

![image](https://user-images.githubusercontent.com/11967971/163541253-0b9d5bdd-9300-4738-b4a0-8a947c3813ed.png)

The current alpha version is available for site of Cimetta (southern Switzerland) only, see https://startleiter.herokuapp.com/. 
