# Startleiter

Startleiter is a recommendation system for paragliding pilots. Based on the nearest and most recently available radio-sounding, it computes the probability of flying on the current day, as well as the expected maximum flying height and distance. 

The machine-learning model, a one-dimensional convolutional neural network (1D CNN), is trained on sounding data retrieved from the [UWYO archive](http://weather.uwyo.edu/upperair/sounding.html) and flight data from [XContest](https://www.xcontest.org/world/en/). Startleiter also includes an explainability plot based on the [SHAP framework](https://github.com/slundberg/shap) that can be used to gain insights on the output of the machine learning model, for example:

![image](https://user-images.githubusercontent.com/11967971/178354681-50b8b017-b007-4dd0-99e9-1c5f30e789cb.png)

The current alpha version is available for site of Cimetta in southern Switzerland only, see https://startleiter.herokuapp.com/.

## Credits

- Training flight data: [XContest](https://www.xcontest.org/)
- Atmospheric soundings: [University of Wyoming](https://weather.uwyo.edu/upperair/sounding.html)
- GFS data: [NOAA](https://rucsoundings.noaa.gov/)
- Explainability score: [SHAP](https://github.com/slundberg/shap)
- SkewT plot: [MetPy](https://unidata.github.io/MetPy/latest/)
