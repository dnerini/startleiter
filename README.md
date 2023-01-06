Welcome to Starleiter! In this project, I used data analysis and machine learning techniques
to explore the relationship between the atmospheric conditions and paragliding.

## Project Overview

Startleiter is a recommendation system for paragliding pilots. Based on the nearest and
most recently available radio-sounding, it computes the probability of flying on the
current day, as well as the expected maximum flying height and distance.

The prediction model, a one-dimensional convolutional neural network (1D CNN),
is trained on radio-sounding data from [UWYO](http://weather.uwyo.edu/upperair/sounding.html)
and flight reports from [XContest](https://www.xcontest.org/world/en/). 
Startleiter also includes an explainability plot based on [SHAP](https://github.com/slundberg/shap)
to gain insights on the output of the machine learning model, for example:

![](https://user-images.githubusercontent.com/11967971/178354681-50b8b017-b007-4dd0-99e9-1c5f30e789cb.png)

## Project Components

The project consists of the following components:

- Data extraction.
- [Data exploration and visualization](https://dnerini.github.io/startleiter/statistics.html).
- Data preprocessing and feature engineering.
- Model training and evaluation.
- Real-time prediction and monitoring.

## Credits and Sources

- Flight reports: [XContest](https://www.xcontest.org/)
- Atmospheric soundings: [University of Wyoming](https://weather.uwyo.edu/upperair/sounding.html)
- GFS forecast data: [NOAA](https://rucsoundings.noaa.gov/)
- Explainability score: [SHAP](https://github.com/slundberg/shap)
- SkewT plot: [MetPy](https://unidata.github.io/MetPy/latest/)
