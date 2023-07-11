# Data Analysis of Accidents

In this document we shall be studying the trends of accidents from the downloaded videos.

## Avg Speed of vehicles

It usually happens that before accidents, average speed of vehicles is quite high and it suddently drops down to 0 mostly.

Here's a graph my program generated after recording the average speed of vehicles from the accident video 

<video src="accidents/cyberabad_traffic_incident1.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident1_speed.png)

At 26th minute, the greatest spike that appears here is indeed tha time when incident took place in the video.

But there are some disadvantages to only keep checking the speed for accident detected, because cars might even stop instantly if traffic light turn red. (But what if there are no traffic lights ahead!)

Like here's another average speed analysis of video 

<video src="accidents/cyberabad_traffic_incident2.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident2_speed.png)

The peak here as well is the point where accident took place.

## Vehicle Count

Because of accidents, traffic on the roads increase immediately, which can also be an identifying factor for accidents.

<video src="accidents/cyberabad_traffic_incident1.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident1_vehicle.png)

<video src="accidents/cyberabad_traffic_incident2.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident2_vehicle.png)

Here after 14:52, you can observe that the count of vehicles keeps increasing drastically, which is not in proportion with the traffic seen ever before. This is indeed the time when accident took place in the video.

## People Count

The crowd that gathers at the point where accident took place, is also an important factor for determing the situations of accident.

<video src="accidents/cyberabad_traffic_incident1.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident1_people.png)

<video src="accidents/cyberabad_traffic_incident2.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident2_people.png)

## People, Vehicle Count, Average Speed Combined

<video src="accidents/cyberabad_traffic_incident1.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident1_combine.png)

<video src="accidents/cyberabad_traffic_incident2.mp4" controls title="Title"></video>

![Alt text](plots/cyberabad_traffic_incident2_combine.png)

# Anomaly Detection

[Anomaly detection](https://www.wikiwand.com/en/Anomaly_detection) is the most crucial piece for solving the puzzle of accident detection. This is the algorithm that'll detect accidents/ situations never seen before. This can include highly critical crime situations like suspicious drug dealings, murder cases, and cases of intrusions without access. It has the capability to detect local and even national security situations, never seen before.

![](https://editor.analyticsvidhya.com/uploads/51995anomaly2.png)

## Algorithms for Anomaly Detection

In order to apply Anomaly Detection, we shall be performing some statistical calculations related to [Normal Distribution](https://www.wikiwand.com/en/Normal_distribution)/ Gaussian Distribution.

![](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)

Gaussian Distribution can help us understand the distribution of various trends taking place in the roads.

So, here are the steps that shall be executed in order to detect any anomaly:

1. Figure out the factors that shall be used in order to detect anomaly. Let's name then x1, x2, ... xi in general.

The processes beyond will be explained in terms of a single factor, apply the same procedure to all the factors

2. Calculating **Running Variance** by **Welford's Algorithm**

> Running variance is a statistical measure that quantifies the variability or dispersion of a set of data points as new data points are added incrementally. It provides a way to calculate the variance of a dataset in a streaming or online fashion without the need to store all previous data points. Welford's algorithm along with variance, can help us find out the running mean accurately.

This algorithm mainly keeps track of 3 variables

`n`: Number of data points
`mean`: Mean value
`M2`: Sum of squared differences from the mean

- **Step 1**: Initialize, `n`, `mean` and `M2` with 0
- **Step 2**: For each new datapoint x:
- Increment `n` by 1
- Calculate the difference between x and current `mean` (delta = x - mean)
- Update the mean by adding delta/n to the current mean.
- Calculate the new difference between x and the updated mean (delta2 = x - mean)
- Update M2 by adding delta*delta2 to the current M2.
- **Step 3**: Calculate the running variance:
- If n < 2, then data is insufficient, so ans is NaN
- Otherwise divide M2 by n-1 to get running variance.
- For the **next data point**: Continue from step 2.

3. When you get the next data point, calculate the probablity of the point lying in that same Gaussian Distribution with this formula

![](https://www.onlinemathlearning.com/image-files/normal-distribution-formula.png)

4. Now set a threshold for a the probablity to be considered as an anomaly. Keep it low.
5. If the data point comes out to be an anomaly, don't get the next variance and mean using that point.