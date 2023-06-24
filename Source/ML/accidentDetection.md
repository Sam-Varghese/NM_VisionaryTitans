# Accident Detection

![](https://images.indianexpress.com/2017/04/kolkata-road-accident_759_yt.jpg)

According to [National Highway Traffic Safety Administration](https://www.nhtsa.gov/); 42,939 lives were lost on U.S. roads.

Detection of accidents is not an easy task, especially in India where majority of road gets so crowded that it becomes next to impossible to detect accidents.

Take the example of this situation:

![](https://static.toiimg.com/thumb/msid-65737489,width-400,resizemode-4/65737489.jpg)

Now in the image shown above, even if 2 card moving in the same direction collide, it becomes difficult even for humans to detect this kind of collision because of the intense crowd.

hence there needs to be a proper study of types and causes of accidents prevailing on roads.

## Causes of Road Accidents

How about detecting the situations prone to accidents, so we could avoid them even before they occur...

![alt](https://cdn.pixabay.com/photo/2017/01/20/20/24/car-accident-1995852_1280.png)

### Drunk Driving Fatalities

According to [National Highway Traffic Safety Administration](https://www.nhtsa.gov/); 13,384 lives were lost because of drunk and drive cases.

According to the [Hyderabad Traffic Police](https://www.htp.gov.in/Drunken.html), drunken drivers lack:

- Alertness in perceiving a danger in the road and reacting to it quickly.
- Accuracy of vision.
- A broad range of vision to take note of events taking place on either side of the road without turning head in either direction.
- Ability to perceive distance between two moving objects and their relative position in space.
- Capacity to distinguish accurately between three traffic light colors e.g. green, amber and red.
- Ability to drive the vehicle safely during night hours.
- Ability to recover the glare effect quickly.

#### Catching Drunk Driving even before accidents occur

- **Ability to drive the vehicle safely during night hours**: If drunken drivers face difficulties in driving in night, then this implies they'll loose track of lanes more often and over-speed frequently.
- **In order to detect this kind of situation, we can make algorithms to track cars at every instant and keep an eye on their movements. If they're found frequently changing lanes, or in some weird behavior, then we can send an alert.**
- **We can also observe their response to traffic signals. Like if they keep moving even during red lights, we may start sending alerts.**

### Speeding Related Fatalities

According to [National Highway Traffic Safety Administration](https://www.nhtsa.gov/); 12,330 people died due to over-speeding in 2021. It contributed to 29% of all traffic fatalities.

#### Catching Over Speeding even before accidents occur

**The YOLOv7 detection algorithm applied by my team can help us get an estimate of speed of each and every car on road, thus it's combination with ANPR (Automatic Number Plate Recognition) technology can help us in catching drivers who over-speed.**

### Drowsy Driving

According to NHTSA, in [2017 91,000 crashes involved drowsy driving](https://www.nhtsa.gov/risky-driving/drowsy-driving). They also reported the following observations:

- Occur most frequently between midnight and 6 a.m., or in the late afternoon. At both times of the day, people experience dips in their circadian rhythm—the human body’s internal clock that regulates sleep;
- Often involve only a single driver (and no passengers) running off the road at a high rate of speed with no evidence of braking; and
- Frequently occur on rural roads and highways.

### Side Impact Collisions

Although it's more of a type of accident, still I've mentioned it here because this situation can be detected even before the occurrence of an accident.

Side impact collisions happen mostly at intersections where front of a vehicle crashes with side of another vehicle forming a "T" shape.

![alt](https://www.researchgate.net/publication/252014161/figure/fig1/AS:298029216223241@1448067110139/Side-collision-scenario-The-target-vehicle-is-closing-in-with-a-trajectory-visualized-in.png)

#### Detecting Drowsy Driving even before accidents occur

As per the observations of NHTSA, drowsy drivers don't seem to be apply brakes mostly, and this usually happens mostly between midnight and 6 a.m.

**So in order to detect these situations, we can make the ML model study the braking pattern of vehicles during night time, and they may send alert if the braking pattern goes offtrack.**

## Types of Road Accidents

But a model still won't be able to detect all accident prone situations with 100% accuracy, hence we need to detect accidents whenever they occur, and alert the officers, hospitals, and legal servicers.

### 