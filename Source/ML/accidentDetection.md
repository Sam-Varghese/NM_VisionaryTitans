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

#### Detecting Drowsy Driving even before accidents occur

As per the observations of NHTSA, drowsy drivers don't seem to be apply brakes mostly, and this usually happens mostly between midnight and 6 a.m.

**So in order to detect these situations, we can make the ML model study the braking pattern of vehicles during night time, and they may send alert if the braking pattern goes offtrack.**

## Types of Road Accidents

But a model still won't be able to detect all accident prone situations with 100% accuracy, hence we need to detect accidents whenever they occur, and alert the officers, hospitals, and legal servicers.

### Rear End Collisions

- Happens when a vehicle crashes at the backside of another vehicle that was in front of it.
- These collisions are most common, and accounts for a significant percentage of accidents.

#### Detecting Rear End Collisions

![](https://i.ytimg.com/vi/HD6QGx9d8ow/maxresdefault.jpg)

- Detecting rear end collisions is comparatively difficult provided the crowdy situations that occur at Indian road, which make it difficult even for humans looking at CCTV capture to accurately detect these collisions.
- **Hence a better approach would be to analyze the speed of cars, and calculate the minimum stopping distance required for safely stopping the vehicle.** If the distance left gets less than the stopping distance, then an immediate alert can be sent to officers for swift actions.

### Side Impact Collisions

Side impact collisions happen mostly at intersections where front of a vehicle crashes with side of another vehicle forming a "T" shape.

![alt](https://florinroebig.com/wp-content/uploads/2020/07/side-collision-accident-scaled.jpg)

**In order to detect these kind of situations, the ML model would be required to analyze the speed and direction of vehicles during intersections, and keep a check of stopping distance required to each one of them.**

### Single Vehicle Accidents

![](https://turnto10.com/resources/media2/16x9/full/1015/center/80/b80c85b2-59d8-46c5-bdcc-6aae23b70983-large16x9_NKCRASHVIDEOPKG5PPKG.transfer_frame_69.jpg)

The next most common form of accidents prevailing on roads is Single Vehicle Accident, wherein a particular vehicle somehow goes uncontrolled and collides. This may be the fault of driver, or the vehicle mechanism itself.

**In order to detect Single Vehicle Accidents, we would need to analyze the speed and track being taken by the vehicle while it's being driven. If it goes off track, and reaches to a speed of 0Km/ hr because of collision, then the ML model can confirm the occurrence of a single vehicle accident.**

### Head on Collisions

![](https://nashfranciskato.com/wp-content/uploads/2022/07/Head-on-collision.jpg)

Head on collisions are less frequent, but quite dangerous. Here the vehicles that collide usually drive in the opposite direction, so the impact is significant.

**In order to detect head on collisions, we can track the vehicles, and keep a check if they are all moving in the same direction or not. If vehicles are found moving in the wrong lane, then the program can send an alert to warn even before an accident occurs.**