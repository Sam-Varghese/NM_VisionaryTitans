```mermaid
---
title: Flow Chart
---
graph TD;
    camInp[Camera input] --> model[SSD+MobileNetV2 Model]
    camInp --> anom_det[Anomaly Detection Model]
    model --> objects[Detect objects and instances]

    anom_det-->|If Detected|al_med[Alert: Medium]
    anom_det-->|Not Detected|ign[Ignore] 
    
    objects --> crimes[Crimes]
    objects --> med[Medical Emergency]

    crimes --> burglary[Burglary]
    crimes --> intrusion[Intrusion]

    burglary-->burg_ft[Features]
    burg_ft --> masked[Masked Person]
    burg_ft --> run[Running]

    masked --> mask_alert[Alert: Low]
    run --> run_alert[Alert: Low]

    masked & run --> mask_run_alert[Alert: High]

    intrusion --> intr_ft[Features]
    intr_ft --> person_detect[Person Detection]

    person_detect --> person_detect_alert[Alert: Medium]

    med --> accident[Accidents]
    med --> collapse[Collapse]

    accident --> acc_ft[Features]
    acc_ft --> crash[Crash Detection]
    acc_ft --> fire[Fire]

    crash --> crash_alert[Alert: High]
    fire --> fire_alert[Alert: High]

    crash & fire --> crash_fire_alert[Alert: High]

    collapse --> coll_ft[Features]
    coll_ft --> pers_grnd[Person on Grnd]

    pers_grnd --> pers_grnd_alert[Alert: High]
```