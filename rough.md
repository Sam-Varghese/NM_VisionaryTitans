```mermaid

---
title: Advance CCTV Analytics
---
classDiagram

    Incident <|-- Crime
    Crime <|-- Burglary
    Crime <|-- Intrusion
    Incident <|-- MedicalIncidents
    MedicalIncidents<|-- Accidents

    class Incident{
        #String ID
        +String incidentType
        +String alertType
        +String incidentTime
        +String incidentLocation
        +bool databaseEntryMade
        +bool alertSent
        +String alertLevel

        Incident()
        +String getIncidentID()
        Destructor()
    }

    class Anomaly{
        #String ID
        +String alertType
        +String anomalyTime
        +bool databaseEntryMade
        +String incidentLocation
        +String alertLevel
        +bool alertSent

        Anomaly()
        +String getAnomalyID()
        +bool sendAnomalyAlert()
        Destructor()
    }

    class Crime{
        #String crimeName
    }

    class Burglary{
        +bool maskedPerson
        +bool running

        +bool sendBurglaryAlert()
    }

    class Intrusion{
        +bool humanDetected
        +bool sendIntrusionAlert()
    }

    class MedicalIncidents{
        #String medicalIncidentName
    }

    class Accidents{
        #bool crashDetected
        #bool fireDetected
        +bool sendAccidentAlert()
    }
```
