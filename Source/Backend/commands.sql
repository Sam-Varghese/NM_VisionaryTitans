create database NM_VisionaryTitans;
create table IncidentsDatabase(
ID varchar(20),
 category varchar(200),
 tim datetime,
 location varchar(100),
 severity int(2),
 Medical_Aid varchar(5),
 Legal_Aid varchar(5),
 Police_Aid varchar(5)
);
create table ProgrammaticActions(
Incident_id varchar(20),
Medical_Aid varchar(5),
Legal_Aid varchar(5),
Police_Aid varchar(5));

create table ManualActions(
Incident_id varchar(20),
Medical_Aid varchar(5),
Legal_Aid varchar(5),
Police_Aid varchar(5));