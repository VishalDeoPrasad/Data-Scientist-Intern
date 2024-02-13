# Data-Scientist-Intern

## Agenda (12-02-2024)
1. KNN Imputation
2. Feature important using decision tree
3. Regulization also help us to understand feature important.
![image](https://github.com/VishalDeoPrasad/Data-Scientist-Intern/assets/44454324/4c8f71f4-d898-4413-835a-86423c523c9a)
4. Hadoop
   + HFES - use to store the data in multiple partation
   + Map Reduce - help you to Process the data
   
#### Spark
+ Spark - new way to process data is spark not map reduce
+ Spark - also help us to analyse the data
+ best (HDFS + Spark)
+ HDFS - HDFS is use to store the large data
+ SparkSQL 
+ MLlibs for spark(similar to sklearn library)
+ Spark Streaming
+ you can apply spark on (csv, json, etc)

__Important notes__
 1. train and text score is used to find the overfitting and underfitting.
 2. if data become very big or data are in different places then doing with the pandas will be very difficult.

### Data Enginerring
![image](https://github.com/VishalDeoPrasad/Data-Scientist-Intern/assets/44454324/808241ad-b07f-4d4f-9b1e-2516db91b326)

#### HDFS
to take out the data from HDFS we need to use
- Apache SQOOP
- Apache Fluma
 
DA, ML, DL, NLP, CV, LLM
Spark
Cloud

#### from where data can be generated
1. IOT sensor
1. Google Form
1. Web Scrapping
1. more...

#### How to analyse and handle the big data
1. Map Reduce
2. Apache Spark

#### How to extract the data from HDFS
1. ApacheSQOOP
2. Apache Flume

#### Data Warehouse
1. AWS Redshift
1. Snowflack

#### dependence management(orchastrating)
1. Apache Airflow

#### Tools to take out data from data warehouse
1. SQL

Date - 13-02-2024
## Advance Stats DA
1. Descriptive Stats
   def : help us to describe the historial data.
   - Allow us to build reports(Summary of data)
   - Visulization
   - help to build Visual Dashboard

2. Infrential Stats
   - Estimation, appoximation
   Q. need of estimation and approximating something
   + A. Data collection is very expensive so we estimate the data
   + example 1: What is avg age of an indian citizen?(pandas, spark can easly find the mean)
   + example 2: total population of india just an estimate. 
   becz counting all people is very difficult



### sampling
- if you don't have enough money to collect the data we do sampling.
- we randomly collecting the data.
- it is not possible to collect the data of 1.4 billion of data, it is depend upon the budget of the company.

Q. what is the avg salary of the population?
Q. what is the avg age of the populaiton?
- do a data collection at very small scale becz
   + time consuming
   + collecting data is very expensive.


### Estimation
- Point estimation - not so good 
- **Central limit theorum** is use to estimate, or make better inference/apporximation/estimation on the entire population using a small random sample.
