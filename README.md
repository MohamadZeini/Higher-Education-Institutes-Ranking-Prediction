# Ranking Prediction of Higher Education Institutes using Financial and Expenditure Data
## Supervised Machine Learning

### Mohamad Zeini Jahromi
### Sep 18th, 2017

To see the project refer to `Capstone_project.pdf` and `Capstone_Project.ipynb` files.

The project visual functions are in `visuals.py`.

### Overview

The total number of awards, certificates, and degrees completed each year in post-secondary education institutions have been widely used to evaluate their performances and are an indication of relative success and ranking of these institutions throughout the nation. Many studies have investigated the effects of different parameters (such as financial aid, institutes funds, revenue, expenditures and etc.) on the institutes completion rates. The results of these studies could help institutions to decide how to allocate funds to their segments more effectively and create a well-balanced money flow within their systems. On the other hand, the costs of higher education in the United Sates are high and it’s being considered as an investment and consequently having a knowledge of success rate and ranking of a specific institution would help both students and their families in making the right decision.

This project focuses on studying the relationship between institutions financial aid and expenditure and their respective completion rates (the total number of awards, certificates, and degrees completed). Furthermore, a predictive model will be developed in order to predict the completion rates using the financial aid and expenditure data of institutions. This study is inspired by Udacity's capstone project guidelines and uses the IPEDS dataset ([Integrated Postsecondary Education Data System Delta Cost Project Database](https://nces.ed.gov/ipeds/deltacostproject/), which is based on surveys on finance, enrollment, staffing, completions, and student aid of post-secondary education institutions within the US.

### Install

This project requires **Python 2.7** or higher and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code

To see the project refer to `Capstone_project.pdf` and `Capstone Project.ipynb` files.

The project code are in `Capstone_project.py` and `visuals.py`.

### Run

In a terminal or command window, navigate to the top-level project directory `Higher-Education-Institutes-Ranking-Prediction/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Capstone Project.ipynb
```  
or
```bash
jupyter notebook Capstone_Project.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

This study is inspired by Udacity's capstone project guidelines and uses the IPEDS dataset ([Integrated Postsecondary Education Data System Delta Cost Project Database](https://nces.ed.gov/ipeds/deltacostproject/)), which is based on surveys on finance, enrollment, staffing, completions, and student aid of post-secondary education institutions within the US. 

For this project you need to download the following file:

1) [IPEDS Analytics: Delta Cost Project Database 1987-2012 (CSV) (109 MB)](https://nces.ed.gov/ipeds/deltacostproject/download/IPEDS_Analytics_DCP_87_12_CSV.zip)

And for more information on features of the datasets and its liberies, you can download the following files.

2) [IPEDS Analytics: Delta Cost Project Database 1987-2012 Data File Documentation PDF File (386 KB)](https://nces.ed.gov/ipeds/deltacostproject/download/DCP_Data_File_Documentation_1987_2012.pdf)

3) [Data Dictionary Excel File (890 KB)](https://nces.ed.gov/ipeds/deltacostproject/download/Delta_Data_Dictionary_1987_2012.xls)

4) [Data Mapping File Excel File (299 KB)](https://nces.ed.gov/ipeds/deltacostproject/download/IPEDS_DCP_Database_Mapping_File_87_12.xls)
