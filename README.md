# Disaster Response Pipeline Project

You can find [the live site on render.com](https://disaster-repository-webpage.onrender.com/)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database 
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
		- Alternately you can [download the classifier data directly from this link and use it.](https://drive.proton.me/urls/QSME3S6WRR#tznjH1e2KwgX)

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to  http://127.0.0.1:3001

### Important Files:
1. **process_data.py** : Used to load raw messages data, clean the data, and finally store as a database
2. **train_classifier.py** : Used to load database and train a classifier using database
3. **run.py** : Script used to generate/manage frontend of the site.
4. *master.html* : Main page of the site
5. *go.html* : Returns predicted message category based on user input
6. *ETL Pipeline Preparation.html* and *ML Pipeline Preparation.html* : Jupyter notebook detailing preparation of pipelines for ETL and ML pipelines respectively. The jupyter nb's can be viewed directly in the *jupyter_nb_repository* directory

## Example Output
![webpage](https://github.com/AayushSo/disaster-repository-webpage/blob/main/project_test_msg.png?raw=True)



