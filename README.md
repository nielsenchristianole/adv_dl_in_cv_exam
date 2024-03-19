# adv_dl_in_cv_exam


## Get data with dvc
To get the data, run the following:
```bash
dvc pull
```

> [!NOTE]  
> You might be redirected to the browser to authenticate with your Google account. This is because the dataset is stored in Google Drive.


## Annotation
To annotate the dataset, run the following:
```bash
make annotate
```
Then write your name (this will be the name of the annotations file) and press enter.

Now you can annotate the dataset! Just press either <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd> or <kbd>4</kbd>. To exit the annotation process, press <kbd>esc</kbd>.

The annotations `.csv` file will be saved on the `annotations_path` specified in the `config.yaml` file. 
The images that we are annotating are loaded from the `raw_path` specified in the `config.yaml` file.'

