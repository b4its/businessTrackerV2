=======================
# Business Tracker V2
=======================

## how to run this models at you're environtment
- pip install packages
```bash
    pip install -r requirements.txt
```

- generate json data
```bash
    python generateDataset.py
    python textGeneratev1.py
    python textGeneratev2.py
```
wait it until finished

- train the models 
```bash
    python train.py
```
check the result in ML Flow:
```bash
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000
```
then open at browser and type:
```bash
    localhost:5000
```

- try model
```bash
    python tesMFlow.py
```
type exit, quit, or keluar for close the program.