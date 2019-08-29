Installation instructions:
1. Install Python 3.7.0 if necessary.
2. Create a new Python virtual environment named 've' in the 'chapter_13_examples' directory:  
    ```$ python3 -m venv ve```
3. Activate the virtual environment:  
    ```$ source ve/bin/activate```
4. Upgrade pip:  
    ```(ve)$ pip install --upgrade pip```
5. Install required packages:  
    ```(ve)$ pip install -r requirements.txt```

Instructions for running scripts:
1. Activate the virtual environment:  
    ```$ source ve/bin/activate```
2. Run the desired scripts:  
    ```(ve)$ python example_13_1.py```  
    ```(ve)$ python figure_13_1.py --confidence_intervals```  
    ```(ve)$ python figure_13_2.py --confidence_intervals```
3. Deactivate the virtual environment when finished:  
    ```(ve)$ deactivate```