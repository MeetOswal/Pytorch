"""
steps for heroku:
$ heroku login - i
-> enter credentials and passowrd
$ heroku create flask-pytorch-app
--> change paths from 'torch_utils' to 'app.torch_utils' and 'mnist_ffn.pth' to 'app.mnist_ffn.pth'
--> this file will be outside app folder and import app.main instead of main
--> Procfile too outside app folder
--> create requirements.txt (heroku does not support GPU)
$ git init
--> create file .gitignore

$ heroku git:remote - a flaskpy-pytorch-app
$ git add .
$ git comit -m 'initial comit'
$ git push heroku master
"""

# from main import app
