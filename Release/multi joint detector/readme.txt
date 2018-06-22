This is a template...
How to use it?
-> put your code into main.py
-> start ./run.sh on the server

...now you get a process id to handle the process:
-  process can be monitored using: top -p {process id}
-  process can be killed using: kill {process id}

...the output of your code is written into a logfile (/logs/dd.mm.yyyy-hh:mm:ss.log)

By default GPU 2 is chosen to run the program (you can change that in ./run.sh)
You can pass the filename of another python script if you want to launch it instead (./run.sh myscript.py)

If you want to start your program in the foreground use start.sh instead
