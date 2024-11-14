'''
Path: bdENS/repo/utils/LumberJack.py

Boundary-Detecting ENN Project
Logging Utility
Author: James R. Elder
Institution: UTSW

DOO: 11-07-2024
LU: 11-07-2024

Reference: bdENS/release/utils/LumberJack.py

- Python 3.12.2
- bdENSenv
- 3.12.x-anaconda
- linux-gnu (BioHPC)
'''

# Imports
import time
import os
import atexit
import matplotlib.pyplot as plt

lineBreakItem = "-"
lineBreakLength = 80
lineBreaker = lineBreakItem * lineBreakLength

# Logging manager class
class LoggingManager:
    def __init__(self,  subprojID="debug", initializer="simple", logFile="default", echoMe=True):
        """Class to manage logging. Initialize with logFile name and echoMe to print to console."""
        self.exitCode = -1 # Default exit code
        
        # Directory dictionary
        self.subprojID = subprojID
        self.curDir = os.getcwd() 
        self.mediaDir = self.curDir + "/media"
        self.loggingDir = self.curDir + "/.logging"
        self.checkDir()

        # Time declarations
        self.today = time.strftime("%m-%d-%Y")
        self.startTimens = time.time()
        self.startTime = time.strftime("%H:%M:%S")

     
        self.exitCode = 0 # Successful initialization

        atexit.register(self.__exit__) # Register exit function

    def checkDir(self):
        _dirDict = {
            "media": self.mediaDir,
            "logging": self.loggingDir
        }

        for key, value in _dirDict.items():
            if not os.path.exists(value):
                os.makedirs(value)
                self.writeLog("Creating {} directory: {}".format(key, value))

    def saveFig(self, fig, figName, figDir="media", figType="png"):
        # Save figure to media directory
        self.writeLog("Saving figure: {}".format(figName))
        fig.savefig("{}/{}".format(self.dirDict[figDir], figName), format=figType)

    def elapsedTime(self):
        # Print time elapsed since start
        self.writeLog("Time elapsed: {}".format(time.time() - self.startTimens))

    def __exit__(self):
        self.endTimens = time.time()
        self.endTime = time.strftime("%H:%M:%S")     
        print("LoggingManager exiting with code: {}".format(self.exitCode))

        
