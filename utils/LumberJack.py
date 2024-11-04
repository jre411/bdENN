# Path: release/utils/LumberJack.py

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
    def __init__(self,  subProjName="debug", initializer="simple", logFile="default", echoMe=True):
        """Class to manage logging. Initialize with logFile name and echoMe to print to console."""
        self.exitCode = -1 # Default exit code
        self.subProjName = subProjName
        self.lineBreaker = lineBreaker

        # Time declarations
        self.today = time.strftime("%m-%d-%Y")
        self.startTimens = time.time()
        self.startTime = time.strftime("%H:%M:%S")
        
        # Path declarations
        self.curDir = os.getcwd() # Current working directory
        if logFile == "default":
            self.logFile = "{}.log".format(self.today)
        else:
            self.logFile = logFile

        self.dirDict = {
            "logging": "{}/.logging/{}/{}".format(self.curDir, self.subProjName, self.today), # Ensure log directory is first in order to be initialized for logging 
            "outputs": "{}/outputs/{}/{}".format(self.curDir, self.subProjName, self.today),
            "media": "{}/media/{}/{}".format(self.curDir, self.subProjName, self.today),
            "inputs": "{}/inputs".format(self.curDir),
            "models": "{}/models".format(self.curDir),
        }        

        # Printing declarations
        self.echoMe = echoMe
        self.printBreakItem = "-"
        self.printBreakLength = 80
        self.printBreak = self.printBreakItem * self.printBreakLength        

        # Initialization message
        print(self.printBreak)
        print("LogMan cometh")   
        print(self.printBreak)     

        self.exitCode = 0 # Successful initialization

        if initializer == "simple":
            self.rev()
            self.writeLog()
            self.writeLog("Simple initialization", headMe=True)
            self.ignite()

        atexit.register(self.__exit__) # Register exit function

    def saveFig(self, fig, figName, figDir="media", figType="png"):
        # Save figure to media directory
        self.writeLog("Saving figure: {}".format(figName))
        fig.savefig("{}/{}".format(self.dirDict[figDir], figName), format=figType)

    def logParams(self, dictParams, title="Parameters"):
        # Log parameters
        self.writeLog("Logging parameters: {}".format(title), headMe=True)
        for key, value in dictParams.items():
            self.writeLog("{}: {}".format(key, value))
        self.writeLog()

    def ignite(self):
        # Ignite the engine (initialize logging once all parameters are set)
        self.writeLog("Igniting {}".format(self.subProjName), headMe=True)
        self.writeLog()
 

    def rev(self):
        # Rev the engine (initialize logging once all parameters are set)
        self.checkPath()
        self.writeLog()
        self.writeLog("Revving engine", headMe=True)
        self.touchLog()

    def checkMate(self):
        # Checkpoint for logging
        self.writeLog("Check, mate!", headMe=True)
        if hasattr(self, 'checkPoint'):
            self.timeSinceCheck = time.time() - self.checkPoint
            self.writeLog("Time since last check: {}".format(self.timeSinceCheck))
        self.checkPoint = time.time()

    def checkPath(self):
        # Check for logging directory in current directory
        for key, value in self.dirDict.items():
            if not os.path.exists(value):
                os.makedirs(value)
                self.writeLog("Creating {} directory: {}".format(key, value))


    def writeLog(self, message="", padMe=False, headMe=False):
        # Write to log file
        self.log = open("{}/{}".format(self.dirDict["logging"], self.logFile), 'a')
        if message == "": # Empty message, used for padding
            self.log.write(self.printBreak + "\n")
            self.log.close() 
            return # Exit function
        if headMe: # Header message, decorate with asterisks
            self.log.write("***\t{}\t***".format(message) + "\n")
        else: # Regular message
            self.log.write(message + "\n")

        if self.echoMe: # Print to console
            print("~ {}".format(message))

        if padMe: # Add padding
            self.log.write(self.printBreak + "\n")

        self.log.close()

    def elapsedTime(self):
        # Print time elapsed since start
        self.writeLog("Time elapsed: {}".format(time.time() - self.startTimens))

    def readLog(self):
        # Read log file
        self.log = open(self.breadCrumb, 'r')
        _logContents = self.log.read()
        self.log.close()
        return _logContents # Return log contents

    def touchLog(self, existsOkay=True):
        # Touch log file, add header
        if existsOkay: # Append to existing log file
            self.log = open("{}/{}".format(self.dirDict["logging"], self.logFile), 'a')
        else: # Create new log file
            self.log = open("{}/{}".format(self.dirDict["logging"], self.logFile), 'w')
        # Add header
        self.writeLog(self.lineBreaker)
        self.writeLog("Logging file: {}".format(self.logFile))
        self.writeLog("Date: {}".format(self.today))
        self.writeLog("Start Time: {}".format(self.startTime))
        self.writeLog("Working in: {}".format(self.curDir))
        self.writeLog("")

    def __exit__(self):
        self.endTimens = time.time()
        self.endTime = time.strftime("%H:%M:%S")        
        
        self.writeLog(self.lineBreaker)
        self.writeLog("Concluding log", headMe=True)
        self.writeLog("")
        self.writeLog("End Time: {}".format(self.endTime))

        if hasattr(self, 'checkPoint'):
            self.timeSinceCheck = time.time() - self.checkPoint
            self.writeLog("Time since last check: {}".format(self.timeSinceCheck))

        self.writeLog("Time elapsed: {}".format(self.endTimens - self.startTimens))
        self.writeLog("Exit Code: {}".format(self.exitCode), padMe=True)
        self.writeLog()

    def escalateLogMan(self, nSteps=1):
        if nSteps > 0:
            self.writeLog("Escalating logMan {} step(s)".format(nSteps), headMe=True)
            for i in range(nSteps):
                self.writeLog("Navigating to parent directory of {}".format(self.curDir))
                self.curDir = os.path.dirname(self.curDir)
                os.chdir(self.curDir)
                self.writeLog("New working directory: {}".format(self.curDir))
            self.writeLog("LogMan escalated", headMe=True, padMe=True)

        