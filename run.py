rint("hello world!")
from browser import document, window, alert, aio
env = window.env

######################
# Start Learning Here
######################

def getNumberOfStepsFromUser():
    userInput = input("Enter number of steps")
    print(userInput)
    return int(userInput)

def MoveManyStepsForward(numberOfSteps):
    for everySingleNumberInTheRange in range(numberOfSteps):
        env.step(0)

totalNumberOfSteps = getNumberOfStepsFromUser()     
MoveManyStepsForward(totalNumberOfSteps)


#######################
## End Learning Here
#######################