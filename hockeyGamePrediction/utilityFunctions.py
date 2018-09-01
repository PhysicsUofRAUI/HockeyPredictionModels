# This file will contain all the python functions needed to take the txt files included
# and make them into the format that will be used for the neural network

# First make a three dimensional array. The teams and the outcome included. Everything
# else should be computable from that

teams = ["Toronto Maple Leafs", "Montreal Canadiens", "Vancouver Canucks", "Boston Bruins"
        , "Edmonton Oilers", "Chicago Blackhawks", "New York Rangers", "Pittsburgh Penguins"
        , "Washington Capitals", "Detroit Red Wings", "Philadelphia Flyers", "Vegas Golden Knights"
        , "Calgary Flames", "Buffalo Sabres", "Ottawa Senators", "New York Islanders", "Anaheim Ducks"
        , "Tampa Bay Lightning", "Los Angeles Kings", "Winnipeg Jets", "St. Louis Blues"
        , "New Jersey Devils", "Nashville Predators", "San Jose Sharks", "Carolina Hurricanes"
        , "Arizona Coyotes", "Columbus Blue Jackets", "Colorado Avalanche", "Minnesota Wild"
        , "Florida Panthers", "Dallas Stars"]


#
# Purpose: This function is made to return the index corresponding to the team
#   name passed in
#
# Parameters:
#   teamName: the name of the team who's index we are looking for
#   teamsList: the list that we are looking in to find the index
#
# Returns:
#   i : it is a counter that will be the index when we return it
#
def getTeamNumber(teamName, teamsList):
    i = 0

    for name in teamsList :
        if name == teamName :
            return i

        i = i + 1



#
# Purpose: To decipher who one the game. If '0'  is returned the visiting team
#   won. If '1' is returned then the home team won
#
# Parameters:
#   scoreOne: score of the visiting team
#   scoreTwo: score of the home team
#
# Returns:
#   a number representing the winner: see purpose to see what represents what
#
def getWinner(scoreOne, scoreTwo):
    if scoreOne > scoreTwo :
        return 0
    else :
        return 1




#
# Purpose: This will create the basic list with the a reference to the teams
#   who participated in the game. The other number represents the winner
#
# Parameters:
#   teamsList: the list that contains all the current teams
#   rawDataFile: The file that contains the raw file that is being used
#
#
# Returns:
#   baseList : This is a list that contains all the games and the winner. The list
#       format is below
#           [team1, team2, winner], [team1, team2, winner], [team1, team2, winner] ...
#
def createBaseDataList(teamsList, rawDataFile):
    inputFile = open(rawDataFile, "r")

    baseList = []

    contentsOfFile = inputFile.read()

    linesInFile = contentsOfFile.split('\n')

    for line in linesInFile :
        newItem = [0,0,0]

        lineContents = line.split(',')

        if lineContents == [''] :
            break

        newItem[0] = getTeamNumber(lineContents[1], teamsList)

        newItem[1] = getTeamNumber(lineContents[3], teamsList)

        newItem[2] = getWinner(lineContents[2], lineContents[4])

        baseList.append(newItem)

    return baseList



#
# Purpose:
#
# Parameters:
#
# Returns:
#
# teamPlace is whether the team is first in the list or second
def addRecordToEntry(game, currentRecord, teamPlace) :
    if teamPlace == 0 :
        if len(game) == 7:
            game[2] = currentRecord[0]
            game[3] = currentRecord[1]

            return game
        else :
            newEntry = [game[0],game[1]]

            # add new data
            newEntry.append(currentRecord[0])
            newEntry.append(currentRecord[1])
            newEntry.append(0)
            newEntry.append(0)
            newEntry.append(game[2])

            return newEntry
    else :
        if len(game) == 7:
            game[4] = currentRecord[0]
            game[5] = currentRecord[1]

            return game
        else :
            newEntry = [game[0],game[1]]

            # add new data
            newEntry.append(0)
            newEntry.append(0)
            newEntry.append(currentRecord[0])
            newEntry.append(currentRecord[1])
            newEntry.append(game[2])

            return newEntry

#
# The next functions will add to our base list.
#
#
# Purpose:
#
# Parameters:
#
# Returns:
#
# teamPlace is whether the team is first in the list or second
def updateCurrentRecord(currentRecord, game, teamPlace):
    if teamPlace == 0:
        if len(game) == 7 :
            if game[6] == 0 :
                currentRecord[0] = currentRecord[0] + 1
            else :
                currentRecord[1] = currentRecord[1] + 1
        else:
            if game[2] == 0 :
                currentRecord[0] = currentRecord[0] + 1
            else :
                currentRecord[1] = currentRecord[1] + 1
    else:
        if len(game) == 7 :
            if game[6] == 1 :
                currentRecord[0] = currentRecord[0] + 1
            else :
                currentRecord[1] = currentRecord[1] + 1
        else:
            if game[2] == 1 :
                currentRecord[0] = currentRecord[0] + 1
            else :
                currentRecord[1] = currentRecord[1] + 1

    return currentRecord



#
# Purpose:
#
# Parameters:
#
# Returns:
#
def addTeamRecordInCurrentSeason(mainList):

    teams = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

    for team in teams :
        counter = 0

        # the first number is the amount of wins and second is the amount of losses
        currentRecord = [0,0]
        for game in mainList :
            if game[0] == team :

                # replace old entry with new extended entry
                mainList[counter] = addRecordToEntry(game, currentRecord, 0)

                currentRecord = updateCurrentRecord(currentRecord, game, 0)

            if game[1] == team :

                # replace old entry with new extended entry
                mainList[counter] = addRecordToEntry(game, currentRecord, 1)

                currentRecord = updateCurrentRecord(currentRecord, game, 1)

            counter = counter + 1


    return mainList


#
# Purpose:
#
# Parameters:
#
# Returns:
#
def returnCurrentRecordBetweenTeams(mainList):
    teams = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

    listOfRecords = []

    for team1 in teams :
        for team2 in teams :
            counter = 0
            for game in mainList :
                newListEntry = [0,0,0]
                if (team1 == game[0] and team2 == game[1]) or (team1 == game[1] and team2 == game[0]):
                    if game[2] == 0:
                        newListEntry[0] = newListEntry[0] + 1
                    else :
                        newListEntry[1] = newListEntry[1] + 1
                    newListEntry[2] = counter
                    listOfRecords.append(newListEntry)

    return listOfRecords

#
# Purpose:
#
# Parameters:
#
# Returns:
#
def addRecordsBetweenTeamsToData(listOfRecords, mainList):
    for entry in listOfRecords:
        mainList[listOfRecords[2]].append(0)
        mainList[listOfRecords[2]].append(mainList[[listOfRecords[2]]][6])

        mainList[listOfRecords[2]][6] = listOfRecords[0]
        mainList[listOfRecords[2]][6] = listOfRecords[1]









# extra lines
