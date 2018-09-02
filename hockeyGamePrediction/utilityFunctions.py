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
# Purpose: This function is meant to add the record of the team in
# 	in the current season.
#
# Parameters:
# 	game: this is the current entry of that will be returned with the record 
# 		of the team that was also passed in. It will be a list with 7 entries
# 		if this is the second time the team has been passed in, or it will 
# 		be a list of 3 if it has not been passed to the function before.
#
# 	currentRecord: this is the currentRecord of the team being passed in. It will 
# 		be added to the newEntry in the correct place based on the team passed in.
#
# 	teamPlace: This is the number that will signify which team was passed in. If it is
# 		0 then the record will be placed first and if it is 1 it will be placed second.
#
# Returns: 
# 	newEntry: This will be entered in the same place as the parameter 'game' was in on the 
# 		main list. What was changed was that the record of the team passed in will have 
# 		been added to the list.
#
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
#
# Purpose: This function is made for the sole purpose of updating the current
# 	record of the team passed in.
#
# Parameters:
# 	currentRecord: this is the record of the team before the game passed in
# 		was played
#
# 	game: this is the game that will change the record of the team now
#
# 	teamPlace: This identifies the team that is being updated
#
# Returns:
# 	currentRecord: this is the updated record.
#
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
# Purpose: This function goes through the entire list of games and add each teams record to
# 	the game's entry.
#
# Parameters:
# 	mainList: This the entire list of games in the season.
#
# Returns: 
# 	mainList: This is the same list passed in but with the current record of each
# 		team added to the game entry.
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
# Purpose: This function is designed to return the current record between the teams
# 	during the current season
#
# Parameters:
#	mainList: This is the list containing all the different games with the team and
# 		winner listed.
#
# Returns:
# 	listOfRecords: this is the list containing all the records. It's 3rd entry will 
# 		identify which entry it belongs to in mainList
#
# Note: there is a mistake below fix it later when indentation erro is less likely
# 	counter needs to be incremented
# 	current set up will not keep track of the records because newListEntry is reset each time
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
# Purpose: this function will add the records between the teams to
# 	the main list
#
# Parameters:
# 	listOfRecords: This is the list that contains the list of the record
# 		between the teams for a game with the index specified at index
# 		2 of the array.
#
# 	mainList: this is the lsit of all the games. It is the one that will modified
# 		to list all the records between the teams.
#
# Returns:
# 	no returns just modification for now :)
#
def addRecordsBetweenTeamsToData(listOfRecords, mainList):
    for entry in listOfRecords:
        mainList[listOfRecords[2]].append(0)
        mainList[listOfRecords[2]].append(mainList[[listOfRecords[2]]][6])

        mainList[listOfRecords[2]][6] = listOfRecords[0]
        mainList[listOfRecords[2]][6] = listOfRecords[1]









# extra lines
