import utilityFunctions.py

from keras.models import Sequential
from keras.layers import Dense
import numpy

seasonThirteenFourteen = createBaseDataList(teamsList, "2013-2014Season.txt")
seasonFourteenFifteen = createBaseDataList(teamsList, "2014-2015Season.txt")
seasonFifteenSixteen = createBaseDataList(teamsList, "2015-2016Season.txt")
seasonSixteenSeventeen = createBaseDataList(teamsList, "2016-2017Season.txt")
seasonSeventeenEightteen = createBaseDataList(teamsList, "2017-2018Season.txt")

ThirteenFourteenRecord = returnCurrentRecordBetweenTeams(seasonThirteenFourteen)
FourteenFifteenRecord = returnCurrentRecordBetweenTeams(seasonFourteenFifteen)
FifteenSixteenRecord = returnCurrentRecordBetweenTeams(seasonFifteenSixteen)
SixteenSeventeenRecord = returnCurrentRecordBetweenTeams(seasonSixteenSeventeen)
SeventeenEightteenRecord = returnCurrentRecordBetweenTeams(seasonSeventeenEightteen)

addRecordsBetweenTeamsToData(ThirteenFourteenRecord, seasonThirteenFourteen)
addRecordsBetweenTeamsToData(FourteenFifteenRecord, seasonFourteenFifteen)
addRecordsBetweenTeamsToData(FifteenSixteenRecord, seasonFifteenSixteen)
addRecordsBetweenTeamsToData(SixteenSeventeenRecord, seasonSixteenSeventeen)
addRecordsBetweenTeamsToData(SeventeenEightteenRecord, seasonSeventeenEightteen)