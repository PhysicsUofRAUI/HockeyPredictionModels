import unittest
from utilityFunctions import *

class testingGetTeamNumber(unittest.TestCase):

    def test_firstTeam(self):
        num = getTeamNumber("Toronto Maple Leafs", teams)

        self.assertEqual(0, num)

    def test_lastTeam(self):
        num = getTeamNumber("Dallas Stars", teams)

        self.assertEqual(30, num)

    def test_someTeamInTheMiddle(self):
        num = getTeamNumber("Ottawa Senators", teams)

        self.assertEqual(14, num)


class testingGetWinner(unittest.TestCase):

    def test_LargeDifferenceFirstTeam(self):
        num = getWinner(100000, 32)

        self.assertEqual(0, num)

    def test_SmallDifferenceFirstTeam(self):
        num = getWinner(2, 1)

        self.assertEqual(0, num)

    def test_LargeDifferenceSecondTeam(self):
        num = getWinner(1, 320000)

        self.assertEqual(1, num)

    def test_SmallDifferenceSecondTeam(self):
        num = getWinner(1, 2)

        self.assertEqual(1, num)


class testingAddRecordToEntry(unittest.TestCase):

    def test_FirstTeamWinsSecondTeamPassedInNoRecordAddedYet(self):
        game = [2,4, 0]

        currentRecord = [5,4]

        teamPlace = 1

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,0,0,5,4,0])

    def test_FirstTeamWinsFirstTeamPassedInNoRecordAddedYet(self):
        game = [2,4,0]

        currentRecord = [4,9]

        teamPlace = 0

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,4,9,0,0,0])

    def test_SecondTeamWinsSecondTeamPassedInNoRecordAddedYet(self):
        game = [2,4, 1]

        currentRecord = [5,4]

        teamPlace = 1

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,0,0,5,4,1])

    def test_SecondTeamWinsFirstTeamPassedInNoRecordAddedYet(self):
        game = [2,4,1]

        currentRecord = [4,9]

        teamPlace = 0

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,4,9,0,0,1])


    # Next Tests will be when a record is already added
    def test_FirstTeamWinsSecondTeamPassedInRecordAdded(self):
        game = [2,4, 4, 3, 0, 0, 0]

        currentRecord = [5,4]

        teamPlace = 1

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,4,3,5,4,0])

    def test_FirstTeamWinsFirstTeamPassedInRecordAdded(self):
        game = [2,4,0,0,5,6,0]

        currentRecord = [4,9]

        teamPlace = 0

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,4,9,5,6,0])

    def test_SecondTeamWinsSecondTeamPassedInRecordAdded(self):
        game = [2,4,3,4,0,0,1]

        currentRecord = [5,4]

        teamPlace = 1

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,3,4,5,4,1])

    def test_SecondTeamWinsFirstTeamPassedInRecordAdded(self):
        game = [2,4,0,0,3,9,1]

        currentRecord = [4,9]

        teamPlace = 0

        newRecord = addRecordToEntry(game, currentRecord, teamPlace)

        self.assertEqual(newRecord, [2,4,4,9,3,9,1])



class testingCreatBaseList(unittest.TestCase):

    def test_PartOfList(self):
        baseList = createBaseDataList(teams, "testList.txt")

        self.assertEqual([[12,4,1],[20,7,0],[10,23,0]], baseList)

class testingAddTeamRecordInCurrentSeason(unittest.TestCase):

    def test_PartOfList(self):
        mainList = [[5,0,1],[4,3,0],[5,4,1],[2,3,0],[0,1,0],[2,1,1],[3,1,0],[2,4,1]]

        whatListShouldBe = [[5,0,0,0,0,0,1],[4,3,0,0,0,0,0],[5,4,0,1,1,0,1]
                            ,[2,3,0,0,0,1,0],[0,1,1,0,0,0,0],[2,1,1,0,0,1,1]
                            ,[3,1,0,2,1,1,0],[2,4,1,1,2,0,1]]

        newList = addTeamRecordInCurrentSeason(mainList)

        self.assertEqual(whatListShouldBe, newList)
















if __name__ == '__main__':
    unittest.main()
