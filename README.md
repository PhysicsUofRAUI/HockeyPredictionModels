# HockeyPredictionModels

## Purpose
This is a place to put all my code, data, and thoughts on messing with hockey data. Right now the only data project that is being persued is an attempt to predict the winner of a hockey game with a neural network. I hope to add more projects here as I learn more about how to work with data. 

## Projects

### Hockey Neural Network
This project is my attempt at building a neural network from game data that was scraped from http://www.espn.com/nhl/ . I got around 5000 good data entries from that site, but I hope to get more from http://www.hockeydb.com since it looks like there is more entries on that site.

Right now the model is built by making all the players and goalies into dummy variables and then running the result through a network built with [Keras](https://keras.io). I have read some things and done some research that makes me think this is not the best way, so I am looking into how to do this better. 

Right now the model gets stuck around 54% and then looks like it is going up, but crossvalidation tells a different story. The crossvalidation shows that it's actual accuracy is 52% ish. It also runs very slowly and takes lots of computing power. It should be noted that 54% is also the accuracy you get, historically if you only pick the home team.

I am reading through the book located at the following link, http://www.dkriesel.com/en/science/neural_networks, and I hope to learn enough to more efficiently make my model.

If I am not able to get it working well using this method I will move to a different method.

## Contributing
Anyone is welcome to contribute just keep in mind this is the first time I've done something like this so I might be doing stupid stuff here.
