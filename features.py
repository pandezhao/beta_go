
'''
Features used by AlphaGo, in approximate order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty  
Ones                    1 Constant plane of 1s 
    (Because of convolution w/ zero-padding, this is the only way the NN can know where the edge of the board is!!!)
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s
All features with 8 planes are 1-hot encoded, with plane i marked with 1 
only if the feature was equal to i. Any features >= 8 would be marked as 8.
'''