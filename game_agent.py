"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


### ------------------------------------------------------------------------------------------- ###

def custom_score(game, player):
    """This heuristic will calculate a random value as the score.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return random.random()

### ------------------------------------------------------------------------------------------- ###

def custom_score_2(game, player):
    """This heuristic will calculate the ratio of my_moves over the total
    moves or my_moves / (my_moves + opponent_moves).

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves / (own_moves + opp_moves))

### ------------------------------------------------------------------------------------------- ###

def custom_score_3(game, player):
    """This heuristic will calculate the distance between the two players.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    y1, x1 = game.get_player_location(player)
    y2, x2 = game.get_player_location(game.get_opponent(player))
    d = (((y2 - y1) ** 2) + ((x2 - x1) ** 2)) ** 0.5
    return float(d)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # by default the game allows 150 milliseconds for each turn, this timer code
        # check ensures that this program doesn't exceed that limit - causing the user
        # to wait a long time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # max_value returns a tuple with a float, and a list inside
        # i.e., (3.0, [(3, 4), (5, 6), (7, 8)])
        # the list is the history where that value originated, we grab
        # the last value (7, 8) since that's where we want to move next
        return self.max_value(game, depth)[1][-1]

    def max_value(self, game, depth):
        # returns the max of the min values
        # perform timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # get all legal moves available to current player
        moves = game.get_legal_moves()
        # if there is nowhere to go or if we are at max depth, then we
        # return a score from this location
        if depth == 0 or not moves:
            return (self.score(game, self), [(-1, -1)])
        # contains all the scores from the min-nodes, which will be recursively called
        results = []
        # iterate over all possible moves
        for move in moves:
            # recursively call min_value, the forcast_move takes the move
            # i want to try and applies it to a copy of the current game
            # also, since we are descending deeper, i decrement depth by 1
            # returns a tuple -> (score, [move, move, move])
            score, history = self.min_value(game.forecast_move(move), depth - 1)
            # save all the moves so we can use it later
            history.append(move)
            results.append((score, history))
        # returning the maximum value from all the min-nodes. includes the history
        return max(results)

    def min_value(self, game, depth):
        # just like the "max_value" function, above, except it returns the
        # min of the max values
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return (self.score(game, self), [(-1, -1)])
        results = []
        for move in moves:
            score, history = self.max_value(game.forecast_move(move), depth - 1)
            history.append(move)
            results.append((score, history))
        return min(results)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)

        # performing iterative deepening search
        # start with depth 0, get best move, then
        # continue to go deeper as time permits
        # an exception will get thrown and caught here
        # which i will then return the latest best move
        try:
            depth = -1
            while True:
                depth += 1
                best_move = self.alphabeta(game, depth)
        except SearchTimeout:
            pass

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # by default the game allows 150 milliseconds for each turn, this timer code
        # check ensures that this program doesn't exceed that limit - causing the user
        # to wait a long time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # we call max_value with alpha = -inf and beta = +inf
        # max_value returns a tuple with a float, and a list inside
        # i.e., (3.0, [(3, 4), (5, 6), (7, 8)])
        # the list is the history where that value originated, we grab
        # the last value (7, 8) since that's where we want to move next
        return self.max_value(game, depth, alpha, beta)[1][-1]

    def max_value(self, game, depth, alpha, beta):
        # returns the max of the min values
        # perform timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        # if there is nowhere to go or if we are at max depth, then we
        # return a score from this location
        if depth == 0 or not moves:
            return (self.score(game, self), [(-1, -1)])
        results = []
        for move in moves:
            # recursively call min_value, the forcast_move takes the move
            # i want to try and applies it to a copy of the current game
            # also, since we are descending deeper, i decrement depth by 1
            # returns a tuple -> (score, [move, move, move])
            score, history = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            # save all the moves so we can use it later
            history.append(move)
            results.append((score, history))
            # alpha/beta pruning
            if score >= beta: return (score, history)
            alpha = max(alpha, score)
        # returning the maximum value from all the min-nodes. includes the history
        return max(results)

    def min_value(self, game, depth, alpha, beta):
        # just like the "max_value" function, above, except it returns the
        # min of the max values
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moves = game.get_legal_moves()
        if depth == 0 or not moves:
            return (self.score(game, self), [(-1, -1)])
        results = []
        for move in moves:
            score, history = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)
            history.append(move)
            results.append((score, history))
            if score <= alpha: return (score, history)
            beta = min(beta, score)
        return min(results)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
