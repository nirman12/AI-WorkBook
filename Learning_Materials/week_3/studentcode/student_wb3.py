
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here

        # Step 1: Calculate the index of the last element in the open_list
        my_index = len(self.open_list) - 1 

        # Step 2: Retrieve the last solution from the open_list
        next_soln = self.open_list[my_index]  

        # Step 3: Remove the last element from the open_list
        self.open_list.pop(my_index) 

        # <==== insert your pseudo-code and code above here

        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        index=0
        
        next_soln = self.open_list[index]  

        
        self.open_list.pop(index) 

        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here

        # Step 1: Check if the open_list is empty
        if not self.open_list:
            # If empty, no next solution to return
            return None
        else:
            # Assume the first solution is the best initially
            best_index = 0

            # Step 2: Iterate through open_list to find the solution with the lowest quality
            for i in range(len(self.open_list)):
                if self.open_list[i].quality < self.open_list[best_index].quality:
                    best_index = i

        # Step 3: Remove and retrieve the best solution from open_list
        next_soln = self.open_list.pop(best_index)

        # <==== insert your pseudo-code and code above here

        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        if not self.open_list:
            
            return None
        
        else:
            best_index=0
            for i in range (len(self.open_list)):
                if self.open_list[i].quality+len(self.open_list[i].variable_values)<self.open_list[best_index].quality+len(self.open_list[best_index].variable_values):
                    best_index=i

            
        return self.open_list.pop(best_index)

        # <==== insert your pseudo-code and code above here
        return next_soln
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    # Remember to comment out any call to show_maze() before submitting your work

    # Initialize the maze from the file "maze.txt"
    maze = Maze(mazefile="maze.txt")

    # Modify specific cells in the maze to represent holes and walls
    maze.contents[3][4] = hole_colour      # Mark cell (3,4) as a hole
    maze.contents[8][4] = wall_colour      # Mark cell (8,4) as a wall

    maze.contents[10][6] = hole_colour     # Mark cell (10,6) as a hole
    maze.contents[14][6] = wall_colour     # Mark cell (14,6) as a wall
    maze.contents[16][1] = hole_colour     # Mark cell (16,1) as a hole
    maze.contents[19][4] = hole_colour     # Mark cell (19,4) as a hole

    maze.contents[8][1] = hole_colour      # Mark cell (8,1) as a hole
    maze.contents[12][9] = wall_colour     # Mark cell (12,9) as a wall
    maze.contents[11][12] = wall_colour    # Mark cell (11,12) as a wall
    maze.contents[9][2] = wall_colour      # Mark cell (9,2) as a wall
    maze.contents[10][19] = wall_colour    # Mark cell (10,19) as a wall
    maze.contents[18][5] = wall_colour     # Mark cell (18,5) as a wall

    # Save the modified maze layout to a new text file
    maze.save_to_txt("maze-breaks-depth.txt")
    # <==== insert your code above here

def create_maze_depth_better():
    # ====> insert your code below here
    # Remember to comment out any calls to show_maze() before submitting your work

    # Create a default 21x21 maze by loading from "maze.txt"
    maze = Maze(mazefile="maze.txt")

    # Set specific positions in the maze to be walls
    maze.contents[1][8] = wall_colour       # wall at row 1, column 8
    maze.contents[9][10] = wall_colour      # wall at row 9, column 10
    maze.contents[15][6] = wall_colour      # wall at row 15, column 6
    maze.contents[13][2] = wall_colour      # wall at row 13, column 2
    maze.contents[12][13] = wall_colour     # wall at row 12, column 13
    maze.contents[2][13] = wall_colour      # wall at row 2, column 13

    # Save the modified maze to a new file "maze-depth-better.txt"
    maze.save_to_txt("maze-depth-better.txt")
    # <==== insert your code above here
