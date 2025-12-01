# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from pyexpat import features
import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # resolve string names in this module's global namespace (prevents NameError)
    if isinstance(first, str):
        first_cls = eval(first, globals())
    else:
        first_cls = first
    if isinstance(second, str):
        second_cls = eval(second, globals())
    else:
        second_cls = second
    return [first_cls(first_index), second_cls(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        
        # Use current agent state
        agent_state = game_state.get_agent_state(self.index)
        my_pos = agent_state.get_position()

        enemies_states = [
            game_state.get_agent_state(i)
            for i in self.get_opponents(game_state)
        ]

        closest_active_ghost_dist = 9999
        closest_scared_ghost_dist = 9999
        has_scared_ghost = False

        for g in enemies_states:
            g_pos = g.get_position()
            if g_pos is None:
                continue
            if g.is_pacman:
                continue  # ignore invaders here

            dist = self.get_maze_distance(my_pos, g_pos)

            if g.scared_timer > 0:
                has_scared_ghost = True
                closest_scared_ghost_dist = min(closest_scared_ghost_dist, dist)
            else:
                closest_active_ghost_dist = min(closest_active_ghost_dist, dist)

        carrying_now = getattr(agent_state, 'numCarrying',
                            getattr(agent_state, 'num_carrying', 0))


        being_safe = 4
        # if the other teams agent is scared, we add as to being safe the ammount of time left divided by 4
        enemies = self.get_opponents(game_state) 
        for enemy in enemies:#enemy is the index of the enemy agent
            enemy_state = game_state.get_agent_state(enemy) #can be known
            if not enemy_state.is_pacman: #enemy hasn't crossed, it is defending its side
                if enemy_state.scared_timer > 0: #enemy is scared we can risk more and collect more food
                    being_safe += enemy_state.scared_timer // 8
                else: #if the enemy is an active ghost trying to eat us, we only know where it is f it is closer than 5 units
                      #so if we know where it is, we have to be more careful
                    if game_state.get_agent_position(enemy) is not None:
                        being_safe = 3 #if an active ghost is closer than 5, we only risk carrying 3 pellets
        
        # Force return home when carrying enough food
        if agent_state.is_pacman and carrying_now >= being_safe:
            best_dist = 3000
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # Return home when very few food pellets left
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_state(self.index).get_position()
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        
        # Food features
        current_food = self.get_food(game_state).as_list()
        food_list = self.get_food(successor).as_list()
        features['eaten_food'] = len(current_food) - len(food_list)
        features['successor_score'] = self.get_score(successor)
        
        # Distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0
        
        # Capsule features
        current_capsules = self.get_capsules(game_state)
        successor_capsules = self.get_capsules(successor)
        features['eaten_capsule'] = 1 if len(current_capsules) > len(successor_capsules) else 0
        
        # Only consider capsules on enemy side
        capsules = [c for c in successor_capsules if 
                    ((self.red and c[0] >= successor.data.layout.width // 2) or
                     (not self.red and c[0] < successor.data.layout.width // 2))]
        
        if capsules:
            features['distance_to_capsule'] = min(self.get_maze_distance(my_pos, c) for c in capsules)
        else:
            features['distance_to_capsule'] = 0
        
        # Ghost features
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if (not e.is_pacman) and e.get_position() is not None]
        
        # Default values
        features['min_ghost_distance'] = 0
        features['close_scared_ghost'] = 0
        features['eaten_scared_ghost'] = 0
        features['imminent_danger'] = 0
        
        if ghosts:
            non_scared_dists, scared_dists = [], []
            
            for g in ghosts:
                d = self.get_maze_distance(my_pos, g.get_position())
                
                if g.scared_timer > 0:
                    scared_dists.append(d)
                    if d == 0:
                        features['eaten_scared_ghost'] = 1
                else:
                    non_scared_dists.append(d)
                    # Mark imminent danger if ghost is very close
                    if d <= 2:
                        features['imminent_danger'] = 1
            
            if non_scared_dists:
                features['min_ghost_distance'] = min(non_scared_dists)
            
            if scared_dists:
                features['close_scared_ghost'] = min(scared_dists)
        
        # Carrying and home features
        carrying = getattr(successor.get_agent_state(self.index), 'numCarrying', 0)
        features['food_carried'] = carrying
        
        # Distance to home
        if carrying > 0:
            width = successor.data.layout.width
            height = successor.data.layout.height
            mid_x = width // 2
            home_x = mid_x - 1 if self.red else mid_x
            
            possible_home_positions = [(home_x, y) for y in range(height)
                                       if not successor.has_wall(home_x, y)]
            if possible_home_positions:
                min_home_distance = min(self.get_maze_distance(my_pos, pos) 
                                       for pos in possible_home_positions)
                features['distance_to_home'] = min_home_distance
            else:
                features['distance_to_home'] = 0
        else:
            features['distance_to_home'] = 0
        
        features['stop'] = 1 if action == Directions.STOP else 0
        
        return features

    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        carrying = getattr(my_state, 'numCarrying', 0)
        
        # Analyze the threat level
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if (not e.is_pacman) and e.get_position() is not None]
        
        danger_level = 0  # 0 = safe, 1 = caution, 2 = danger, 3 = extreme danger
        closest_ghost_dist = 999
        has_scared_ghost = False # track if any ghost is scared
        
        if ghosts:
            my_pos = my_state.get_position()
            for g in ghosts:
                if g.scared_timer == 0:  # active ghost
                    dist = self.get_maze_distance(my_pos, g.get_position())
                    closest_ghost_dist = min(closest_ghost_dist, dist)
                else:
                    has_scared_ghost = True
            
            if closest_ghost_dist <= 3:
                danger_level = 3  # EXTREME: ghost very close
            elif closest_ghost_dist <= 5:
                danger_level = 2  # HIGH: ghost nearby
            elif closest_ghost_dist <= 6:
                danger_level = 1  # MODERATE: ghost in area
        
        # Decision logic: return home if carrying enough OR in danger
        should_return = carrying >= 5 or (danger_level >= 2 and carrying >= 4) or (danger_level == 3 and carrying >= 1)
        
        if has_scared_ghost:
            home_priority = 50 + (carrying * 30)  # ITERATIVE FORMULA
            return {
                'eaten_food': 2000,
                'successor_score': 50,
                'distance_to_food': -3,
                'distance_to_capsule': -8,
                'min_ghost_distance': 200,
                'close_scared_ghost': 0,
                'food_carried': 0,
                'distance_to_home': -home_priority,
                'stop': -300,
                'eaten_capsule': 500,
                'eaten_scared_ghost': 30,
                'imminent_danger': -5000
            }
            
        if danger_level == 3 and carrying > 0:
            # PANIC MODE: Ghost extremely close with food - RUN HOME!
            return {
                'eaten_food': 3,
                'successor_score': 10,
                'distance_to_food': 0,
                'distance_to_capsule': -5,
                'min_ghost_distance': 300,
                'close_scared_ghost': 0,
                'food_carried': 0,
                'distance_to_home': -500,
                'stop': -1000,
                'eaten_capsule': 1000,
                'eaten_scared_ghost': 0,
                'imminent_danger': -10000
            }
        
        elif should_return:
            # RETURN HOME MODE: carrying enough food or moderate danger
            return {
                'eaten_food': 50,
                'successor_score': 50,
                'distance_to_food': -3,
                'distance_to_capsule': -8,
                'min_ghost_distance': 200,
                'close_scared_ghost': -20,
                'food_carried': 0,
                'distance_to_home': -200,
                'stop': -300,
                'eaten_capsule': 500,
                'eaten_scared_ghost': 300,
                'imminent_danger': -5000
            }
        
        elif danger_level == 1:
            # CAUTIOUS MODE: ghost nearby but manageable
            return {
                'eaten_food': 900,
                'successor_score': 100,
                'distance_to_food': -25,
                'distance_to_capsule': -50,
                'min_ghost_distance': 100,
                'close_scared_ghost': -150,
                'food_carried': 0,
                'distance_to_home': -5,
                'stop': -200,
                'eaten_capsule': 900,
                'eaten_scared_ghost': 1000,
                'imminent_danger': -3000
            }
        
        else:
            # AGGRESSIVE MODE: safe area, eat everything!
            return {
                'eaten_food': 9000,
                'successor_score': 150,
                'distance_to_food': -50,
                'distance_to_capsule': -60,
                'min_ghost_distance': 40,
                'close_scared_ghost': -200,
                'food_carried': 20,
                'distance_to_home': -1,
                'stop': -100,
                'eaten_capsule': 900,
                'eaten_scared_ghost': 1000,
                'imminent_danger': 0
            }


class DefensiveReflexAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # avoid crossing without valid cause
        features['on_own_side'] = 1
        if my_state.is_pacman: features['on_own_side'] = 0

        # detect invaders from CURRENT state
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        #patrol closer to food
        our_food = self.get_food_you_are_defending(game_state).as_list()

        if our_food and my_pos is not None:
            # Compute distance to the closest food we are defending
            food_dists = [self.get_maze_distance(my_pos, tuple(f)) for f in our_food]
            c = len(food_dists)//5 
            features['distance_to_our_food'] = food_dists[c] #go to a close food but not the closest
        else:
            # No food left or unknown position
            features['distance_to_our_food'] = 0


        # Distance to closest invader
        if invaders:
            dists = []
            for a in invaders:
                pos = a.get_position()
                if pos is not None and my_pos is not None:
                    dists.append(self.get_maze_distance(my_pos, pos))
            features['invader_distance'] = min(dists)
        else:
            features['invader_distance'] = 5

        # Patrol target: middle of home boundary
        width = successor.data.layout.width
        mid_x = width // 2
        boundary_x = mid_x - 1 if self.start[0] < mid_x else mid_x + 1

        mid_y = successor.data.layout.height // 2
        boundary_pos = (boundary_x, mid_y)

        if my_pos is not None:
            try:
                features['distance_to_boundary'] = self.get_maze_distance(my_pos, boundary_pos)
            except Exception:
                features['distance_to_boundary'] = 0
        else:
            features['distance_to_boundary'] = 0

        # Patrol vertical distance
        distance_to_side = abs(my_pos[1] - mid_y)
        features['distance_to_sides'] = distance_to_side

        # Catch invader reward
        features['caught_invader'] = 0
        enemies_now = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders_now = [inv for inv in enemies_now if inv.is_pacman and inv.get_position() is not None]

        enemies_later = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders_later = [inv for inv in enemies_later if inv.is_pacman and inv.get_position() is not None]

        if len(invaders_now) > len(invaders_later):
            features['caught_invader'] = 1

        # Stop / reverse penalties
        features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0

        return features


    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        scared = getattr(my_state, 'scared_timer', 0) > 0

        # --- FIX 1 repeated: use CURRENT state for invader visibility
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        has_invaders = len(invaders) > 0

        # Chase mode
        if has_invaders and not scared:
            return {
                'num_invaders': -8000,
                'invader_distance': -5000,  # stronger to chase now
                'distance_to_boundary': 0,
                'distance_to_sides': 0,
                'stop': -500,
                'reverse': -10,
                'caught_invader': 5000,
                'on_own_side': 0, #it is supposed to be on own side but adding weight is redundant
                'distance_to_our_food': 0
            }

        # Scared mode
        elif scared:
            return {
                'num_invaders': -800,
                'invader_distance': 10,
                'distance_to_boundary': -2,
                'distance_to_sides': -1,
                'stop': -100,
                'reverse': -5,
                'caught_invader': 100,
                'on_own_side': -2,
                'distance_to_our_food': 0
            }

        # Patrol mode
        else:
            return {
                'num_invaders': -1000,
                'invader_distance': -300,
                'distance_to_boundary': -2.3,
                'distance_to_sides': 0,
                'stop': -100,
                'reverse': -3,
                'caught_invader': 1000,
                'on_own_side': 3,
                'distance_to_our_food': -3
            }
