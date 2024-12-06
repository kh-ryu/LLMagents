import numba as nb
import numpy as np
from typing import Optional, Literal
from ..core.agent import AgentState
from ..core.constants import Color, Direction, State, Type
from ..core.world_object import Wall, WorldObj

from numpy.typing import NDArray as ndarray
from ..core.actions import Action




### Typing

AgentID = int


### Constants

WALL_ENCODING = Wall().encode()
UNSEEN_ENCODING = WorldObj(Type.unseen, Color.from_index(0)).encode()
ENCODE_DIM = WorldObj.dim

GRID_ENCODING_IDX = slice(None)

AGENT_DIR_IDX = AgentState.DIR
AGENT_POS_IDX = AgentState.POS
AGENT_TERMINATED_IDX = AgentState.TERMINATED
AGENT_CARRYING_IDX = AgentState.CARRYING
AGENT_ENCODING_IDX = AgentState.ENCODING

TYPE = WorldObj.TYPE
COLOR = WorldObj.COLOR
STATE = WorldObj.STATE

WALL = int(Type.wall)
DOOR = int(Type.door)

OPEN = int(State.open)
CLOSED = int(State.closed)
LOCKED = int(State.locked)

RIGHT = int(Direction.right)
LEFT = int(Direction.left)
UP = int(Direction.up)
DOWN = int(Direction.down)




### Observation Functions

@nb.njit(cache=True)
def see_behind(world_obj: ndarray[np.int_]) -> bool:
    """
    Can an agent see behind this object?

    Parameters
    ----------
    world_obj : ndarray[int] of shape (encode_dim,)
        World object encoding
    """
    if world_obj is None:
        return True
    if world_obj[TYPE] == WALL:
        return False
    elif world_obj[TYPE] == DOOR and world_obj[STATE] != OPEN:
        return False

    return True


def  gen_obs_grid_encoding(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int,
    see_through_walls: bool
):
    """
    Generate encoding for the sub-grid observed by an agent (including visibility mask).

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids
    see_through_walls : bool
        Whether the agent can see through walls

    Returns
    -------
    img : ndarray[int] of shape (num_agents, view_size, view_size, encode_dim)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size)
   
    # Generate and apply visibility masks
    vis_mask = get_vis_mask(obs_grid)
    num_agents = len(agent_state)
    for agent in range(num_agents):
        if not see_through_walls:
            for i in range(agent_view_size):
                for j in range(agent_view_size):
                    if not vis_mask[agent, i, j]:
                        obs_grid[agent, i, j] = UNSEEN_ENCODING
    
  
    obs_text = obs_to_text(obs_grid)
   
    return obs_grid, obs_text


@nb.njit(cache=True)
def gen_obs_grid_vis_mask(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int) -> ndarray[np.int_]:
    """
    Generate visibility mask for the sub-grid observed by an agent.

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids

    Returns
    -------
    mask : ndarray[int] of shape (num_agents, view_size, view_size)
        Encoding of observed sub-grid for each agent
    """
    obs_grid = gen_obs_grid(grid_state, agent_state, agent_view_size)
    return get_vis_mask(obs_grid)


@nb.njit(cache=True)
def gen_obs_grid(
    grid_state: ndarray[np.int_],
    agent_state: ndarray[np.int_],
    agent_view_size: int) -> ndarray[np.int_]:
    """
    Generate the sub-grid observed by each agent (WITHOUT visibility mask).

    Parameters
    ----------
    grid_state : ndarray[int] of shape (width, height, grid_state_dim)
        Array representation for each grid object
    agent_state : ndarray[int] of shape (num_agents, agent_state_dim)
        Array representation for each agent
    agent_view_size : int
        Width and height of observation sub-grids

    Returns
    -------
    obs_grid : ndarray[int] of shape (num_agents, width, height, encode_dim)
        Observed sub-grid for each agent
    """
    num_agents = len(agent_state)
    obs_width, obs_height = agent_view_size, agent_view_size
    
    # Process agent states
    agent_grid_encoding = agent_state[..., AGENT_ENCODING_IDX]
    agent_dir = agent_state[..., AGENT_DIR_IDX]
    agent_pos = agent_state[..., AGENT_POS_IDX]
    agent_terminated = agent_state[..., AGENT_TERMINATED_IDX]
    agent_carrying_encoding = agent_state[..., AGENT_CARRYING_IDX]

    # Get grid encoding
    if num_agents > 1:
        grid_encoding = np.empty((*grid_state.shape[:-1], ENCODE_DIM), dtype=np.int_)
        grid_encoding[...] = grid_state[..., GRID_ENCODING_IDX]

        # Insert agent grid encodings
        for agent in range(num_agents):
            if not agent_terminated[agent]:
                i, j = agent_pos[agent]
                grid_encoding[i, j, GRID_ENCODING_IDX] = agent_grid_encoding[agent]
    else:
        grid_encoding = grid_state[..., GRID_ENCODING_IDX]

    # Get top left corner of observation grids
    top_left = get_view_exts(agent_dir, agent_pos, agent_view_size)
    topX, topY = top_left[:, 0], top_left[:, 1]

    # Populate observation grids
    num_left_rotations = (agent_dir + 1) % 4
    obs_grid = np.empty((num_agents, obs_width, obs_height, ENCODE_DIM), dtype=np.int_)
    for agent in range(num_agents):
        for i in range(0, obs_width):
            for j in range(0, obs_height):
                # Absolute coordinates in world grid
                x, y = topX[agent] + i, topY[agent] + j

                # Rotated relative coordinates for observation grid
                if num_left_rotations[agent] == 0:
                    i_rot, j_rot = i, j
                elif num_left_rotations[agent] == 1:
                    i_rot, j_rot = j, obs_width - i - 1
                elif num_left_rotations[agent] == 2:
                    i_rot, j_rot = obs_width - i - 1, obs_height - j - 1
                elif num_left_rotations[agent] == 3:
                    i_rot, j_rot = obs_height - j - 1, i
                # Set observation grid
                if 0 <= x < grid_encoding.shape[0] and 0 <= y < grid_encoding.shape[1]:
                    obs_grid[agent, i_rot, j_rot] = grid_encoding[x, y]
                else:
                    obs_grid[agent, i_rot, j_rot] = WALL_ENCODING
                
    # Make it so each agent sees what it's carrying
    # We do this by placing the carried object at the agent position
    # in each agent's partially observable view
    obs_grid[:, obs_width // 2, obs_height - 1] = agent_carrying_encoding

    return obs_grid

'''
def obs_to_text(
    obs_grid: ndarray[np.int_],
    agent_states
) -> list[str]:
    """
    Convert the observation grid to a human-readable text representation.

    Parameters
    ----------
    obs_grid : ndarray[int] of shape (num_agents, width, height, encode_dim)
        Observed sub-grid for each agent.
    agent_states : list[AgentState]
        List of agent states, including attributes like color, direction, and position.

    Returns
    -------
    descriptions : list[str]
        A list of descriptions for each agent's observation.
    """
    # Extract dimensions
    num_agents, obs_width, obs_height, _ = obs_grid.shape
    descriptions = []
    direction = {RIGHT: "Right", LEFT: "Left", UP: "Up", DOWN: "Down"}
    
    type_symbols = {
        "unseen": "░",
        "empty": " ",
        "wall": "█", 
        "floor": ".", 
        "door": "║",
        "key": "†",
        "ball": "●", 
        "box": "□", 
        "goal": "★",
        "lava": "~", 
        "agent": "@", 
    }
    
    
    direction = {RIGHT: ">", LEFT: "<", UP: "^", DOWN: "v"}
    
    # Process each agent
    for agent in range(num_agents):
        agent_obs = [f"Agent {agent} Observations:"]
        
        for i in range(obs_width):
            row_description = []
            
            for j in range(obs_height):
                # Extract cell information
                cell = obs_grid[agent, i, j]
                cell_type_index = cell[TYPE]
                cell_color_index = cell[COLOR]
                cell_state_index = cell[STATE]

            
                # Map indices to human-readable types
                cell_type = Type.from_index(cell_type_index).name
                cell_color = (
                    Color.from_index(cell_color_index).name if cell_type != "unseen" else ""
                )
                
                if [i, j] == [obs_width // 2, obs_height - 1]:  # Agent's position
                    cell_state = State.from_index(cell_state_index).name
            
                    row_description.append(
                        f"You (Carrying {cell_type}, Color: {cell_color}, State: {cell_state}), "
                        f"Your Color: {agent_states[agent].color.name}, "
                        
                        f"Your Direction: {agent_states[agent].dir.name}"
                    )
                elif cell_type == "agent":  # Another agent in the grid
                    agent_dir = cell_state_index
                    agent_dir = Direction(agent_dir).name
                    row_description.append(
                        f"Other Agent (Color: {agent_states[agent].color.name}, Direction: {agent_dir})"
                    )
                elif cell_type == "empty":  # Empty cell
                    row_description.append("Empty")
                elif cell_type == "unseen":  
                    row_description.append("Unseen")
                else:  # Generic object
                    cell_state = State.from_index(cell_state_index).name
                    row_description.append(
                        f"{cell_type} (Color: {cell_color}, State: {cell_state})"
                    )
            
            
            # Add row description
            agent_obs.append(" | ".join(row_description))
        
        # Append full observation for this agent
        descriptions.append("\n".join(agent_obs))
    
    return descriptions
'''

def obs_to_text(
    obs_grid: ndarray[np.int_]
) -> list[str]:
    """
    Convert the observation grid to a human-readable text representation with dynamic alignment.

    Parameters
    ----------
    obs_grid : ndarray[int] of shape (num_agents, width, height, encode_dim)
        Observed sub-grid for each agent.
        
    Returns
    -------
    descriptions : list[str]
        A list of descriptions for each agent's observation formatted as a grid.
    """
    # Extract dimensions
    num_agents, obs_width, obs_height, _ = obs_grid.shape
    descriptions = []

    # Define symbol mapping for types
    type_symbols = {
        "unseen": "░",
        "empty": ".", 
        "wall": "█", 
        "floor": ".", 
        "door": "≡",
        "key": "†",
        "ball": "●", 
        "box": "□", 
        "goal": "★",
        "lava": "~", 
        "agent": "@", 
    }

    for agent in range(num_agents):
    
        agent_obs = ["Observation:"]  # Initialize agent-specific observation

        agent_carrying_type = obs_grid[agent, obs_width // 2, obs_height - 1][TYPE]
        agent_carrying_type = Type.from_index(agent_carrying_type).name

        # Collect content and calculate maximum width for each column
        cell_contents = [[None for _ in range(obs_height)] for _ in range(obs_width)]
        column_widths = [0] * obs_height

        for i in range(obs_width):

            for j in range(obs_height):
                # Extract cell information
                cell = obs_grid[agent, i, j]
                cell_type_index = cell[TYPE]
                cell_color_index = cell[COLOR]
                cell_state_index = cell[STATE]
                # Map indices to human-readable symbols
                cell_type = Type.from_index(cell_type_index).name
                cell_symbol = type_symbols.get(cell_type, "?")
                cell_color = Color.from_index(cell_color_index).name
                cell_state = State.from_index(cell_state_index).name
                # Generate cell content
                if [i, j] == [obs_width // 2, obs_height - 1]:  # Agent's position
                    cell_content = f"^(You, Carrying {cell_type})"
                elif cell_type == "empty":  # Empty cell
                    cell_content = f"{cell_symbol}"
                elif cell_type == "unseen" or cell_type == "wall":  # Unseen cell
                    cell_content = f"{cell_symbol}"
                else:  # Generic object
                    cell_content = f"{cell_symbol} (Color: {cell_color}, State: {cell_state})"
              
                cell_contents[i][j] = cell_content
                column_widths[j] = max(column_widths[j], len(cell_content))

        # Rotate and flip the matrix for display
        rotated_matrix = [
            [cell_contents[j][i] for j in range(obs_width)]
            for i in range(obs_height - 1, -1, -1)
        ]
        flipped_matrix = rotated_matrix[::-1]
        cell_contents = flipped_matrix
        
        # Update column_widths after rotation
        new_obs_width = len(cell_contents)  # Now rows are the width
        new_obs_height = len(cell_contents[0]) if new_obs_width > 0 else 0  # Columns are the height

        # Recompute column widths for the rotated matrix
        column_widths = [0] * new_obs_height
        for row in cell_contents:
            for j, content in enumerate(row):
                column_widths[j] = max(column_widths[j], len(content))
                
        # Generate the character-based list representation of the matrix
        # Convert cell_contents into a nested string representation
        matrix_str = "[ " + ",\n  ".join([str(row) for row in cell_contents]) + " ]"

        # Add the formatted matrix string to agent_obs
        agent_obs.append(matrix_str)

        # Append agent_obs to descriptions
        descriptions.append("\n".join(agent_obs))

    return descriptions
  
    
@nb.njit(cache=True)
def get_see_behind_mask(grid_array: ndarray[np.int_]) -> ndarray[np.int_]:
    """
    Return boolean mask indicating which grid locations can be seen through.

    Parameters
    ----------
    grid_array : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent

    Returns
    -------
    see_behind_mask : ndarray[bool] of shape (width, height)
        Boolean visibility mask
    """
    num_agents, width, height = grid_array.shape[:3]
    see_behind_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    for agent in range(num_agents):
        for i in range(width):
            for j in range(height):
                see_behind_mask[agent, i, j] = see_behind(grid_array[agent, i, j])

    return see_behind_mask

@nb.njit(cache=True)
def get_vis_mask(obs_grid: ndarray[np.int_]) -> ndarray[np.bool_]:
    """
    Generate a boolean mask indicating which grid locations are visible to each agent.

    Parameters
    ----------
    obs_grid : ndarray[int] of shape (num_agents, width, height, dim)
        Grid object array for each agent observation

    Returns
    -------
    vis_mask : ndarray[bool] of shape (num_agents, width, height)
        Boolean visibility mask for each agent
    """
    num_agents, width, height = obs_grid.shape[:3]
    see_behind_mask = get_see_behind_mask(obs_grid)
    vis_mask = np.zeros((num_agents, width, height), dtype=np.bool_)
    vis_mask[:, width // 2, height - 1] = True # agent relative position

    for agent in range(num_agents):
        for j in range(height - 1, -1, -1):
            # Forward pass
            for i in range(0, width - 1):
                if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
                    vis_mask[agent, i + 1, j] = True
                    if j > 0:
                        vis_mask[agent, i + 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True

            # Backward pass
            for i in range(width - 1, 0, -1):
                if vis_mask[agent, i, j] and see_behind_mask[agent, i, j]:
                    vis_mask[agent, i - 1, j] = True
                    if j > 0:
                        vis_mask[agent, i - 1, j - 1] = True
                        vis_mask[agent, i, j - 1] = True

    return vis_mask

@nb.njit(cache=True)
def get_view_exts(
    agent_dir: ndarray[np.int_],
    agent_pos: ndarray[np.int_],
    agent_view_size: int) -> ndarray[np.int_]:
    """
    Get the extents of the square set of grid cells visible to each agent.

    Parameters
    ----------
    agent_dir : ndarray[int] of shape (num_agents,)
        Direction of each agent
    agent_pos : ndarray[int] of shape (num_agents, 2)
        The (x, y) position of each agent
    agent_view_size : int
        Width and height of agent view

    Returns
    -------
    top_left : ndarray[int] of shape (num_agents, 2)
        The (x, y) coordinates of the top-left corner of each agent's observable view
    """
    agent_x, agent_y = agent_pos[:, 0], agent_pos[:, 1]
    top_left = np.zeros((agent_dir.shape[0], 2), dtype=np.int_)

    # Facing right
    top_left[agent_dir == RIGHT, 0] = agent_x[agent_dir == RIGHT]
    top_left[agent_dir == RIGHT, 1] = agent_y[agent_dir == RIGHT] - agent_view_size // 2

    # Facing down
    top_left[agent_dir == DOWN, 0] = agent_x[agent_dir == DOWN] - agent_view_size // 2
    top_left[agent_dir == DOWN, 1] = agent_y[agent_dir == DOWN]

    # Facing left
    top_left[agent_dir == LEFT, 0] = agent_x[agent_dir == LEFT] - agent_view_size + 1
    top_left[agent_dir == LEFT, 1] = agent_y[agent_dir == LEFT] - agent_view_size // 2

    # Facing up
    top_left[agent_dir == UP, 0] = agent_x[agent_dir == UP] - agent_view_size // 2
    top_left[agent_dir == UP, 1] = agent_y[agent_dir == UP] - agent_view_size + 1

    return top_left
