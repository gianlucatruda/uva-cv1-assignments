import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        height_map[:, 0] += np.cumsum(q[:, 0])
        height_map += np.cumsum(p, axis=1)

    elif path_type=='row':
        height_map[0] += np.cumsum(p[0])
        height_map += np.cumsum(q, axis=0)

    elif path_type=='average':
        h, w = p.shape
        height_map_rows = np.zeros([h, w])
        height_map_columns = np.zeros([h, w])

        height_map_rows[:, 0] += np.cumsum(q[:, 0])
        height_map_rows += np.cumsum(p, axis=1)

        height_map_columns[0] += np.cumsum(p[0])
        height_map_columns += np.cumsum(q, axis=0)

        height_map = (height_map_rows + height_map_columns)/2
        
    return height_map
        
