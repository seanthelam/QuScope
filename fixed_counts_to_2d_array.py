def counts_to_2d_array(counts, grid_size):
    """Convert quantum measurement counts to a 2D array for visualization.
    
    Parameters:
    ----------
    counts : dict
        Dictionary of bitstrings and their counts from quantum measurement
    grid_size : tuple
        Tuple (rows, cols) specifying the size of the output grid
        
    Returns:
    --------
    numpy.ndarray
        2D array with counts mapped to corresponding positions
    """
    import numpy as np
    
    # Create an empty result array
    result_array = np.zeros((grid_size[0], grid_size[1]))
    
    # Calculate the number of qubits needed to represent the grid
    n_qubits = grid_size[0] * grid_size[1]
    
    for bitstring, count in counts.items():
        # Truncate the bitstring to the expected length if needed
        truncated_bitstring = bitstring[-n_qubits:] if len(bitstring) >= n_qubits else bitstring.zfill(n_qubits)
        
        try:
            # Convert bitstring to integer index
            index = int(truncated_bitstring, 2)
            
            # Calculate row and column, ensuring they're within bounds
            row = min(index // grid_size[1], grid_size[0] - 1)
            col = min(index % grid_size[1], grid_size[1] - 1)
            
            # Print debug info
            print(f"Bitstring: {truncated_bitstring}, Index: {index}, Row: {row}, Col: {col}, Count: {count}")
            
            # Add the count to the result array
            result_array[row, col] += count
            
        except ValueError as e:
            print(f"Error processing bitstring {bitstring}: {e}")
    
    return result_array
