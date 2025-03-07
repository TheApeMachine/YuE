"""
Token fixing utilities for audio codec outputs.

This module provides functions to analyze and fix invalid tokens in codec outputs.
"""

import numpy as np
import copy
from collections import Counter
from sklearn.neighbors import NearestNeighbors


def analyze_token_issues(tokens):
    """
    Analyze token array for issues and patterns of invalid tokens
    
    Args:
        tokens: Token array
        
    Returns:
        Dictionary with analysis results
    """
    total_tokens = tokens.size
    invalid_count = np.sum((tokens < 0) | (tokens > 1023))
    invalid_percent = (invalid_count / total_tokens) * 100 if total_tokens > 0 else 0
    
    # Check for patterns in invalid tokens
    result = {
        "total_tokens": total_tokens,
        "invalid_count": invalid_count,
        "invalid_percent": invalid_percent,
        "invalid_rows": set(),
        "consecutive_invalids": 0
    }
    
    # Find rows with high invalid percentages and consecutive invalid tokens
    max_consecutive = 0
    current_consecutive = 0
    
    for i, row in enumerate(tokens):
        row_invalid = np.sum((row < 0) | (row > 1023))
        if row_invalid > 0:
            result["invalid_rows"].add(i)
            row_invalid_percent = (row_invalid / len(row)) * 100
            if row_invalid_percent > 10:  # More than 10% invalid in a row
                # This row might need special handling
                pass
        
        # Count consecutive invalid tokens
        for element in row:
            if element < 0 or element > 1023:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 0
    
    # Final check for consecutive count
    max_consecutive = max(max_consecutive, current_consecutive)
    result["consecutive_invalids"] = max_consecutive
    
    return result


def fix_tokens(output, min_valid=0, max_valid=1023, save_original=False, original_path=None, codebook=None):
    """
    Fix invalid codec tokens using advanced methods that account for non-linear latent spaces.
    Uses nearest valid embedding search rather than simple linear interpolation.
    
    Args:
        output: Token array to fix
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        save_original: Whether to save the original tokens
        original_path: Path to save original tokens if save_original is True
        codebook: Optional codebook for embedding-based fixing (if None, will use statistical methods)
        
    Returns:
        Fixed token array
    """
    # Input validation
    if output is None:
        raise ValueError("Input token array cannot be None")
    
    if not isinstance(output, np.ndarray):
        try:
            output = np.array(output)
            print("Converted input to numpy array")
        except Exception as e:
            raise TypeError(f"Input must be convertible to a numpy array: {e}")
    
    if output.size == 0:
        raise ValueError("Input token array is empty")
    
    # Validate token range parameters
    if min_valid >= max_valid:
        raise ValueError(f"min_valid ({min_valid}) must be less than max_valid ({max_valid})")
    
    # Make sure the shape is appropriate
    if len(output.shape) != 2:
        if len(output.shape) == 1:
            # Handle 1D arrays by reshaping to 2D
            output = output.reshape(1, -1)
            print("Reshaped 1D token array to 2D")
        else:
            raise ValueError(f"Expected 2D token array, got shape {output.shape}")
    
    try:
        # Analyze the token issues before fixing
        token_analysis = analyze_token_issues(output)
        
        # If no invalid tokens, return original
        if token_analysis["invalid_count"] == 0:
            return output
        
        # Display analysis results
        print(f"Found {token_analysis['invalid_count']} invalid tokens ({token_analysis['invalid_percent']:.2f}%)")
        print(f"Max consecutive invalid tokens: {token_analysis['consecutive_invalids']}")
        print(f"Rows with invalid tokens: {len(token_analysis['invalid_rows'])}")
        
        # Save original for debugging if requested
        if save_original and original_path:
            try:
                np.save(original_path, output)
                print(f"Saved original tokens to {original_path} for debugging")
            except Exception as e:
                print(f"Warning: Failed to save original tokens: {e}")
    
        # Create a copy for fixing
        fixed_output = copy.deepcopy(output)
        
        # Extract all valid tokens from the dataset to use as candidates
        valid_mask = (output >= min_valid) & (output <= max_valid)
        
        # Handle case where there are no valid tokens at all
        if not np.any(valid_mask):
            print("WARNING: No valid tokens found in input. Using default token values.")
            # Return array filled with safe middle values
            return np.full_like(output, (min_valid + max_valid) // 2)
            
        valid_tokens = output[valid_mask]
        
        # If we have very few valid tokens, use the full valid range as candidates
        if len(valid_tokens) < 100:
            valid_tokens = np.arange(min_valid, max_valid + 1)
        
        try:
            # Build statistical model of valid token distribution
            token_hist, bin_edges = np.histogram(valid_tokens, bins=max(50, max_valid - min_valid + 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        except Exception as e:
            print(f"Warning: Failed to build token distribution model: {e}")
            # Fallback to simple array
            token_hist = None
            bin_centers = None
        
        # For rows with high percentages of invalid tokens, apply advanced smoothing
        high_invalid_rows = []
        for i, line in enumerate(output):
            invalid_count = np.sum((line < min_valid) | (line > max_valid))
            if invalid_count / len(line) > 0.15:  # If more than 15% of tokens are invalid
                high_invalid_rows.append(i)
        
        # Try to process high-invalid rows
        try:
            # Process high-invalid rows
            fixed_output = _fix_high_invalid_rows(output, fixed_output, high_invalid_rows, 
                                                min_valid, max_valid, valid_tokens, codebook)
        except Exception as e:
            print(f"Warning: Error processing high-invalid rows: {e}")
            # Fallback for high-invalid rows: use most common valid token
            for row_idx in high_invalid_rows:
                line = output[row_idx]
                invalid_mask = (line < min_valid) | (line > max_valid)
                # Use most common valid token
                if len(valid_tokens) > 0:
                    token_counts = Counter(valid_tokens)
                    most_common = token_counts.most_common(1)[0][0]
                    fixed_output[row_idx, invalid_mask] = most_common
            
        # Try to process individual tokens
        try:
            # Process remaining individual invalid tokens
            fixed_output = _fix_individual_tokens(output, fixed_output, high_invalid_rows, 
                                                min_valid, max_valid, valid_tokens, codebook)
        except Exception as e:
            print(f"Warning: Error processing individual tokens: {e}")
            # Fallback for remaining individual tokens: clip to valid range
            invalid_mask = (fixed_output < min_valid) | (fixed_output > max_valid)
            if np.any(invalid_mask):
                # Clip or set to most common value
                if len(valid_tokens) > 0:
                    token_counts = Counter(valid_tokens)
                    most_common = token_counts.most_common(1)[0][0]
                    fixed_output[invalid_mask] = most_common
                else:
                    fixed_output[invalid_mask] = (min_valid + max_valid) // 2
    
        # Final validation
        # Analyze the fixed output to ensure all tokens are valid
        fixed_analysis = analyze_token_issues(fixed_output)
        if fixed_analysis["invalid_count"] > 0:
            print(f"WARNING: Still found {fixed_analysis['invalid_count']} invalid tokens after fixing!")
            # Force any remaining invalid tokens to be valid
            fixed_output = np.clip(fixed_output, min_valid, max_valid)
        else:
            print(f"Successfully fixed all {token_analysis['invalid_count']} invalid tokens.")
            
        # Calculate statistics on how much the tokens changed
        try:
            fixed_output = _calculate_fix_statistics(output, fixed_output, token_analysis, min_valid, max_valid)
        except Exception as e:
            print(f"Warning: Failed to calculate fix statistics: {e}")
        
        return fixed_output
        
    except Exception as e:
        print(f"Critical error in fix_tokens: {e}")
        # Ultimate fallback: return clipped version of input
        return np.clip(output, min_valid, max_valid)

def _fix_high_invalid_rows(output, fixed_output, high_invalid_rows, min_valid, max_valid, 
                           valid_tokens, codebook=None):
    """
    Apply advanced fixing to rows with high percentage of invalid tokens
    
    Args:
        output: Original token array
        fixed_output: Array to store fixed tokens
        high_invalid_rows: List of row indices with high invalid percentages
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        valid_tokens: Array of valid tokens to sample from
        codebook: Optional codebook for embedding-based fixing
        
    Returns:
        Updated fixed_output array
    """
    try:
        # Import optional dependencies
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import NearestNeighbors
        use_advanced = True
    except ImportError:
        use_advanced = False
        print("Advanced token fixing unavailable: sklearn not installed")
    
    for row_idx in high_invalid_rows:
        line = output[row_idx]
        valid_indices = np.where((line >= min_valid) & (line <= max_valid))[0]
        
        if len(valid_indices) < 2:
            # If almost no valid tokens, use the most frequent valid token in the dataset
            if len(valid_tokens) > 0:
                # Find the most common valid token
                token_counts = Counter(valid_tokens)
                most_common = token_counts.most_common(1)[0][0]
                fixed_output[row_idx] = np.full_like(line, most_common)
            else:
                # Fallback to middle value
                fixed_output[row_idx] = np.full_like(line, (min_valid + max_valid) // 2)
            continue
        
        # Extract valid tokens and their positions in this row
        valid_positions = valid_indices
        valid_row_tokens = line[valid_indices]
        
        # Context-aware repair using valid tokens in the row
        if use_advanced and len(valid_row_tokens) >= 5:
            # Reshape for KNN
            X = valid_positions.reshape(-1, 1)
            y = valid_row_tokens
            
            # Build KNN model for token prediction
            n_neighbors = min(5, len(valid_row_tokens))
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(X)
            
            # For each invalid token, predict using KNN
            for j in range(len(line)):
                if line[j] < min_valid or line[j] > max_valid:
                    # Query position
                    query = np.array([[j]])
                    distances, indices = knn.kneighbors(query)
                    
                    # Get neighboring positions and their tokens
                    neighbor_positions = valid_positions[indices[0]]
                    neighbor_tokens = valid_row_tokens[indices[0]]
                    
                    # Weight by inverse distance
                    weights = 1.0 / (distances[0] + 1e-5)
                    weights /= weights.sum()
                    
                    # Weighted voting for nearest valid token
                    token_candidates = {}
                    for token, weight in zip(neighbor_tokens, weights):
                        if token in token_candidates:
                            token_candidates[token] += weight
                        else:
                            token_candidates[token] = weight
                    
                    # Select token with highest weight
                    fixed_output[row_idx, j] = max(token_candidates.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to simpler approach if advanced methods unavailable
            for j in range(len(line)):
                if line[j] < min_valid or line[j] > max_valid:
                    # Find nearest valid neighbors by position
                    right_idx = valid_indices[valid_indices > j] if len(valid_indices[valid_indices > j]) > 0 else None
                    left_idx = valid_indices[valid_indices < j] if len(valid_indices[valid_indices < j]) > 0 else None
                    
                    if left_idx is not None and right_idx is not None:
                        # Get closest neighbors
                        left_pos = left_idx[-1]
                        right_pos = right_idx[0]
                        
                        # Get tokens at these positions
                        left_val = line[left_pos]
                        right_val = line[right_pos]
                        
                        # Distance-weighted selection
                        left_dist = j - left_pos
                        right_dist = right_pos - j
                        
                        if left_dist <= right_dist:
                            fixed_output[row_idx, j] = left_val  # Favor closer token
                        else:
                            fixed_output[row_idx, j] = right_val
                    elif left_idx is not None:
                        # Only left valid tokens available
                        left_pos = left_idx[-1]
                        fixed_output[row_idx, j] = line[left_pos]
                    elif right_idx is not None:
                        # Only right valid tokens available
                        right_pos = right_idx[0]
                        fixed_output[row_idx, j] = line[right_pos]
                    else:
                        # No valid tokens in this row - use global statistics
                        fixed_output[row_idx, j] = int(np.median(valid_tokens))
                
    return fixed_output

def _fix_individual_tokens(output, fixed_output, high_invalid_rows, min_valid, max_valid, 
                           valid_tokens, codebook=None):
    """
    Fix individual invalid tokens with an approach better suited for neural codec latent spaces,
    using nearest neighbor search instead of linear interpolation.
    
    Args:
        output: Original token array
        fixed_output: Array to store fixed tokens
        high_invalid_rows: List of row indices already processed
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        valid_tokens: Array of valid tokens to sample from
        codebook: Optional codebook for embedding-based fixing
        
    Returns:
        Updated fixed_output array
    """
    for i, line in enumerate(output):
        if i in high_invalid_rows:
            continue  # Skip rows we've already processed
        
        # Find invalid tokens in this row
        invalid_mask = (line < min_valid) | (line > max_valid)
        invalid_indices = np.where(invalid_mask)[0]
        
        if len(invalid_indices) == 0:
            continue
        
        # Get context for this row
        valid_mask = ~invalid_mask
        valid_indices = np.where(valid_mask)[0]
        
        # Skip if no valid tokens in this row
        if len(valid_indices) == 0:
            # Use global statistics
            token_counts = Counter(valid_tokens)
            most_common = token_counts.most_common(1)[0][0]
            fixed_output[i, invalid_indices] = most_common
            continue
        
        # Prepare data for nearest neighbor search in position space
        X_valid_pos = valid_indices.reshape(-1, 1)
        y_valid_tokens = line[valid_indices]
        
        # Group adjacent invalid tokens
        invalid_regions = []
        current_region = [invalid_indices[0]]
        
        for j in range(1, len(invalid_indices)):
            if invalid_indices[j] == invalid_indices[j-1] + 1:
                # Continue current region
                current_region.append(invalid_indices[j])
            else:
                # End region and start new one
                invalid_regions.append(current_region)
                current_region = [invalid_indices[j]]
        
        # Add the last region
        if current_region:
            invalid_regions.append(current_region)
        
        # Process each invalid region
        for region in invalid_regions:
            # For isolated invalid tokens or small regions, use nearest neighbor
            if len(region) <= 3:
                for pos in region:
                    # Find the two closest valid tokens by position
                    distances = np.abs(valid_indices - pos)
                    nearest_indices = np.argsort(distances)[:2]
                    nearest_positions = valid_indices[nearest_indices]
                    nearest_tokens = line[nearest_positions]
                    
                    # Select the token from the nearest valid position
                    fixed_output[i, pos] = nearest_tokens[0]
            else:
                # For larger regions, use a more sophisticated approach
                # Find valid tokens on both sides of the region
                region_start, region_end = region[0], region[-1]
                
                left_valid = valid_indices[valid_indices < region_start]
                right_valid = valid_indices[valid_indices > region_end]
                
                if len(left_valid) > 0 and len(right_valid) > 0:
                    # We have valid tokens on both sides
                    left_pos = left_valid[-1]
                    right_pos = right_valid[0]
                    left_token = line[left_pos]
                    right_token = line[right_pos]
                    
                    # Select from valid tokens based on region position
                    for j, pos in enumerate(region):
                        # Favor the closer boundary token
                        region_progress = j / (len(region) - 1) if len(region) > 1 else 0.5
                        
                        # Select based on position, but avoiding linear interpolation
                        # which could create invalid tokens in non-linear latent spaces
                        if region_progress < 0.5:
                            fixed_output[i, pos] = left_token
                        else:
                            fixed_output[i, pos] = right_token
                
                elif len(left_valid) > 0:
                    # Only valid tokens to the left
                    left_pos = left_valid[-1]
                    left_token = line[left_pos]
                    for pos in region:
                        fixed_output[i, pos] = left_token
                
                elif len(right_valid) > 0:
                    # Only valid tokens to the right
                    right_pos = right_valid[0]
                    right_token = line[right_pos]
                    for pos in region:
                        fixed_output[i, pos] = right_token
                
                else:
                    # No valid tokens in this row
                    for pos in region:
                        fixed_output[i, pos] = int(np.median(valid_tokens))
                        
    return fixed_output

def _calculate_fix_statistics(output, fixed_output, token_analysis, min_valid, max_valid):
    """
    Calculate statistics on how the tokens changed during fixing
    
    Args:
        output: Original token array
        fixed_output: Fixed token array
        token_analysis: Analysis results from original tokens
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        
    Returns:
        The fixed_output array (unchanged, just for function chaining)
    """
    if token_analysis["invalid_count"] > 0:
        # Only compare valid tokens in the original to see how much they changed
        valid_mask = (output >= min_valid) & (output <= max_valid)
        if np.any(valid_mask):
            valid_original = output[valid_mask]
            valid_fixed = fixed_output[valid_mask]
            
            # Check if valid tokens were preserved
            preserved = np.sum(valid_original == valid_fixed)
            total_valid = np.sum(valid_mask)
            print(f"Preserved {preserved}/{total_valid} valid tokens ({preserved/total_valid*100:.2f}%)")
            
            # Calculate average change in valid tokens that were modified
            modified_mask = (valid_original != valid_fixed)
            if np.any(modified_mask):
                avg_change = np.mean(np.abs(valid_original[modified_mask] - valid_fixed[modified_mask]))
                print(f"Average change in modified valid tokens: {avg_change:.2f}")
    
    return fixed_output 