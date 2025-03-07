"""
Token fixing utilities for audio codec outputs.

This module provides functions to analyze and fix invalid tokens in codec outputs.
"""

import numpy as np
import copy
from collections import Counter


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


def fix_tokens(output, min_valid=0, max_valid=1023, save_original=False, original_path=None):
    """
    Fix invalid codec tokens by using interpolation and smoothing.
    
    Args:
        output: Token array to fix
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        save_original: Whether to save the original tokens
        original_path: Path to save original tokens if save_original is True
        
    Returns:
        Fixed token array
    """
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
        np.save(original_path, output)
        print(f"Saved original tokens to {original_path} for debugging")

    # Create a copy for fixing
    fixed_output = copy.deepcopy(output)
    
    # For rows with high percentages of invalid tokens, apply smoothing
    high_invalid_rows = []
    for i, line in enumerate(output):
        invalid_count = np.sum((line < min_valid) | (line > max_valid))
        if invalid_count / len(line) > 0.15:  # If more than 15% of tokens are invalid
            high_invalid_rows.append(i)
    
    # Process high-invalid rows
    fixed_output = _fix_high_invalid_rows(output, fixed_output, high_invalid_rows, min_valid, max_valid)
    
    # Process remaining individual invalid tokens
    fixed_output = _fix_individual_tokens(output, fixed_output, high_invalid_rows, min_valid, max_valid)

    # Analyze the fixed output to ensure all tokens are valid
    fixed_analysis = analyze_token_issues(fixed_output)
    if fixed_analysis["invalid_count"] > 0:
        print(f"WARNING: Still found {fixed_analysis['invalid_count']} invalid tokens after fixing!")
        # Force any remaining invalid tokens to be valid
        fixed_output = np.clip(fixed_output, min_valid, max_valid)
    else:
        print(f"Successfully fixed all {token_analysis['invalid_count']} invalid tokens.")
        
    # Calculate statistics on how much the tokens changed
    fixed_output = _calculate_fix_statistics(output, fixed_output, token_analysis, min_valid, max_valid)
    
    return fixed_output

def _fix_high_invalid_rows(output, fixed_output, high_invalid_rows, min_valid, max_valid):
    """
    Apply a more aggressive smoothing to rows with high percentage of invalid tokens
    
    Args:
        output: Original token array
        fixed_output: Array to store fixed tokens
        high_invalid_rows: List of row indices with high invalid percentages
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        
    Returns:
        Updated fixed_output array
    """
    for row_idx in high_invalid_rows:
        line = output[row_idx]
        valid_indices = np.where((line >= min_valid) & (line <= max_valid))[0]
        
        if len(valid_indices) < 2:
            # If almost no valid tokens, fill with safe value
            fixed_output[row_idx] = np.full_like(line, (min_valid + max_valid) // 2)
            continue
            
        # Create a smoothed version using valid tokens
        valid_tokens = line[valid_indices]
        
        # Interpolate between valid tokens for all positions
        for j in range(len(line)):
            if line[j] >= min_valid and line[j] <= max_valid:
                # Keep valid tokens as they are
                continue
                
            # Find nearest valid indices
            right_idx = valid_indices[valid_indices > j]
            left_idx = valid_indices[valid_indices < j]
            
            if len(left_idx) > 0 and len(right_idx) > 0:
                # Interpolate between nearest valid tokens
                left_pos = left_idx[-1]
                right_pos = right_idx[0]
                left_val = line[left_pos]
                right_val = line[right_pos]
                weight = (j - left_pos) / (right_pos - left_pos)
                fixed_output[row_idx, j] = int(left_val * (1 - weight) + right_val * weight)
            elif len(left_idx) > 0:
                # Only left valid tokens available
                left_pos = left_idx[-1]
                fixed_output[row_idx, j] = line[left_pos]
            elif len(right_idx) > 0:
                # Only right valid tokens available
                right_pos = right_idx[0]
                fixed_output[row_idx, j] = line[right_pos]
                
    return fixed_output

def _fix_individual_tokens(output, fixed_output, high_invalid_rows, min_valid, max_valid):
    """
    Fix individual invalid tokens in rows that don't have high percentages of invalids
    
    Args:
        output: Original token array
        fixed_output: Array to store fixed tokens
        high_invalid_rows: List of row indices already processed with aggressive smoothing
        min_valid: Minimum valid token value
        max_valid: Maximum valid token value
        
    Returns:
        Updated fixed_output array
    """
    for i, line in enumerate(output):
        if i in high_invalid_rows:
            continue  # Skip rows we've already processed with smoothing
            
        for j, element in enumerate(line):
            if element < min_valid or element > max_valid:
                # Find nearest valid neighbors for interpolation
                left_idx, right_idx = j-1, j+1
                left_valid, right_valid = False, False
                
                # Find valid left neighbor
                while left_idx >= 0 and not left_valid:
                    if min_valid <= line[left_idx] <= max_valid:
                        left_valid = True
                    else:
                        left_idx -= 1
                
                # Find valid right neighbor
                while right_idx < len(line) and not right_valid:
                    if min_valid <= line[right_idx] <= max_valid:
                        right_valid = True
                    else:
                        right_idx += 1
                
                # Determine replacement value based on valid neighbors
                if left_valid and right_valid:
                    # Interpolate between valid neighbors
                    left_val = line[left_idx]
                    right_val = line[right_idx]
                    weight = (j - left_idx) / (right_idx - left_idx)
                    fixed_output[i, j] = int(left_val * (1 - weight) + right_val * weight)
                elif left_valid:
                    # Use left neighbor
                    fixed_output[i, j] = line[left_idx]
                elif right_valid:
                    # Use right neighbor
                    fixed_output[i, j] = line[right_idx]
                else:
                    # Fallback to most frequent token if no valid neighbors
                    counter = Counter(line)
                    valid_tokens = {k: v for k, v in counter.items() if min_valid <= k <= max_valid}
                    if valid_tokens:
                        most_frequent = sorted(valid_tokens.items(), key=lambda x: x[1], reverse=True)[0][0]
                        fixed_output[i, j] = most_frequent
                    else:
                        # If no valid tokens in line, use a safe middle value
                        fixed_output[i, j] = (min_valid + max_valid) // 2
                        
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