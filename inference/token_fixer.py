import numpy as np
import copy
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Optional: for progress bars
from sklearn.neighbors import NearestNeighbors  # for KNN if available


def get_masks_and_indices(tokens, min_valid, max_valid):
    """
    Return valid/invalid masks and valid indices for the given token array.
    
    Args:
        tokens: 1D or 2D numpy array of tokens.
        min_valid: Minimum valid token value.
        max_valid: Maximum valid token value.
        
    Returns:
        valid_mask: Boolean mask where tokens are valid.
        invalid_mask: Boolean mask where tokens are invalid.
        valid_indices: Indices where tokens are valid (only meaningful if tokens are 1D).
    """
    invalid_mask = (tokens < min_valid) | (tokens > max_valid)
    valid_mask = ~invalid_mask
    valid_indices = np.nonzero(valid_mask)[0] if tokens.ndim == 1 else None
    return valid_mask, invalid_mask, valid_indices


def get_fallback_token(valid_tokens, min_valid, max_valid):
    """
    Get a fallback token from the most common valid token or a midpoint if none found.
    """
    if len(valid_tokens) > 0:
        token_counts = Counter(valid_tokens)
        return token_counts.most_common(1)[0][0]
    else:
        return (min_valid + max_valid) // 2


def analyze_token_issues(tokens, min_valid=0, max_valid=1023):
    """
    Analyze token array for issues: how many are invalid, consecutive invalid runs, etc.
    Returns a dictionary of stats.
    """
    total_tokens = tokens.size
    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "invalid_count": 0,
            "invalid_percent": 0.0,
            "invalid_rows": set(),
            "consecutive_invalids": 0,
        }

    # We treat each row individually if it's 2D.
    # If it's 1D, treat it like a single row.
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)

    invalid_mask = (tokens < min_valid) | (tokens > max_valid)
    invalid_count = np.sum(invalid_mask)
    invalid_percent = (invalid_count / total_tokens) * 100 if total_tokens > 0 else 0

    max_consecutive = 0
    invalid_rows = set()
    
    for i, row in enumerate(tokens):
        row_invalid = invalid_mask[i]
        if np.any(row_invalid):
            invalid_rows.add(i)

        # Count consecutive invalid in this row
        consecutive = 0
        for val in row_invalid:
            if val:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

    return {
        "total_tokens": total_tokens,
        "invalid_count": invalid_count,
        "invalid_percent": invalid_percent,
        "invalid_rows": invalid_rows,
        "consecutive_invalids": max_consecutive,
    }


def _fix_invalid_region(
    row, region, valid_indices, fallback_token
):
    """
    Fix a consecutive region of invalid tokens by either position-based or boundary-based approach.
    If region length <= 3, we do a local position-based fix.
    If region length > 3, we do a boundary-based fix.
    
    Returns a dict of {invalid_index: fixed_token_value}.
    """
    fixes = {}
    region_len = len(region)
    if region_len == 0:
        return fixes

    # Decide strategy
    strategy = "position" if region_len <= 3 else "boundary"

    # If using boundary approach, find outer valid tokens on each side
    region_start, region_end = region[0], region[-1]
    left_side = valid_indices[valid_indices < region_start]
    right_side = valid_indices[valid_indices > region_end]

    left_token = row[left_side[-1]] if len(left_side) > 0 else None
    right_token = row[right_side[0]] if len(right_side) > 0 else None

    if strategy == "boundary":
        # We gradually interpolate between left_token and right_token by position,
        # but since these tokens aren't numeric in a strict sense, we can just
        # do a simple "use left if < halfway, else use right".
        for idx_offset, invalid_pos in enumerate(region):
            if left_token is not None and right_token is not None:
                progress = idx_offset / (region_len - 1) if region_len > 1 else 0.5
                fixes[invalid_pos] = left_token if progress < 0.5 else right_token
            elif left_token is not None:
                fixes[invalid_pos] = left_token
            elif right_token is not None:
                fixes[invalid_pos] = right_token
            else:
                fixes[invalid_pos] = fallback_token

    else:
        # Position-based fix for short regions.
        for invalid_pos in region:
            # Attempt to find nearest valid token by position:
            dists = np.abs(valid_indices - invalid_pos)
            nearest_idx = valid_indices[np.argmin(dists)] if len(dists) > 0 else None
            if nearest_idx is not None:
                fixes[invalid_pos] = row[nearest_idx]
            else:
                fixes[invalid_pos] = fallback_token

    return fixes


def _fix_invalid_tokens_in_row(
    row, min_valid, max_valid, fallback_token, use_knn
):
    """
    Given a single row of tokens, fix all invalid entries using:
      1) KNN-based approach if row isn't too heavily invalid and enough valid points exist.
      2) Otherwise group invalid tokens into regions and fix them with boundary/position approach.
    
    Returns a dict { invalid_index: fixed_value }.
    """
    _, invalid_mask, valid_indices = get_masks_and_indices(row, min_valid, max_valid)
    invalid_indices = np.nonzero(invalid_mask)[0]
    
    # If no invalid tokens, nothing to fix.
    if len(invalid_indices) == 0:
        return {}

    # If no valid tokens, fix all invalid with fallback immediately.
    if len(valid_indices) == 0:
        return {idx: fallback_token for idx in invalid_indices}

    fixes = {}
    row_invalid_percent = 100.0 * len(invalid_indices) / len(row)
    is_high_invalid = row_invalid_percent > 15.0

    # Attempt KNN fix if we have enough valid indices, not too many invalids
    # and if the user wants to use_knn
    if use_knn and len(valid_indices) >= 3 and not is_high_invalid:
        try:
            X = valid_indices.reshape(-1, 1)
            knn = NearestNeighbors(n_neighbors=min(3, len(X)))
            knn.fit(X)

            # Distances, neighbor-indices for each invalid index
            dist, idx_neighbors = knn.kneighbors(invalid_indices.reshape(-1, 1))
            weights = 1.0 / (dist + 1e-9)
            weights /= np.sum(weights, axis=1, keepdims=True)

            for i, inv_idx in enumerate(invalid_indices):
                neighbor_positions = valid_indices[idx_neighbors[i]]
                neighbor_tokens = row[neighbor_positions]

                # Weighted "vote" on neighbor tokens:
                token_score = {}
                for neighbor_tkn, w in zip(neighbor_tokens, weights[i]):
                    token_score[neighbor_tkn] = token_score.get(neighbor_tkn, 0.0) + w

                best_token = max(token_score.items(), key=lambda x: x[1])[0]
                fixes[inv_idx] = best_token

        except Exception:
            # If KNN fails, we do region-based
            pass

    # Identify which invalid indices are unfixed (need region-based approach)
    unfixed = [idx for idx in invalid_indices if idx not in fixes]

    if unfixed:
        # Group unfixed invalid indices into consecutive regions
        regions = []
        cur_region = [unfixed[0]]
        for idx in unfixed[1:]:
            if idx == cur_region[-1] + 1:
                cur_region.append(idx)
            else:
                regions.append(cur_region)
                cur_region = [idx]
        regions.append(cur_region)

        # Fix each region
        for reg in regions:
            region_fixes = _fix_invalid_region(
                row, reg, valid_indices, fallback_token
            )
            fixes.update(region_fixes)

    return fixes


def fix_tokens(
    output,
    min_valid=0,
    max_valid=1023,
    save_original=False,
    original_path=None,
    codebook=None,  # if needed
    parallel=True,
    show_progress=True,
):
    """
    Fix invalid codec tokens using advanced methods that account for non-linear latent spaces.
    Uses nearest valid embedding search (KNN) if available, and fallback region-based approach.
    
    Args:
        output: Token array to fix (1D or 2D).
        min_valid: Minimum valid token value.
        max_valid: Maximum valid token value.
        save_original: Whether to save the original tokens.
        original_path: Path to save original tokens if save_original is True.
        codebook: Optional codebook (not actively used in this refactoring, but kept to match the signature).
        parallel: Whether to use parallel processing for large arrays.
        show_progress: Whether to show progress bar.
        
    Returns:
        Fixed token array (2D).
    """
    if output is None:
        raise ValueError("Input token array cannot be None")

    # Ensure numpy array
    if not isinstance(output, np.ndarray):
        output = np.array(output)

    if output.size == 0:
        raise ValueError("Input token array is empty")

    # Validate min/max
    if min_valid >= max_valid:
        raise ValueError(f"min_valid ({min_valid}) must be < max_valid ({max_valid})")

    # Reshape to 2D if 1D
    reshaped = False
    if output.ndim == 1:
        output = output.reshape(1, -1)
        reshaped = True

    # Analyze tokens
    analysis = analyze_token_issues(output, min_valid, max_valid)
    if analysis["invalid_count"] == 0:
        print("No invalid tokens found. Nothing to fix.")
        return output if not reshaped else output.flatten()

    print(
        f"Found {analysis['invalid_count']} invalid tokens "
        f"({analysis['invalid_percent']:.2f}%). "
        f"Max consecutive invalid: {analysis['consecutive_invalids']}."
    )

    # Save original if requested
    if save_original and original_path:
        try:
            np.save(original_path, output)
            print(f"Saved original tokens to {original_path}")
        except Exception as e:
            print(f"Warning: Failed to save original tokens: {e}")

    # Copy for fixing
    fixed_output = copy.deepcopy(output)

    # Gather valid tokens from entire array
    overall_valid_mask = (fixed_output >= min_valid) & (fixed_output <= max_valid)
    if not np.any(overall_valid_mask):
        # If absolutely no valid tokens, just fill with midpoint
        print("No valid tokens at all in the array. Filling everything with fallback.")
        fallback = (min_valid + max_valid) // 2
        return np.full_like(fixed_output, fallback)

    valid_tokens = fixed_output[overall_valid_mask]

    # If valid tokens are sparse, generate synthetic valid tokens
    if len(valid_tokens) < 10:
        print(
            f"WARNING: Very few valid tokens ({len(valid_tokens)}). "
            "Using fallback approach with synthetic valid tokens."
        )
        synthetic = np.linspace(min_valid, max_valid, 100, dtype=int)
        valid_tokens = np.concatenate([valid_tokens, synthetic]) if len(valid_tokens) else synthetic

    fallback_token = get_fallback_token(valid_tokens, min_valid, max_valid)
    # Attempt to see if KNN can be used
    use_knn = True  # We already imported NearestNeighbors at the top

    # Prepare row data
    row_data = [
        (i, fixed_output[i], min_valid, max_valid, fallback_token, use_knn)
        for i in range(fixed_output.shape[0])
    ]

    # Process rows in parallel or sequentially
    def _sequential_fix(rd_list):
        iterator = tqdm(rd_list) if show_progress else rd_list
        for row_args in iterator:
            i, fixes = _process_row_wrapper(row_args)
            for pos, tok in fixes.items():
                fixed_output[i, pos] = tok

    def _process_row_wrapper(args):
        i, row, mn, mx, fb, knn_flag = args
        fixes = _fix_invalid_tokens_in_row(row, mn, mx, fb, knn_flag)
        return i, fixes

    try:
        if parallel and fixed_output.shape[0] > 10 and fixed_output.size > 10000:
            num_workers = min(cpu_count(), 8)
            print(f"Using parallel processing with {num_workers} workers.")
            with Pool(num_workers) as pool:
                if show_progress:
                    results = list(tqdm(pool.imap(_process_row_wrapper, row_data), total=len(row_data)))
                else:
                    results = pool.map(_process_row_wrapper, row_data)
            for i, fixes in results:
                for pos, tok in fixes.items():
                    fixed_output[i, pos] = tok
        else:
            _sequential_fix(row_data)
    except Exception as e:
        print(f"Parallel processing failed ({e}). Falling back to sequential.")
        _sequential_fix(row_data)

    # Clip any remaining invalid tokens as final fallback
    final_mask = (fixed_output < min_valid) | (fixed_output > max_valid)
    remaining_invalid = np.sum(final_mask)
    if remaining_invalid > 0:
        print(
            f"WARNING: Still found {remaining_invalid} invalid tokens after fixing; "
            "they will be clipped."
        )
        np.clip(fixed_output, min_valid, max_valid, out=fixed_output)
    else:
        print(f"All {analysis['invalid_count']} invalid tokens successfully fixed.")

    # Optionally compare how many valid tokens were changed
    try:
        original_valid = output[overall_valid_mask]
        new_valid = fixed_output[overall_valid_mask]
        preserved = np.sum(original_valid == new_valid)
        total_valid = len(original_valid)
        print(
            f"Preserved {preserved}/{total_valid} originally valid tokens "
            f"({(preserved / total_valid) * 100:.2f}%)."
        )
    except Exception as e:
        print(f"Failed to compute preservation stats: {e}")

    # Reshape back if needed
    return fixed_output if not reshaped else fixed_output.flatten()
