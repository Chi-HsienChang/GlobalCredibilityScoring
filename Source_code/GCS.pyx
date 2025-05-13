#distutils: extra_link_args=-fopenmp
from cython import parallel
from cython.parallel import prange
import numpy as np
import time
from sklearn.neighbors import KernelDensity
from awkde import GaussianKDE
import math
from math import exp 
from libc.math cimport exp as c_exp
cimport openmp



def keep_top_k_states(state_dict, k):
    # If the number of states is less than or equal to k, return the original dictionary
    if len(state_dict) <= k:
        return state_dict

    # Sort the items in the dictionary by score in descending order
    sorted_items = sorted(state_dict.items(), key=lambda x: x[1], reverse=True)

    # Initialize a new dictionary to store the top-k states
    new_dict = {}

    # Keep only the top-k highest-scoring states
    for i in range(k):
        st, score = sorted_items[i]
        new_dict[st] = score

    return new_dict  # Return the dictionary with only the top-k states


cdef double logsumexp(list vals):
    # Get the number of elements in the input list
    cdef int n = len(vals)
    
    # If the list is empty, return negative infinity (log of 0)
    if n == 0:
        return float('-inf')

    # Find the maximum value in the list to use for numerical stability
    cdef double max_val = vals[0]
    cdef int i
    for i in range(1, n):
        if vals[i] > max_val:
            max_val = vals[i]

    # If all values are negative infinity, return negative infinity
    if max_val == float('-inf'):
        return float('-inf')

    # Compute the sum of exp(x - max_val) for all x in the list
    cdef double total = 0.0, x
    for i in range(n):
        x = vals[i] - max_val
        total += math.exp(x)

    # Return the log-sum-exp: log(sum(exp(x))) computed in a numerically stable way
    return max_val + math.log(total)



cdef tuple transition_dp(
    tuple state, double log_score, int pos, int symbol, 
    object sequences, int length,
    double pME, double[:] pELF, double[:] pIL, 
    double pEE, double[:] pELM,
    double[:] emissions5, double[:] emissions3
):
    cdef:
        # Unpack current state values
        int used5       = state[0]  # Number of 5' splice sites used
        int used3       = state[1]  # Number of 3' splice sites used
        int lastSymbol  = state[2]  # Last symbol used (0, 5, or 3)
        int zeroCount   = state[3]  # Count of consecutive non-splice symbols (symbol == 0)
        int last5Pos    = state[4]  # Position of the last 5' splice site
        int last3Pos    = state[5]  # Position of the last 3' splice site

        # Initialize new state values
        double new_log_score = log_score
        int newUsed5         = used5
        int newUsed3         = used3
        int newZeroCount     = zeroCount
        int newLast5Pos      = last5Pos
        int newLast3Pos      = last3Pos
        int gap_5, gap_3
        int newLastSymbol

    # Case: symbol is 0 (non-splice site)
    if symbol == 0:
        if lastSymbol == 5 or lastSymbol == 3:
            newZeroCount = zeroCount + 1
        newLastSymbol = 0

    # Case: symbol is 5 (5' splice site)
    elif symbol == 5:
        # Skip if emission score is invalid or boundary condition
        if emissions5[pos] <= float('-inf'):
            return None
        if pos + 1 >= length:
            return None
        # Check for GT motif at 5' splice site
        if not (sequences[pos] == 'G' and sequences[pos+1] == 'T'):
            return None
        # Enforce transition and structure constraints
        if lastSymbol == 5 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')) or (used5 != used3):
            return None

        # First 5'SS gets initial probability; others depend on gap length
        if used5 == 0:
            new_log_score += pME + pELF[pos - 1] + emissions5[pos]
        else:
            gap_5 = (pos - last3Pos) - 2
            if gap_5 < 0 or gap_5 >= pELM.shape[0]:
                return None
            new_log_score += pEE + pELM[gap_5] + emissions5[pos]

        newUsed5 = used5 + 1
        newLast5Pos = pos
        newZeroCount = 0
        newLastSymbol = 5

    # Case: symbol is 3 (3' splice site)
    elif symbol == 3:
        # Skip if emission score is invalid or boundary condition
        if emissions3[pos] <= float('-inf'):
            return None
        if pos - 1 < 0:
            return None
        # Check for AG motif at 3' splice site
        if not (sequences[pos] == 'G' and sequences[pos-1] == 'A'):
            return None
        # Enforce transition and structure constraints
        if lastSymbol == 3 or (((lastSymbol == 5) or (lastSymbol == 3)) and zeroCount < float('inf')) or (used5 != used3 + 1):
            return None

        gap_3 = (pos - last5Pos) - 2
        if gap_3 < 0 or gap_3 >= pIL.shape[0]:
            return None
        new_log_score += pIL[gap_3] + emissions3[pos]

        newUsed3 = used3 + 1
        newLast3Pos = pos
        newZeroCount = 0
        newLastSymbol = 3

    else:
        return None  # Invalid symbol

    # Pack updated state and return with updated log-score
    cdef tuple new_state = (newUsed5, newUsed3, newLastSymbol, newZeroCount, newLast5Pos, newLast3Pos)
    return (new_state, new_log_score)


cdef dict forward_dp_step(
    dict F_prev, 
    int pos, 
    object sequences, 
    int length,
    double pME, 
    double[:] pELF, 
    double[:] pIL, 
    double pEE, 
    double[:] pELM, 
    double[:] emissions5, 
    double[:] emissions3,
    int top_k
):
    # Initialize the current forward dynamic programming dictionary
    cdef dict F_curr = {}

    # Determine allowed symbols at this position
    # Only symbol 0 is allowed at the first and last positions
    cdef list allowed_symbols
    if pos == 0 or pos == length - 1:
        allowed_symbols = [0]
    else:
        allowed_symbols = [0, 5, 3]

    cdef tuple state, new_state_tuple, new_state
    cdef double alpha_score, new_log_score
    cdef int symbol

    # Iterate over all previous states and their scores
    for state, alpha_score in F_prev.items():
        # Skip invalid or undefined scores
        if alpha_score == float('-inf') or math.isnan(alpha_score):
            continue

        # Try each allowed symbol transition from the current state
        for symbol in allowed_symbols:
            new_state_tuple = transition_dp(
                state, alpha_score, pos, symbol,
                sequences, length,
                pME, pELF, pIL, pEE, pELM,
                emissions5, emissions3
            )

            # Skip invalid transitions
            if new_state_tuple is None:
                continue

            # Unpack the new state and score from the transition result
            new_state, new_log_score = new_state_tuple

            # If this new state already exists, accumulate score using logsumexp
            if new_state in F_curr:
                F_curr[new_state] = logsumexp([F_curr[new_state], new_log_score])
            else:
                F_curr[new_state] = new_log_score

    # Keep only the top-k states with the highest scores to control memory usage
    F_curr = keep_top_k_states(F_curr, top_k)

    return F_curr  # Return the updated state dictionary for this position

##########################
#### Compute GCS_SS ####
##########################

cpdef tuple GCS_SS(
    object sequences,
    double pME,
    double[:] pELF,
    double[:] pIL,
    double pEE,
    double[:] pELM,
    double pEO,
    double[:] pELL,
    double[:] emissions5,
    double[:] emissions3,
    int length,
    int checkpoint_interval, 
    int top_k_parse                 
):
    cdef dict checkpoints = {}  
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()

    cdef int top_k = top_k_parse 
    cdef int pos

    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM, emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()

    cdef dict B_current = {}
    cdef double tail
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
     
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index >= 0 and ell_index < pELL.shape[0]:
                tail += pEO + pELL[ell_index]
            B_current[state] = tail

    cdef double logZ
    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    if terminal_logs:
        logZ = logsumexp(terminal_logs)
    else:
        logZ = float('-inf')


    cdef list post_list = [ {} for _ in range(length) ]
    cdef list ckpt_positions = sorted(checkpoints.keys())
    cdef dict B_next_segment = B_current
    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F 
    cdef list seg_B 
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double alpha_val, new_log_score
    cdef double b_val 
    cdef double val 
    cdef double prob
    cdef int sym 
    cdef list allowed_symbols

    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start


        seg_F = [None] * (seg_len + 1)
        seg_F[0] = checkpoints[seg_start].copy()

        for j in range(0, seg_len):
            global_pos = seg_start + j
            seg_F[j + 1] = forward_dp_step(
                seg_F[j], global_pos, sequences, length,
                pME, pELF, pIL, pEE, pELM, 
                emissions5, emissions3,
                top_k
            )


        seg_B = [None] * (seg_len + 1)
        seg_B[seg_len] = B_next_segment
      
        for j in range(seg_len - 1, -1, -1):
            global_pos = seg_start + j
            B_seg = {}

            if global_pos == 0 or global_pos == length - 1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            for state, alpha_val in seg_F[j].items():
                contributions = []
                for sym in allowed_symbols:
                    new_state_tuple = transition_dp(
                        state, alpha_val, global_pos, sym,
                        sequences, length,
                        pME, pELF, pIL, pEE, pELM,
                        emissions5, emissions3
                    )
                    if new_state_tuple is None:
                        continue
                    new_state, new_log_score = new_state_tuple
                    if new_state in seg_B[j + 1]:
                        contributions.append(new_log_score - alpha_val + seg_B[j + 1][new_state])

                if contributions:
                    B_seg[state] = logsumexp(contributions)

            seg_B[j] = B_seg

            for state, alpha_val in seg_F[j].items():
                if state in seg_B[j]:
                    b_val = seg_B[j][state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)
                            sym = state[2]
                            if sym in post_list[global_pos]:
                                post_list[global_pos][sym] += prob
                            else:
                                post_list[global_pos][sym] = prob

        B_next_segment = seg_B[0]

    return post_list, logZ


##########################
#### Compute GCS_Exon ####
##########################

cpdef tuple GCS_Exon(
    object sequences,
    double pME,
    double[:] pELF,
    double[:] pIL,
    double pEE,
    double[:] pELM,
    double pEO,
    double[:] pELL,
    double[:] emissions5,
    double[:] emissions3,
    int length,
    int checkpoint_interval,
    int top_k_parse
):
    cdef dict checkpoints = {}
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()
    cdef int top_k = top_k_parse 
    cdef int pos
    # Forward pass: 儲存 checkpoint
    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM,
            emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()

    cdef dict B_current = {}
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    cdef double alpha_score, tail
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
 
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index > 0 and ell_index < pELL.shape[0]:
                tail = pEO + pELL[ell_index]
            B_current[state] = tail

    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    cdef double common_logZ = float('-inf')
    if terminal_logs:
        common_logZ = logsumexp(terminal_logs)

    cdef dict first_exon_dict = {}   
    cdef dict exon_dict = {}        
    cdef list ckpt_positions = sorted(checkpoints.keys())
    cdef dict B_next_segment = B_current
    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F, seg_B
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double new_log_score, b_val, val, prob, alpha_val
    cdef list allowed_symbols
    cdef int sym, a_pos, b_pos
    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start
        # 區段 forward.
        seg_F = [None]*(seg_len+1)
        seg_F[0] = checkpoints[seg_start].copy()
        for j in range(seg_len):
            global_pos = seg_start + j
            seg_F[j+1] = forward_dp_step(
                seg_F[j], global_pos, sequences, length,
                pME, pELF, pIL, pEE, pELM,
                emissions5, emissions3,
                top_k)
   
        seg_B = [None]*(seg_len+1)
        seg_B[seg_len] = B_next_segment
        for j in range(seg_len-1, -1, -1):
            global_pos = seg_start + j
            B_seg = {}
            if global_pos == 0 or global_pos == length-1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]
            for state, alpha_val in seg_F[j].items():
                if alpha_val == float('-inf') or math.isnan(alpha_val):
                    continue
                contributions = []
                for sym in allowed_symbols:
                    new_state_tuple = transition_dp(
                        state, alpha_val, global_pos, sym,
                        sequences, length,
                        pME, pELF, pIL, pEE, pELM,
                        emissions5, emissions3
                    )
                    if new_state_tuple is None:
                        continue
                    new_state, new_log_score = new_state_tuple
                    if new_state in seg_B[j+1]:
                        contributions.append(new_log_score - alpha_val + seg_B[j+1][new_state])
                if contributions:
                    B_seg[state] = logsumexp(contributions)
            seg_B[j] = B_seg
            for state, alpha_val in seg_F[j].items():
                if state in B_seg:
                    b_val = B_seg[state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - common_logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)
     
                            if state[2] == 5:
                                if state[5] < 0:
                                    b_pos = global_pos
                                    if (0, b_pos) in first_exon_dict:
                                        first_exon_dict[(0, b_pos)] += prob
                                    else:
                                        first_exon_dict[(0, b_pos)] = prob
                                else:
                                    a_pos = state[5]
                                    b_pos = global_pos
                                    if (a_pos, b_pos) in exon_dict:
                                        exon_dict[(a_pos, b_pos)] += prob
                                    else:
                                        exon_dict[(a_pos, b_pos)] = prob
        B_next_segment = seg_B[0]
    return first_exon_dict, exon_dict, common_logZ, F_current, B_current


cpdef tuple forward_backward_last_exon(
    dict F_current,  
    dict B_current,  
    int length,
    double common_logZ
):
    cdef dict last_exon_dict = {}
    cdef int used5, used3, lastSymbol, last3Pos, start
    cdef double alpha_score, tail, val, prob
    cdef tuple state

    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]

        if (used5 + used3) == 0:
            continue

        if lastSymbol != 0:
            continue
       
        start = last3Pos if last3Pos >= 0 else 0

        if state in B_current:
            tail = B_current[state]

            val = alpha_score + tail - common_logZ
            if val == float('-inf') or math.isnan(val):
                continue
            prob = math.exp(val)
            key = (start, 0)
            if key in last_exon_dict:
                last_exon_dict[key] += prob
            else:
                last_exon_dict[key] = prob
        else:
            tail = 0.0

    if not last_exon_dict:
        last_exon_dict[(0, length-1)] = 1.0

    return last_exon_dict, common_logZ


##########################
#### Compute GCS_Intron ####
##########################

cpdef tuple GCS_Intron(
object sequences,
double pME,
double[:] pELF,
double[:] pIL,
double pEE,
double[:] pELM,
double pEO,
double[:] pELL,
double[:] emissions5,
double[:] emissions3,
int length,
int checkpoint_interval,
int top_k_parse
):
    cdef dict checkpoints = {}
    cdef dict F_current = {}
    cdef tuple init_state = (0, 0, 0, 1, -1, -1)
    F_current[init_state] = 0.0
    checkpoints[0] = F_current.copy()

    cdef int top_k = top_k_parse 
    cdef int pos


    for pos in range(0, length):
        F_current = forward_dp_step(
            F_current, pos, sequences, length,
            pME, pELF, pIL, pEE, pELM,
            emissions5, emissions3,
            top_k
        )
        if ((pos + 1) % checkpoint_interval == 0) or (pos + 1 == length):
            checkpoints[pos + 1] = F_current.copy()


    cdef dict B_current = {}
    cdef int used5, used3, lastSymbol, last3Pos, ell_index
    cdef double alpha_score, tail
    for state, alpha_score in F_current.items():
        used5 = state[0]
        used3 = state[1]
        lastSymbol = state[2]
        last3Pos = state[5]
    
        if lastSymbol == 0 and (used5 == used3) and ((used5 + used3) > 0):
            tail = 0.0
            ell_index = (length - last3Pos) - 2
            if ell_index >= 0 and ell_index < pELL.shape[0]:
                tail += pEO + pELL[ell_index]
            B_current[state] = tail


    cdef list terminal_logs = []
    for state, alpha_score in F_current.items():
        if state in B_current:
            if (not math.isnan(alpha_score)) and (not math.isnan(B_current[state])):
                terminal_logs.append(alpha_score + B_current[state])
    cdef double logZ = float('-inf')
    if terminal_logs:
        logZ = logsumexp(terminal_logs)

    cdef dict intron_dict = {}
    cdef list ckpt_positions = sorted(checkpoints.keys())
    cdef dict B_next_segment = B_current
    cdef int i, seg_start, seg_end, seg_len, j, global_pos
    cdef list seg_F
    cdef list seg_B
    cdef dict B_seg
    cdef list contributions
    cdef tuple new_state_tuple, new_state
    cdef double new_log_score
    cdef double b_val, val, prob
    cdef list allowed_symbols
    cdef double alpha_val
    cdef int sym
    cdef int a_pos 
    cdef int b_pos 
    for i in range(len(ckpt_positions) - 1, 0, -1):
        seg_end = ckpt_positions[i]
        seg_start = ckpt_positions[i - 1]
        seg_len = seg_end - seg_start

        # 區段 forward
        seg_F = [None]*(seg_len+1)
        seg_F[0] = checkpoints[seg_start].copy()

        for j in range(seg_len):
            global_pos = seg_start + j
            seg_F[j+1] = forward_dp_step(
                seg_F[j], global_pos, sequences, length,
                pME, pELF, pIL, pEE, pELM,
                emissions5, emissions3,
                top_k)

        seg_B = [None]*(seg_len+1)
        seg_B[seg_len] = B_next_segment

       
        for j in range(seg_len-1, -1, -1):
            global_pos = seg_start + j
            B_seg = {}

            if global_pos == 0 or global_pos == length-1:
                allowed_symbols = [0]
            else:
                allowed_symbols = [0, 5, 3]

            for state, alpha_val in seg_F[j].items():
                if alpha_val == float('-inf') or math.isnan(alpha_val):
                    continue

                contributions = []
                for sym in allowed_symbols:
                    new_state_tuple = transition_dp(
                        state, alpha_val, global_pos, sym,
                        sequences, length,
                        pME, pELF, pIL, pEE, pELM,
                        emissions5, emissions3
                    )
                    if new_state_tuple is None:
                        continue
                    new_state, new_log_score = new_state_tuple
                    if new_state in seg_B[j+1]:
                        contributions.append(new_log_score - alpha_val + seg_B[j+1][new_state])

                if contributions:
                    B_seg[state] = logsumexp(contributions)

            seg_B[j] = B_seg
            
            for state, alpha_val in seg_F[j].items():
                if state in B_seg:
                    b_val = B_seg[state]
                    if (not math.isnan(alpha_val)) and (not math.isnan(b_val)):
                        val = alpha_val + b_val - logZ
                        if val != float('-inf') and (not math.isnan(val)):
                            prob = math.exp(val)

                            if state[2] == 3:
                                a_pos = state[4]
                                b_pos = global_pos
                                if a_pos >= 0:
                                    if (a_pos, b_pos) in intron_dict:
                                        intron_dict[(a_pos, b_pos)] += prob
                                    else:
                                        intron_dict[(a_pos, b_pos)] = prob

        B_next_segment = seg_B[0]

    return intron_dict, logZ
